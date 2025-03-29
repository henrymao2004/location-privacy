import tensorflow as tf
import keras
import numpy as np
import random
from tensorflow.keras import layers
import tensorflow_probability as tfp
import os
import json
import wandb
from tensorflow.keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

from losses import d_bce_loss, trajLoss, TrajLossLayer, CustomTrajLoss, compute_advantage, compute_returns, compute_trajectory_ratio, compute_entropy_loss

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

class RL_Enhanced_Transformer_TrajGAN():
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        self.keys = keys
        self.vocab_size = vocab_size
        
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        
        self.x_train = None
        
        # RL parameters
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.c1 = 0.5  # value function coefficient (reduced)
        self.c2 = 0.01  # entropy coefficient
        self.ppo_epochs = 4  # Number of PPO epochs per batch
        
        # Load or initialize TUL classifier for privacy rewards
        self.tul_classifier = self.load_tul_classifier()
        
        # Reward balance parameters (alpha, beta, gamma as per the paper)
        self.alpha_0 = 0.4  # Initial privacy weight
        self.beta_0 = 0.4     # Initial utility weight
        self.gamma_0 = 0.2    # Initial adversarial weight
        
        # Current adaptive weights (will be updated during training)
        self.alpha_t = self.alpha_0  
        self.beta_t = self.beta_0
        self.gamma_t = self.gamma_0
        
        # Target metrics for adaptive balancing
        self.acc_at_1_target = 0.2  # Target identification accuracy (lower is better for privacy)
        self.fid_target = 0.5       # Target FID score for utility (lower is better)
        self.fid_max = 5.0          # Maximum expected FID score
            
        # Utility component weights
        self.w1 = 0.4    # Spatial loss weight (d_spatial)
        self.w2 = 0.3    # Temporal loss weight (d_temporal)
        self.w3 = 0.3    # Semantic/category loss weight (d_semantic)
        
        # Define optimizers with reduced learning rates and gradient clipping
        self.actor_optimizer = Adam(0.0005, clipnorm=1.0)
        self.critic_optimizer = Adam(0.001, clipnorm=1.0)
        self.discriminator_optimizer = Adam(0.001, clipnorm=1.0)

        # Build networks
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.discriminator = self.build_discriminator()
        
        # Compile models
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer)
        self.critic.compile(loss='mse', optimizer=self.critic_optimizer)
        
        # Combined model for training
        self.setup_combined_model()
        
        # Add wandb related attributes
        self.wandb = None
        self.best_reward = float('-inf')
        self.best_g_loss = float('inf')
        self.best_d_loss = float('inf')
        
        # Tracking metrics for adaptive reward balancing
        self.current_acc_at_1 = 0.0
        self.current_fid = 0.0

    def set_wandb(self, wandb_instance):
        """Set wandb instance for logging."""
        self.wandb = wandb_instance

    def get_config(self):
        """Return the configuration of the model for serialization."""
        return {
            "latent_dim": self.latent_dim,
            "keys": self.keys,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "lat_centroid": self.lat_centroid,
            "lon_centroid": self.lon_centroid,
            "scale_factor": self.scale_factor,
            # RL parameters
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "c1": self.c1,
            "c2": self.c2,
            # Reward weights
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
            "gamma_0": self.gamma_0,
            # Adaptive targets
            "acc_at_1_target": self.acc_at_1_target,
            "fid_target": self.fid_target,
            "fid_max": self.fid_max,
            # Utility component weights
            "w1": self.w1,
            "w2": self.w2,
            "w3": self.w3,
        }
        
    @classmethod
    def from_config(cls, config):
        """Create a model from its config."""
        return cls(**config)

    @classmethod
    def from_saved_checkpoint(cls, epoch, checkpoint_dir='results'):
        """Load a model from saved checkpoints.
        
        Args:
            epoch: The epoch number of the checkpoint to load
            checkpoint_dir: Directory where checkpoints are saved
            
        Returns:
            An instance of the model with loaded weights
        """
        # Load model config
        try:
            with open(f'{checkpoint_dir}/model_config_{epoch}.json', 'r') as f:
                config = json.load(f)
            
            # Create model instance
            model = cls(**config)
            
            # Try to load saved weights
            model.generator.load_weights(f'{checkpoint_dir}/generator_{epoch}.weights.h5')
            model.discriminator.load_weights(f'{checkpoint_dir}/discriminator_{epoch}.weights.h5')
            model.critic.load_weights(f'{checkpoint_dir}/critic_{epoch}.weights.h5')
            
            print(f"Successfully loaded model from epoch {epoch}")
            return model
            
        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            raise

    def build_generator(self):
        # Input Layer
        inputs = []
        embeddings = []
        
        # Noise input
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        mask = Input(shape=(self.max_length, 1), name='input_mask')
        
        # Embedding layers for each feature
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Add noise input to the inputs list
        inputs.append(noise)
        
        # Add noise embedding
        noise_repeated = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, self.max_length, 1]))(noise)
        embeddings.append(noise_repeated)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(100, activation='relu')(concat_input)  # Project to embed_dim=100
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(concat_input, training=True)
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(x, training=True)
        
        # Output layers
        outputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                output_mask = Lambda(lambda x: x)(mask)
                outputs.append(output_mask)
            elif key == 'lat_lon':
                output = TimeDistributed(Dense(2, activation='tanh'), name='output_latlon')(x)
                scale_factor = self.scale_factor
                output_stratched = Lambda(lambda x: x * scale_factor)(output)
                outputs.append(output_stratched)
            else:
                output = TimeDistributed(Dense(self.vocab_size[key], activation='softmax'), 
                                       name='output_' + key)(x)
                outputs.append(output)
        
        return Model(inputs=inputs, outputs=outputs)

    def build_critic(self):
        # Input Layer
        inputs = []
        embeddings = []
        
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(100, activation='relu')(concat_input)  # Project to embed_dim=100
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(concat_input, training=True)
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(x, training=True)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Value head
        value = Dense(1)(x)
        
        return Model(inputs=inputs, outputs=value)

    def build_discriminator(self):
        # Similar to original discriminator but with Transformer blocks
        inputs = []
        embeddings = []
        
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=100, activation='relu', use_bias=True,  # Changed to 100 to match embed_dim
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(100, activation='relu')(concat_input)  # Project to embed_dim=100
        
        # Transformer blocks
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(concat_input, training=True)
        x = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)(x, training=True)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Output
        sigmoid = Dense(1, activation='sigmoid')(x)
        
        return Model(inputs=inputs, outputs=sigmoid)

    def setup_combined_model(self):
        # Generator inputs
        inputs = []
        mask = Input(shape=(self.max_length, 1), name='input_mask')
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        
        # Create inputs in the same order as build_generator
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
            inputs.append(i)
        
        # Add noise as the last input (to match the generator's input order)
        inputs.append(noise)
        
        # Generate trajectories
        gen_trajs = self.generator(inputs)
        
        # Discriminator predictions
        pred = self.discriminator(gen_trajs[:4])
        
        # Create the combined model
        self.combined = Model(inputs, pred)
        
        # Create a custom loss instance for trajectory optimization
        self.traj_loss = CustomTrajLoss()
        # Store input and generator output references
        self.input_tensors = inputs
        self.generated_trajectories = gen_trajs
        
        # Compile the model with the custom loss
        self.combined.compile(loss=self.traj_loss, optimizer=self.actor_optimizer)
        
        # Store the generator outputs for later use in reward computation
        self.gen_trajs_symbolic = gen_trajs

    def compute_rewards(self, real_trajs, gen_trajs, tul_classifier):
        """Compute the three-part reward function as described in the paper."""
        # Cast inputs to float32 for consistent typing
        gen_trajs = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
        real_trajs = [tf.cast(tensor, tf.float32) for tensor in real_trajs]
        
        batch_size = gen_trajs[0].shape[0]
        
        # 1. Adversarial Reward - measures realism based on discriminator output
        d_pred = self.discriminator.predict(gen_trajs[:4])
        d_pred = tf.cast(d_pred, tf.float32)
        d_pred = tf.clip_by_value(d_pred, 1e-7, 1.0 - 1e-7)
        r_adv = tf.math.log(d_pred)
        
        # 2. Utility Preservation Reward - measures statistical similarity
        # 2.1 Spatial loss - Use simpler L2 distance instead of Haversine to avoid NaN issues
        spatial_loss = tf.reduce_mean(tf.square(gen_trajs[0] - real_trajs[0]), axis=[1, 2])
        spatial_loss = tf.cast(spatial_loss, tf.float32)
        spatial_loss = tf.clip_by_value(spatial_loss, 0.0, 10.0)
        
        # 2.2 Temporal loss - Use simple cross-entropy with safe clipping
        gen_trajs_day_clipped = tf.clip_by_value(gen_trajs[2], 1e-7, 1.0 - 1e-7)
        real_trajs_day_clipped = tf.clip_by_value(real_trajs[2], 1e-7, 1.0 - 1e-7)
        
        gen_trajs_hour_clipped = tf.clip_by_value(gen_trajs[3], 1e-7, 1.0 - 1e-7)
        real_trajs_hour_clipped = tf.clip_by_value(real_trajs[3], 1e-7, 1.0 - 1e-7)
        
        # Simply use cross-entropy
        day_pattern_loss = -tf.reduce_sum(real_trajs_day_clipped * tf.math.log(gen_trajs_day_clipped), axis=[1, 2])
        hour_pattern_loss = -tf.reduce_sum(real_trajs_hour_clipped * tf.math.log(gen_trajs_hour_clipped), axis=[1, 2])
        
        # Combined temporal loss
        temp_loss = day_pattern_loss + hour_pattern_loss
        temp_loss = tf.cast(temp_loss, tf.float32)
        temp_loss = tf.clip_by_value(temp_loss, 0.0, 10.0)
        
        # 2.3 Categorical loss - Use simple cross-entropy instead of JS divergence
        gen_trajs_cat_clipped = tf.clip_by_value(gen_trajs[1], 1e-7, 1.0 - 1e-7)
        real_trajs_cat_clipped = tf.clip_by_value(real_trajs[1], 1e-7, 1.0 - 1e-7)
        
        # Simplify to cross-entropy for stability
        cat_loss = -tf.reduce_sum(real_trajs_cat_clipped * tf.math.log(gen_trajs_cat_clipped), axis=[1, 2])
        cat_loss = tf.cast(cat_loss, tf.float32)
        cat_loss = tf.clip_by_value(cat_loss, 0.0, 10.0)
        
        # Combined utility components with appropriate weights
        r_util = -(self.w1 * spatial_loss + self.w2 * temp_loss + self.w3 * cat_loss)
        
        # 3. Privacy preservation reward using TUL classifier
        try:
            # Format inputs for TUL classifier as before
            day_indices = tf.cast(tf.argmax(gen_trajs[2], axis=-1), tf.int32)
            day_indices = tf.clip_by_value(day_indices, 0, 6)
            
            hour_indices = tf.cast(tf.argmax(gen_trajs[3], axis=-1), tf.int32)
            hour_indices = tf.clip_by_value(hour_indices, 0, 23)
            
            category_indices = tf.cast(tf.argmax(gen_trajs[1], axis=-1), tf.int32)
            category_indices = tf.clip_by_value(category_indices, 0, 9)
            
            # Format lat_lon for TUL model
            lat_lon_padded = tf.pad(gen_trajs[0], [[0, 0], [0, 0], [0, 38]])
            
            # Get TUL predictions
            tul_preds = tul_classifier([day_indices, hour_indices, category_indices, lat_lon_padded])
            
            # Extract probability for correct user identification
            num_users = tul_preds.shape[1]
            user_indices = np.arange(batch_size) % num_users
            batch_indices = tf.range(batch_size, dtype=tf.int32)
            indices = tf.stack([batch_indices, tf.cast(user_indices, tf.int32)], axis=1)
            user_probs = tf.gather_nd(tul_preds, indices)
            
            # Calculate top-1 identification accuracy for adaptive balancing
            sorted_indices = tf.argsort(tul_preds, axis=1, direction='DESCENDING')
            top1_indices = sorted_indices[:, 0]
            correct_predictions = tf.cast(tf.equal(top1_indices, tf.cast(user_indices, tf.int32)), tf.float32)
            self.current_acc_at_1 = tf.reduce_mean(correct_predictions).numpy()
            
            # Privacy reward: -log(p_TUL(u_i|T_i))
            user_probs = tf.clip_by_value(user_probs, 1e-7, 1.0 - 1e-7)
            r_priv = -tf.math.log(user_probs)
            
        except Exception as e:
            print(f"Error computing privacy reward: {e}")
            # Use placeholder privacy reward
            r_priv = tf.zeros((batch_size,), dtype=tf.float32)
            self.current_acc_at_1 = 0.5  # Default value
        
        # Calculate approximate FID score for utility
        self.current_fid = tf.reduce_mean(spatial_loss + temp_loss + cat_loss).numpy()
        if np.isnan(self.current_fid):
            self.current_fid = 2.0  # Use default if NaN
        
        # Fixed weights if adaptive balancing isn't working
        if np.isnan(self.current_acc_at_1) or np.isnan(self.current_fid):
            self.alpha_t = tf.constant(self.alpha_0, dtype=tf.float32)
            self.beta_t = tf.constant(self.beta_0, dtype=tf.float32)
            self.gamma_t = tf.constant(self.gamma_0, dtype=tf.float32)
        else:
            # Apply adaptive reward balancing based on privacy and utility metrics
            # Adjust alpha (privacy weight) based on re-identification accuracy
            self.alpha_t = self.alpha_0 * tf.minimum(
                1.0, 
                tf.cast(self.current_acc_at_1 / self.acc_at_1_target, tf.float32)
            )
            
            # Adjust beta (utility weight) based on FID score
            self.beta_t = self.beta_0 * tf.maximum(
                0.0,
                1.0 - tf.cast((self.current_fid - self.fid_target) / self.fid_max, tf.float32)
            )
            
            # Adjust gamma to normalize weights (alpha + beta + gamma = 1)
            self.gamma_t = 1.0 - (self.alpha_t + self.beta_t)
            
            # Ensure weights are valid
            self.alpha_t = tf.clip_by_value(self.alpha_t, 0.05, 0.95)
            self.beta_t = tf.clip_by_value(self.beta_t, 0.05, 0.95)
            self.gamma_t = tf.clip_by_value(self.gamma_t, 0.05, 0.95)
            
            # Normalize weights to sum to 1
            total = self.alpha_t + self.beta_t + self.gamma_t
            self.alpha_t = self.alpha_t / total
            self.beta_t = self.beta_t / total
            self.gamma_t = self.gamma_t / total
        
        # Print current adaptive weights
        print(f"Adaptive weights - α: {self.alpha_t.numpy():.3f}, β: {self.beta_t.numpy():.3f}, γ: {self.gamma_t.numpy():.3f}")
        print(f"Current metrics - ACC@1: {self.current_acc_at_1:.3f}, FID: {self.current_fid:.3f}")
        
        # Ensure r_adv, r_util, and r_priv have appropriate shapes
        r_adv = tf.reshape(r_adv, [-1])
        r_util = tf.reshape(r_util, [-1])
        r_priv = tf.reshape(r_priv, [-1])
        
        # Compute the combined reward with adaptive weights
        combined_rewards = self.gamma_t * r_adv + self.beta_t * r_util + self.alpha_t * r_priv
        
        # Ensure the rewards have shape [batch_size, 1]
        rewards = tf.reshape(combined_rewards, [batch_size, 1])
        
        # Replace any NaN values with zeros
        rewards = tf.where(tf.math.is_nan(rewards), tf.zeros_like(rewards), rewards)
        
        # Normalize rewards for training stability
        rewards_mean = tf.reduce_mean(rewards)
        rewards_std = tf.math.reduce_std(rewards) + 1e-8
        
        # Only normalize if mean and std are not NaN
        if not (tf.math.is_nan(rewards_mean) or tf.math.is_nan(rewards_std)):
            rewards = (rewards - rewards_mean) / rewards_std
        
        # Clip rewards to reasonable range to prevent training instability
        rewards = tf.clip_by_value(rewards, -5.0, 5.0)
        
        # Final check for NaN values
        rewards = tf.where(tf.math.is_nan(rewards), tf.zeros_like(rewards), rewards)
        
        # Debug print to check rewards shape and values
        print(f"Rewards shape: {rewards.shape}, min: {tf.reduce_min(rewards)}, max: {tf.reduce_max(rewards)}, mean: {tf.reduce_mean(rewards)}")
        
        return rewards

    def preprocess_batch(self, batch):
        """Preprocess a batch of trajectories to ensure consistent tensor format.
        
        Args:
            batch: List of trajectory features, potentially with object dtype
            
        Returns:
            List of preprocessed tensors in the format expected by the model
        """
        processed_batch = []
        
        for i, feature in enumerate(batch):
            feature_name = self.keys[i] if i < len(self.keys) else f"feature_{i}"
            
            try:
                if feature.dtype == 'object':
                    print(f"Converting object array for feature {feature_name}")
                    # For object arrays, we need to pad to fixed size
                    sample_size = len(feature)
                    
                    # Check the dimensionality of the first non-None element
                    sample_item = None
                    for item in feature:
                        if item is not None and hasattr(item, 'shape'):
                            sample_item = item
                            break
                    
                    if sample_item is None:
                        print(f"Warning: Could not find valid sample for feature {feature_name}")
                        # Create placeholder with reasonable dimensions
                        placeholder = np.zeros((sample_size, self.max_length, 1), dtype=np.float32)
                        processed_batch.append(placeholder)
                        continue
                    
                    # Get the feature dimension from the sample
                    feat_dim = sample_item.shape[-1] if len(sample_item.shape) > 1 else 1
                    
                    # Create a padded array
                    padded = np.zeros((sample_size, self.max_length, feat_dim), dtype=np.float32)
                    
                    # Fill in the data
                    for j, traj in enumerate(feature):
                        if traj is not None and hasattr(traj, 'shape'):
                            # Only copy up to max_length
                            actual_length = min(traj.shape[0], self.max_length)
                            if len(traj.shape) > 1:
                                # Multi-dimensional feature
                                padded[j, :actual_length, :] = traj[:actual_length]
                            else:
                                # One-dimensional feature
                                padded[j, :actual_length, 0] = traj[:actual_length]
                    
                    processed_batch.append(padded)
                else:
                    # Already a proper tensor
                    processed_batch.append(tf.cast(feature, tf.float32))
            
            except Exception as e:
                print(f"Error preprocessing feature {feature_name}: {e}")
                # Create a placeholder with zeros
                placeholder = np.zeros((len(batch[0]), self.max_length, 1), dtype=np.float32)
                processed_batch.append(placeholder)
        
        return processed_batch

    def train_step(self, real_trajs, batch_size=256):
        """Modified train_step to include tracking of RL agent metrics."""
        # First preprocess the batch to ensure consistent tensor format
        real_trajs = self.preprocess_batch(real_trajs)
        
        # Generate trajectories
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        gen_trajs = self.generator.predict([*real_trajs, noise])
        
        # Ensure consistent data types
        real_trajs = [tf.cast(tensor, tf.float32) for tensor in real_trajs]
        gen_trajs = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
        
        # Update the custom loss with the current real and generated trajectories
        self.traj_loss.set_trajectories(real_trajs, gen_trajs)
        
        # Compute full rewards using the adaptive reward function
        rewards = self.compute_rewards(real_trajs, gen_trajs, self.tul_classifier)
        
        # Compute advantages and returns for PPO
        values = self.critic.predict(real_trajs[:4])
        values = tf.cast(values, tf.float32)
        advantages = compute_advantage(rewards, values, self.gamma, self.gae_lambda)
        returns = compute_returns(rewards, self.gamma)
        
        # Ensure returns has the same batch size as what the critic expects
        if returns.shape[0] != batch_size:
            print(f"Warning: Reshaping returns from {returns.shape} to [{batch_size}, 1]")
            returns = tf.reshape(returns, [batch_size, 1])
        
        # Update critic (value function) using returns
        c_loss = self.critic.train_on_batch(real_trajs[:4], returns)
        
        # Update discriminator - real samples -> 1, generated samples -> 0
        d_loss_real = self.discriminator.train_on_batch(
            real_trajs[:4],
            np.ones((batch_size, 1))
        )
        d_loss_fake = self.discriminator.train_on_batch(
            gen_trajs[:4],
            np.zeros((batch_size, 1))
        )
        
        # Update generator (actor) using advantages from PPO
        g_loss = self.update_actor(real_trajs, gen_trajs, advantages)
        
        # Get mean reward and other metrics for tracking
        mean_reward = tf.reduce_mean(rewards).numpy()
        
        # Calculate entropy of trajectory distributions for exploration monitoring
        entropy = self._calculate_trajectory_entropy(gen_trajs)
        
        # Return metrics including reward and RL-specific metrics
        return {
            "d_loss_real": d_loss_real, 
            "d_loss_fake": d_loss_fake, 
            "g_loss": g_loss, 
            "c_loss": c_loss,
            "reward": mean_reward,
            "entropy": entropy,
            "acc_at_1": self.current_acc_at_1,
            "fid": self.current_fid,
            "alpha_t": self.alpha_t.numpy(),
            "beta_t": self.beta_t.numpy(),
            "gamma_t": self.gamma_t.numpy()
        }

    def _calculate_trajectory_entropy(self, gen_trajs):
        """Calculate entropy of generated trajectories to monitor exploration vs exploitation.
        
        Higher entropy indicates more exploration, lower indicates exploitation.
        """
        # Calculate entropy for categorical outputs (category, day, hour)
        entropy = 0.0
        for i in range(1, 4):  # Categorical outputs (category, day, hour)
            # Get probability distributions
            probs = gen_trajs[i]
            # Clip probabilities to avoid log(0)
            probs = tf.clip_by_value(probs, 1e-7, 1.0 - 1e-7)
            # Calculate entropy: -sum(p * log(p))
            cat_entropy = -tf.reduce_sum(probs * tf.math.log(probs), axis=-1)
            # Average over timesteps and batch
            cat_entropy = tf.reduce_mean(cat_entropy)
            entropy += cat_entropy
            
        # For continuous outputs (coordinates), estimate entropy with Gaussian approximation
        # Estimate variance of coordinates as proxy for entropy
        coord_var = tf.math.reduce_variance(gen_trajs[0], axis=[0, 1])
        coord_entropy = tf.reduce_sum(tf.math.log(2 * np.pi * np.e * coord_var + 1e-7)) / 2
        
        # Combine all entropy components
        total_entropy = entropy + coord_entropy
        
        return total_entropy.numpy()

    def train(self, epochs=2000, batch_size=256, sample_interval=10, save_best=True, checkpoint_dir='results'):
        """Train the model with WandB tracking, best model checkpointing, and early stopping."""
        # Make sure the checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get the early stopping callback if attached
        early_stopping = getattr(self, 'early_stopping_callback', None)
        
        # Check training data and try to preprocess it
        if self.x_train is None:
            print("Error: No training data available.")
            return 0
            
        print(f"Starting training with {len(self.x_train[0])} trajectories")
        self.check_data_types()
        
        # Initialize tracking variables
        completed_epochs = 0
        
        # Training loop
        print(f"Starting training for {epochs} epochs with early stopping patience of "
              f"{early_stopping.patience if early_stopping else 'N/A'}")
        
        for epoch in range(epochs):
            try:
                # Get a batch of training data
                idx = np.random.randint(0, len(self.x_train[0]), batch_size)
                raw_batch = []
                
                # Extract batch for each feature
                for X in self.x_train:
                    if X.dtype == 'object':
                        # For object arrays, extract the selected indices
                        feature_batch = [X[i] for i in idx]
                        raw_batch.append(np.array(feature_batch, dtype='object'))
                    else:
                        # For regular arrays, do normal indexing
                        feature_batch = X[idx]
                        raw_batch.append(feature_batch)
                
                # Training step will handle preprocessing
                metrics = self.train_step(raw_batch, batch_size)
                
                # Log metrics to WandB
                if self.wandb:
                    wandb_metrics = {
                        "epoch": epoch,
                        "d_loss_real": metrics['d_loss_real'],
                        "d_loss_fake": metrics['d_loss_fake'],
                        "g_loss": metrics['g_loss'],
                        "c_loss": metrics['c_loss'],
                        "reward": metrics['reward'],
                        # New RL metrics
                        "entropy": metrics['entropy'],
                        "acc_at_1": metrics['acc_at_1'],
                        "fid": metrics['fid'],
                        # Adaptive weights
                        "alpha_t": metrics['alpha_t'],
                        "beta_t": metrics['beta_t'],
                        "gamma_t": metrics['gamma_t'],
                    }
                    
                    # Track and save the best model based on reward
                    if save_best and metrics['reward'] > self.best_reward:
                        self.best_reward = metrics['reward']
                        wandb_metrics["best_reward"] = self.best_reward
                        self.save_best_checkpoint(checkpoint_dir, f"best_reward_model")
                        print(f"New best reward: {self.best_reward:.4f} at epoch {epoch}")
                    
                    # Add early stopping metrics
                    if early_stopping:
                        improved = early_stopping.on_epoch_end(epoch, metrics)
                        wandb_metrics["es_wait_count"] = early_stopping.wait_count
                        wandb_metrics["es_best_epoch"] = early_stopping.best_epoch
                        
                        if improved:
                            print(f"Early stopping: improvement detected at epoch {epoch}")
                            
                        if early_stopping.should_stop:
                            print(f"\nEarly stopping triggered after {early_stopping.wait_count} epochs without improvement")
                            print(f"Best model was at epoch {early_stopping.best_epoch} with reward {early_stopping.best_reward:.4f}")
                            break
                    
                    # Log to wandb
                    self.wandb.log(wandb_metrics)
                
                # Print progress
                if epoch % sample_interval == 0:
                    print(f"[Epoch {epoch}/{epochs}] "
                          f"[D loss: {(metrics['d_loss_real'] + metrics['d_loss_fake'])/2:.4f}] "
                          f"[G loss: {metrics['g_loss']:.4f}] "
                          f"[Reward: {metrics['reward']:.4f}] "
                          f"[ACC@1: {metrics['acc_at_1']:.4f}] "
                          f"[FID: {metrics['fid']:.4f}]")
                    
                    # Generate visual samples for wandb
                    if self.wandb:
                        self.sample_trajectories_for_wandb(epoch)
                    
                    # Save model checkpoint
                    if epoch > 0 and epoch % (sample_interval * 5) == 0:
                        self.save_checkpoint(epoch)
                
                completed_epochs = epoch + 1
                
            except Exception as e:
                print(f"Error during epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing to next epoch...")
                continue
        
        # Final report
        print(f"Training completed after {completed_epochs} epochs")
        if early_stopping:
            print(f"Best model was found at epoch {early_stopping.best_epoch} with reward {early_stopping.best_reward:.4f}")
        
        return completed_epochs

    def save_checkpoint(self, epoch):
        # Make sure the results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Save model weights
        try:
            self.generator.save_weights(f'results/generator_{epoch}.weights.h5')
            self.discriminator.save_weights(f'results/discriminator_{epoch}.weights.h5')
            self.critic.save_weights(f'results/critic_{epoch}.weights.h5')
            print(f"Model weights saved for epoch {epoch}")
        except Exception as e:
            print(f"Warning: Could not save weights for epoch {epoch}: {e}")
        
        # Now try to save the full models with architecture
        try:
            # Save the Keras models
            self.generator.save(f'results/generator_architecture_{epoch}.keras')
            self.discriminator.save(f'results/discriminator_architecture_{epoch}.keras')
            self.critic.save(f'results/critic_architecture_{epoch}.keras')
            print(f"Model architectures saved for epoch {epoch}")
            
            # Also save the main model's configuration
            with open(f'results/model_config_{epoch}.json', 'w') as f:
                json.dump(self.get_config(), f, indent=4)
            
        except Exception as e:
            print(f"Warning: Could not save full model architectures for epoch {epoch}: {e}")
            print("Only weights were saved. You'll need to recreate the model structure to load them.")

    def save_best_checkpoint(self, checkpoint_dir, name_prefix):
        """Save the current best model based on some metric."""
        try:
            # Save model weights
            self.generator.save_weights(f'{checkpoint_dir}/{name_prefix}_generator.weights.h5')
            self.discriminator.save_weights(f'{checkpoint_dir}/{name_prefix}_discriminator.weights.h5')
            self.critic.save_weights(f'{checkpoint_dir}/{name_prefix}_critic.weights.h5')
            
            # Save model architecture
            self.generator.save(f'{checkpoint_dir}/{name_prefix}_generator.keras')
            self.discriminator.save(f'{checkpoint_dir}/{name_prefix}_discriminator.keras')
            self.critic.save(f'{checkpoint_dir}/{name_prefix}_critic.keras')
            
            # Save model config
            with open(f'{checkpoint_dir}/{name_prefix}_config.json', 'w') as f:
                json.dump(self.get_config(), f, indent=4)
                
            print(f"Saved best model as {name_prefix}")
        except Exception as e:
            print(f"Error saving best model {name_prefix}: {e}")

    def update_actor(self, states, actions, advantages):
        """Update generator using a simplified policy gradient approach with safeguards against NaNs."""
        # Get a single batch of data with random noise for the generator
        batch_size = states[0].shape[0]
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Prepare inputs for the generator
        all_inputs = [*states, noise]
        
        # Use ones as targets for the discriminator output (we want to fool the discriminator)
        targets = np.ones((batch_size, 1))
        
        # Process advantages safely, ensuring no NaN values
        try:
            # First ensure advantages is a tensor with float32 type
            advantages = tf.cast(advantages, tf.float32)
            
            # Check for and replace NaN values
            advantages = tf.where(tf.math.is_nan(advantages), tf.zeros_like(advantages), advantages)
            
            # Clip advantages to reasonable range before any other operations
            advantages = tf.clip_by_value(advantages, -10.0, 10.0)
            
            # Convert to numpy and flatten
            advantages_np = advantages.numpy().flatten()
            
            # Scale advantages to be positive (sample_weight should be positive)
            min_adv = np.min(advantages_np)
            if min_adv < 0:
                advantages_np = advantages_np - min_adv + 1e-3
            
            # Normalize to reasonable values (0 to 1 range)
            max_adv = np.max(advantages_np)
            if max_adv > 0:
                advantages_np = advantages_np / max_adv
            else:
                # If all advantages are non-positive, use uniform weights
                advantages_np = np.ones(batch_size) * 0.5
                
            # Ensure it has the right shape
            if len(advantages_np) != batch_size:
                print(f"Warning: Reshaping advantages from {len(advantages_np)} to {batch_size}")
                advantages_np = np.ones(batch_size) * 0.5
                
            # Final safety check - replace any NaN or inf values
            advantages_np = np.nan_to_num(advantages_np, nan=0.5, posinf=1.0, neginf=0.0)
                
        except Exception as e:
            print(f"Error processing advantages: {e}")
            # Fallback to uniform weights
            advantages_np = np.ones(batch_size) * 0.5
            
        # Print statistics about advantage values used for training
        print(f"Advantage stats - min: {np.min(advantages_np):.4f}, max: {np.max(advantages_np):.4f}, " +
              f"mean: {np.mean(advantages_np):.4f}, std: {np.std(advantages_np):.4f}")
        
        # Train the combined model with sample weights from advantages
        try:
            loss = self.combined.train_on_batch(
                all_inputs, 
                targets,
                sample_weight=advantages_np
            )
            
            # If loss is NaN, retry with uniform weights
            if np.isnan(loss) or np.isinf(loss):
                print(f"Warning: Loss is NaN or Inf ({loss}). Retrying with uniform weights.")
                loss = self.combined.train_on_batch(
                    all_inputs, 
                    targets,
                    sample_weight=np.ones(batch_size) * 0.5
                )
            
            # If loss is extremely high, clip it for reporting purposes
            if loss > 10000:
                print(f"Warning: Loss is very high ({loss:.2f}), consider reducing learning rate")
                loss = min(loss, 10000.0)
                
            return loss
            
        except Exception as e:
            print(f"Error during actor update: {e}")
            # Return a placeholder loss value
            return 0.0

    def compute_trajectory_ratio(self, new_predictions, old_predictions):
        """Compute the PPO policy ratio between new and old policies for trajectory data.
        
        For trajectories, we take the product of point-wise prediction ratios,
        which is equivalent to the ratio of trajectory probabilities under each policy.
        """
        # Initialize ratio as ones
        ratio = tf.ones(shape=(len(new_predictions[0]), 1))
        
        # For each categorical output (category, day, hour), compute ratio
        for i in range(1, 4):  # Categorical outputs
            # Get probabilities under new policy
            new_probs = new_predictions[i]
            
            # Get probabilities under old policy
            old_probs = old_predictions[i]
            
            # Compute the ratio of probabilities (with small epsilon to avoid division by zero)
            point_ratio = new_probs / (old_probs + 1e-10)
            
            # Reduce along the category dimension to get per-point ratio
            point_ratio = tf.reduce_sum(point_ratio, axis=-1, keepdims=True)
            
            # Multiply with current ratio
            ratio = ratio * point_ratio
        
        # For continuous outputs (coordinates), use normal distribution likelihood ratio
        # This is simplified - in practice you'd need proper distribution modeling
        coord_ratio = tf.exp(-0.5 * tf.reduce_sum(tf.square(new_predictions[0] - old_predictions[0]), axis=-1, keepdims=True))
        ratio = ratio * coord_ratio
        
        # Mask out padding
        mask = new_predictions[4]
        ratio = ratio * mask
        
        # Average over the trajectory length
        ratio = tf.reduce_mean(ratio, axis=1)
        
        return ratio

    def load_tul_classifier(self):
        """Load the pre-trained Trajectory-User Linking classifier.
        
        Returns:
            A trained model that predicts user IDs from trajectories.
        """
        try:
            # Import MARC class and initialize it
            from MARC.marc import MARC
            
            # Create and initialize the MARC model
            marc_model = MARC()
            marc_model.build_model()
            
            # Load pre-trained weights
            marc_model.load_weights('MARC/MARC_Weight.h5')
            
            print("Loaded pre-trained MARC TUL classifier")
            return marc_model
            
        except Exception as e:
            print(f"Error loading MARC model: {e}")
            print("Creating a fallback TUL classifier model")
            
            # If MARC model loading fails, create a fallback model that matches MARC's input format
            # MARC expects 4 inputs: day, hour, category (all as indices), and lat_lon (with shape 144,40)
            
            # Create input layers with the same names and shapes as MARC
            input_day = Input(shape=(144,), dtype='int32', name='input_day')
            input_hour = Input(shape=(144,), dtype='int32', name='input_hour')
            input_category = Input(shape=(144,), dtype='int32', name='input_category')
            input_lat_lon = Input(shape=(144, 40), name='input_lat_lon')
            
            # Create embeddings like MARC
            emb_day = Embedding(input_dim=7, output_dim=32, input_length=144)(input_day)
            emb_hour = Embedding(input_dim=24, output_dim=32, input_length=144)(input_hour)
            emb_category = Embedding(input_dim=10, output_dim=32, input_length=144)(input_category)
            
            # Process lat_lon
            lat_lon_dense = Dense(32, activation='relu')(input_lat_lon)
            
            # Concatenate all embeddings
            concat = Concatenate(axis=2)([emb_day, emb_hour, emb_category, lat_lon_dense])
            
            # LSTM layer for sequence processing
            lstm_out = LSTM(64, return_sequences=False)(concat)
            
            # Dense layers
            dense1 = Dense(128, activation='relu')(lstm_out)
            
            # Output layer - assuming 100 users for classification
            # (We'll adjust this if needed based on the actual dataset)
            output = Dense(193, activation='softmax')(dense1)
            
            # Create model
            fallback_model = Model(
                inputs=[input_day, input_hour, input_category, input_lat_lon],
                outputs=output
            )
            
            fallback_model.compile(
                loss='sparse_categorical_crossentropy',
                optimizer=Adam(0.001),
                metrics=['accuracy']
            )
            
            print("Created fallback TUL classifier model with matching input format")
            
            # Since the fallback model uses Keras functional API, it already supports __call__
            return fallback_model
            
    def sample_trajectories_for_wandb(self, epoch):
        """Simplified method that only visualizes real trajectories without attempting to generate samples."""
        if not self.wandb:
            return
            
        try:
            # Sample a few real trajectories
            idx = np.random.randint(0, len(self.x_train[0]), 4)
            
            # Handle variable-length trajectories by padding each batch separately
            real_batch = []
            for X in self.x_train:
                if X.dtype == 'object':
                    # For object arrays with variable-length trajectories
                    samples = [X[i] for i in idx]
                    
                    # First, check if the first feature has coordinates
                    if len(samples) > 0 and isinstance(samples[0], np.ndarray) and samples[0].shape[-1] >= 2:
                        # This is a valid coordinate feature, save it separately for visualization
                        real_batch.append(samples)
                    else:
                        # For other features, just create a placeholder
                        # We won't attempt to visualize these
                        dummy = np.zeros((4, self.max_length, 1))
                        real_batch.append(dummy)
                else:
                    # For fixed-shape tensors
                    X_subset = X[idx]
                    real_batch.append(X_subset)
            
            # Create visualization directly from the coordinates
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i in range(min(4, len(real_batch[0]))):
                try:
                    # Plot real trajectory - safely extract coordinates
                    real_coords = real_batch[0][i]
                    if isinstance(real_coords, np.ndarray) and real_coords.shape[-1] >= 2:
                        axes[i].scatter(
                            real_coords[:, 0], 
                            real_coords[:, 1], 
                            c='blue', 
                            alpha=0.7, 
                            label='Real'
                        )
                    
                    # Set title and legend
                    axes[i].set_title(f'Trajectory Sample {i+1}')
                    axes[i].legend()
                    
                except Exception as e:
                    print(f"Error plotting trajectory {i+1}: {e}")
                    # Skip this trajectory if there's an error
                    continue
                
            plt.tight_layout()
            
            # Log to wandb
            self.wandb.log({f"trajectory_samples": self.wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"Error generating trajectory samples for wandb: {e}")
            import traceback
            traceback.print_exc()

    def check_data_types(self):
        """Diagnostic method to check data types and shapes in the training data."""
        if self.x_train is None:
            print("No training data loaded (x_train is None)")
            return
            
        print("\n=== Training Data Diagnostics ===")
        for i, x in enumerate(self.x_train):
            try:
                key_name = self.keys[i] if i < len(self.keys) else f"feature_{i}"
                print(f"Feature {i} ({key_name}):")
                print(f"  - Type: {type(x)}")
                print(f"  - Dtype: {x.dtype}")
                print(f"  - Shape: {x.shape if hasattr(x, 'shape') else 'unknown'}")
                
                # Check for NaN values
                if hasattr(x, 'dtype') and np.issubdtype(x.dtype, np.number):
                    nan_count = np.isnan(x).sum() if hasattr(x, 'sum') else "unknown"
                    print(f"  - NaN values: {nan_count}")
                
                # For object arrays, check the first element
                if x.dtype == 'object':
                    first_elem = x[0]
                    print(f"  - First element type: {type(first_elem)}")
                    print(f"  - First element shape: {first_elem.shape if hasattr(first_elem, 'shape') else 'unknown'}")
                
            except Exception as e:
                print(f"  - Error analyzing feature {i}: {e}")
        
        print("===========================\n")
        
    def set_training_data(self, x_train):
        """Set the training data and verify its format."""
        self.x_train = x_train
        self.check_data_types()
        
        # Try to convert object arrays to proper tensor format
        if self.x_train is not None:
            converted_data = []
            for i, x in enumerate(self.x_train):
                if x.dtype == 'object':
                    try:
                        # For object arrays, try to stack the elements into a proper tensor
                        stacked = np.stack(x)
                        converted_data.append(stacked)
                        print(f"Successfully converted feature {i} from object to tensor with shape {stacked.shape}")
                    except Exception as e:
                        # If stacking fails, keep the original
                        print(f"Could not convert feature {i}: {e}")
                        converted_data.append(x)
                else:
                    converted_data.append(x)
                    
            # Update x_train with converted data
            self.x_train = converted_data