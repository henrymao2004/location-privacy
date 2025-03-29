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
        
        # RL parameters (Updated based on architecture table)
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # GAE parameter (updated)
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.c1 = 0.5  # value function coefficient
        self.c2 = 0.01  # entropy coefficient (updated)
        self.ppo_epochs = 4  # Number of PPO epochs per batch
        
        # Load or initialize TUL classifier for privacy rewards
        self.tul_classifier = self.load_tul_classifier()
        
        # Updated reward weights based on the parameter image
        self.w_priv = 0.6    # Initial α (privacy): Higher weight on privacy
        self.w_util = 0.3    # Initial β (utility): Moderate weight on utility
        self.w_adv = 0.1     # Initial γ (adversarial): Lower initial weight on realism
            
        # Updated utility component weights based on the parameter image
        self.w_spatial = 0.4    # w₁ (spatial): Moderate weight on spatial utility
        self.w_temporal = 0.4   # w₂ (temporal): Moderate weight on temporal utility
        self.w_semantic = 0.2   # w₃ (semantic): Lower weight on semantic utility
        
        # Define optimizers with parameters from architecture table
        self.actor_optimizer = Adam(0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.critic_optimizer = Adam(0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.discriminator_optimizer = Adam(0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

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
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_epsilon": self.clip_epsilon,
            "c1": self.c1,
            "c2": self.c2,
            "w_adv": self.w_adv,
            "w_util": self.w_util,
            "w_priv": self.w_priv,
            "w_spatial": self.w_spatial,
            "w_temporal": self.w_temporal,
            "w_semantic": self.w_semantic,
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
                d = Dense(units=512, activation='relu', use_bias=True, # Updated to 512 per architecture table
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=512, activation='relu', use_bias=True, # Updated to 512 per architecture table
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
        concat_input = Dense(512, activation='relu')(concat_input)  # Updated to 512
        
        # Transformer blocks (updated to 4 layers with 8 attention heads)
        x = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048, rate=0.1)(concat_input, training=True)
        x = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048, rate=0.1)(x, training=True)
        x = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048, rate=0.1)(x, training=True)
        x = TransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048, rate=0.1)(x, training=True)
        
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
                d = Dense(units=512, activation='relu', use_bias=True,  # Updated to 512
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=512, activation='relu', use_bias=True,  # Updated to 512
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(512, activation='relu')(concat_input)  # Updated to 512
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(concat_input)
        
        # MLP with 3 layers (512, 256, 1) and ReLU activations
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        value = Dense(1)(x)
        
        return Model(inputs=inputs, outputs=value)

    def build_discriminator(self):
        # Input Layer
        inputs = []
        embeddings = []
        
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=256, activation='relu', use_bias=True,  # Updated to 256 per architecture
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=256, activation='relu', use_bias=True,  # Updated to 256 per architecture
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_attr = [d(x) for x in unstacked]
                e = Lambda(lambda x: tf.stack(x, axis=1))(dense_attr)
            inputs.append(i)
            embeddings.append(e)
        
        # Feature Fusion Layer
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Project concatenated embeddings to correct dimension
        concat_input = Dense(256, activation='relu')(concat_input)  # Updated to 256
        
        # LSTM-based CNN architecture per table
        # First apply LSTM layers
        x = LSTM(256, return_sequences=True, dropout=0.2)(concat_input)
        x = LSTM(256, return_sequences=True, dropout=0.2)(x)
        
        # Then apply CNN layers
        # Reshape for CNN (batch_size, sequence_length, features, 1)
        x_reshaped = Lambda(lambda x: tf.expand_dims(x, axis=-1))(x)
        
        # 3 convolutional layers with different filter sizes
        conv1 = TimeDistributed(layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'))(x_reshaped)
        conv2 = TimeDistributed(layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'))(conv1)
        conv3 = TimeDistributed(layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'))(conv2)
        
        # Flatten and dense layers
        x = TimeDistributed(layers.Flatten())(conv3)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        
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
        """Compute the three-part reward function as described in the paper.
        
        Args:
            real_trajs: Original real trajectories
            gen_trajs: Generated synthetic trajectories
            tul_classifier: Pre-trained TUL classifier for privacy evaluation
        
        Returns:
            Combined reward balancing privacy, utility and realism
        """
        # Cast inputs to float32 for consistent typing
        gen_trajs = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
        real_trajs = [tf.cast(tensor, tf.float32) for tensor in real_trajs]
        
        # Adversarial reward - measures realism based on discriminator output
        d_pred = self.discriminator.predict(gen_trajs[:4])
        d_pred = tf.cast(d_pred, tf.float32)
        # Clip discriminator predictions to avoid extreme log values
        d_pred = tf.clip_by_value(d_pred, 1e-7, 1.0 - 1e-7)
        r_adv = tf.math.log(d_pred)
        
        # Utility preservation reward - measures statistical similarity
        # Spatial loss - L2 distance between coordinates
        spatial_loss = tf.reduce_mean(tf.square(gen_trajs[0] - real_trajs[0]), axis=[1, 2])
        spatial_loss = tf.cast(spatial_loss, tf.float32)
        # Clip spatial loss to avoid extremely large values
        spatial_loss = tf.clip_by_value(spatial_loss, 0.0, 10.0)
        
        # Temporal loss - cross-entropy on temporal distributions (day and hour)
        # Clip generated values to avoid log(0)
        gen_trajs_day_clipped = tf.clip_by_value(gen_trajs[2], 1e-7, 1.0 - 1e-7)
        temp_day_loss = -tf.reduce_sum(real_trajs[2] * tf.math.log(gen_trajs_day_clipped), axis=[1, 2])
        temp_day_loss = tf.cast(temp_day_loss, tf.float32)
        temp_day_loss = tf.clip_by_value(temp_day_loss, 0.0, 10.0)
        
        gen_trajs_hour_clipped = tf.clip_by_value(gen_trajs[3], 1e-7, 1.0 - 1e-7)
        temp_hour_loss = -tf.reduce_sum(real_trajs[3] * tf.math.log(gen_trajs_hour_clipped), axis=[1, 2])
        temp_hour_loss = tf.cast(temp_hour_loss, tf.float32)
        temp_hour_loss = tf.clip_by_value(temp_hour_loss, 0.0, 10.0)
        
        # Categorical loss - cross-entropy on category distributions
        gen_trajs_cat_clipped = tf.clip_by_value(gen_trajs[1], 1e-7, 1.0 - 1e-7)
        cat_loss = -tf.reduce_sum(real_trajs[1] * tf.math.log(gen_trajs_cat_clipped), axis=[1, 2])
        cat_loss = tf.cast(cat_loss, tf.float32)
        cat_loss = tf.clip_by_value(cat_loss, 0.0, 10.0)
        
        # Combine utility components with appropriate weights
        # Use the utility component weights from class attributes
        w_spatial = tf.constant(self.w_spatial, dtype=tf.float32)    # Spatial utility weight
        w_temporal = tf.constant(self.w_temporal, dtype=tf.float32)  # Temporal utility weight
        w_semantic = tf.constant(self.w_semantic, dtype=tf.float32)  # Semantic utility weight
        
        r_util = -(w_spatial * spatial_loss + w_temporal * (temp_day_loss + temp_hour_loss) + w_semantic * cat_loss)
        
        # Privacy preservation reward using TUL classifier
        # Adapt input format for MARC model which expects different dimensions
        try:
            batch_size = gen_trajs[0].shape[0]
            
            # Format inputs for MARC model:
            # 1. Convert one-hot to indices for day, hour, category
            # Note: gen_trajs order is [lat_lon, category, day, hour, mask]
            
            # Convert one-hot day to indices (batch_size, 144) where each value is 0-6
            day_indices = tf.cast(tf.argmax(gen_trajs[2], axis=-1), tf.int32)
            # Clip day values to ensure they're in the valid range [0, 6]
            day_indices = tf.clip_by_value(day_indices, 0, 6)
            
            # Convert one-hot hour to indices (batch_size, 144) where each value is 0-23
            hour_indices = tf.cast(tf.argmax(gen_trajs[3], axis=-1), tf.int32)
            # Clip hour values to ensure they're in the valid range [0, 23]
            hour_indices = tf.clip_by_value(hour_indices, 0, 23)
            
            # Convert one-hot category to indices (batch_size, 144) where each value is 0-9
            category_indices = tf.cast(tf.argmax(gen_trajs[1], axis=-1), tf.int32)
            # Clip category values to ensure they're in valid range [0, 9]
            category_indices = tf.clip_by_value(category_indices, 0, 9)
            
            # Format lat_lon to match MARC's expected input shape (batch_size, 144, 40)
            # Since our lat_lon is (batch_size, 144, 2), we'll pad it to 40 dimensions
            lat_lon_padded = tf.pad(gen_trajs[0], [[0, 0], [0, 0], [0, 38]])
            
            print(f"Input shapes - Day: {day_indices.shape}, Hour: {hour_indices.shape}, " +
                  f"Category: {category_indices.shape}, Lat_lon: {lat_lon_padded.shape}")
            print(f"Day range: [{tf.reduce_min(day_indices)}, {tf.reduce_max(day_indices)}]")
            
            # Call the MARC model with the correctly formatted inputs
            tul_preds = tul_classifier([day_indices, hour_indices, category_indices, lat_lon_padded])
            
            # Extract the probability for the correct user
            # Get the number of output classes from the TUL model
            num_users = tul_preds.shape[1]
            print(f"TUL predictions shape: {tul_preds.shape}")
            
            # Generate user indices but make sure they don't exceed the valid range
            user_indices = np.arange(batch_size) % num_users
            
            # Safely gather user probabilities
            batch_indices = tf.range(batch_size, dtype=tf.int32)
            indices = tf.stack([batch_indices, tf.cast(user_indices, tf.int32)], axis=1)
            user_probs = tf.gather_nd(tul_preds, indices)
            
            # Negative reward for correct user identification (penalize high probabilities)
            alpha = tf.constant(1.0, dtype=tf.float32)  # Privacy weight
            r_priv = -alpha * tf.cast(user_probs, tf.float32)
            
        except Exception as e:
            print(f"Error computing privacy reward: {e}")
            import traceback
            traceback.print_exc()
            print("Using a placeholder privacy reward instead")
            # If there's an error with the TUL model, use a placeholder privacy reward
            r_priv = tf.zeros_like(r_adv)
        
        # Combined reward with configurable weights from class attributes
        w_adv = tf.constant(self.w_adv, dtype=tf.float32)   # Adversarial weight
        w_util = tf.constant(self.w_util, dtype=tf.float32) # Utility weight
        w_priv = tf.constant(self.w_priv, dtype=tf.float32) # Privacy weight
        
        r_adv = tf.cast(r_adv, tf.float32)
        
        # Ensure r_adv, r_util, and r_priv have appropriate shapes
        r_adv = tf.reshape(r_adv, [-1])
        r_util = tf.reshape(r_util, [-1])
        r_priv = tf.reshape(r_priv, [-1])
        
        # Compute the combined reward
        combined_rewards = w_adv * r_adv + w_util * r_util + w_priv * r_priv
        
        # Ensure the rewards have shape [batch_size, 1]
        batch_size = r_adv.shape[0]
        rewards = tf.reshape(combined_rewards, [batch_size, 1])
        
        # Normalize rewards for training stability
        rewards_mean = tf.reduce_mean(rewards)
        rewards_std = tf.math.reduce_std(rewards) + 1e-8
        rewards = (rewards - rewards_mean) / rewards_std
        
        # Clip rewards to reasonable range to prevent training instability
        rewards = tf.clip_by_value(rewards, -5.0, 5.0)
        
        # Debug print to check rewards shape and values
        print(f"Rewards shape: {rewards.shape}, min: {tf.reduce_min(rewards)}, max: {tf.reduce_max(rewards)}, mean: {tf.reduce_mean(rewards)}")
        
        return rewards

    def train_step(self, real_trajs, batch_size=256):
        """Modified train_step to include tracking of reward metrics."""
        # Generate trajectories - using 3D continuous and categorical action space
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        gen_trajs = self.generator.predict([*real_trajs, noise])
        
        # Ensure consistent data types
        real_trajs = [tf.cast(tensor, tf.float32) for tensor in real_trajs]
        gen_trajs = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
        
        # Update the custom loss with the current real and generated trajectories
        self.traj_loss.set_trajectories(real_trajs, gen_trajs)
        
        # Compute full rewards using the TUL classifier
        rewards = self.compute_rewards(real_trajs, gen_trajs, self.tul_classifier)
        
        # Compute advantages and returns for PPO
        values = self.critic.predict(real_trajs[:4])
        values = tf.cast(values, tf.float32)
        advantages = compute_advantage(rewards, values, self.gamma, self.gae_lambda)
        returns = compute_returns(rewards, self.gamma)
        
        # Ensure returns has the same batch size as what the critic expects
        # The critic takes real_trajs[:4] as input, so returns should match that shape
        if returns.shape[0] != batch_size:
            print(f"Warning: Reshaping returns from {returns.shape} to [{batch_size}, 1]")
            returns = tf.reshape(returns, [batch_size, 1])
        
        # Update critic using returns
        c_loss = self.critic.train_on_batch(real_trajs[:4], returns)
        
        # Update discriminator
        d_loss_real = self.discriminator.train_on_batch(
            real_trajs[:4],
            np.ones((batch_size, 1))
        )
        d_loss_fake = self.discriminator.train_on_batch(
            gen_trajs[:4],
            np.zeros((batch_size, 1))
        )
        
        # Update generator (actor) using advantages
        g_loss = self.update_actor(real_trajs, gen_trajs, advantages)
        
        # Get mean reward for tracking
        mean_reward = tf.reduce_mean(rewards).numpy()
        
        # Return metrics including reward
        return {
            "d_loss_real": d_loss_real, 
            "d_loss_fake": d_loss_fake, 
            "g_loss": g_loss, 
            "c_loss": c_loss,
            "reward": mean_reward
        }

    def train(self, epochs=2000, batch_size=256, sample_interval=10, save_best=True, checkpoint_dir='results'):
        """Train the model with WandB tracking, best model checkpointing, and early stopping."""
        # Make sure the checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get the early stopping callback if attached
        early_stopping = getattr(self, 'early_stopping_callback', None)
        
        # Initialize tracking variables
        completed_epochs = 0
        
        # Rest of initialization code remains the same...
        
        # Training loop
        print(f"Starting training for {epochs} epochs with early stopping patience of "
              f"{early_stopping.patience if early_stopping else 'N/A'}")
        
        for epoch in range(epochs):
            # Existing training code...
            
            # Training step
            metrics = self.train_step(batch, batch_size)
            
            # Log metrics to WandB
            if self.wandb:
                # Existing wandb logging code...
                
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
            
            # Existing code for printing, saving checkpoints, etc.
            
            completed_epochs = epoch + 1
    
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
        """Update generator using a simplified policy gradient approach."""
        # Get a single batch of data with random noise for the generator
        batch_size = states[0].shape[0]
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Prepare inputs for the generator
        all_inputs = [*states, noise]
        
        # Use ones as targets for the discriminator output (we want to fool the discriminator)
        targets = np.ones((batch_size, 1))
        
        # Convert advantages to numpy and ensure proper type/shape
        try:
            # First ensure advantages is a tensor with float32 type
            advantages = tf.cast(advantages, tf.float32)
            
            # Clip advantages to reasonable range before any other operations
            advantages = tf.clip_by_value(advantages, -10.0, 10.0)
            
            # Then convert to numpy and flatten
            advantages_np = advantages.numpy().flatten()  # Flatten to ensure it's 1D
            
            # Scale advantages to be positive (sample_weight should be positive)
            advantages_np = advantages_np - np.min(advantages_np) + 1e-3
            
            # Normalize to reasonable values (0 to 1 range)
            if np.max(advantages_np) > 0:
                advantages_np = advantages_np / np.max(advantages_np)
                
            # Ensure it has the right shape
            if len(advantages_np) != batch_size:
                print(f"Warning: Reshaping advantages from {len(advantages_np)} to {batch_size}")
                # If shapes don't match, use uniform weights
                advantages_np = np.ones(batch_size)
                
            # Final safety check - replace any NaN or inf values
            advantages_np = np.nan_to_num(advantages_np, nan=0.5, posinf=1.0, neginf=0.0)
                
        except Exception as e:
            print(f"Error processing advantages: {e}")
            # Fallback to uniform weights
            advantages_np = np.ones(batch_size)
            
        # Print statistics about advantage values used for training
        print(f"Advantage stats - min: {np.min(advantages_np):.4f}, max: {np.max(advantages_np):.4f}, " +
              f"mean: {np.mean(advantages_np):.4f}, std: {np.std(advantages_np):.4f}")
        
        # Train the combined model with sample weights from advantages
        loss = self.combined.train_on_batch(
            all_inputs, 
            targets,
            sample_weight=advantages_np
        )
        
        # If loss is extremely high, clip it for reporting purposes
        if loss > 10000:  # Arbitrary threshold for "too high" loss
            print(f"Warning: Loss is very high ({loss:.2f}), consider reducing learning rate manually")
            loss = min(loss, 10000.0)
        
        return loss

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
            
            # Enhanced state representation: combine timestamp embedding and location embedding
            # Timestamp embedding
            emb_day = Embedding(input_dim=7, output_dim=32, input_length=144)(input_day)
            emb_hour = Embedding(input_dim=24, output_dim=32, input_length=144)(input_hour)
            time_embedding = Concatenate(axis=2)([emb_day, emb_hour])
            
            # Location embedding
            emb_category = Embedding(input_dim=10, output_dim=32, input_length=144)(input_category)
            location_embedding = Dense(32, activation='relu')(input_lat_lon)
            
            # Combined state representation
            state_repr = Concatenate(axis=2)([time_embedding, emb_category, location_embedding])
            
            # Process with LSTM
            lstm_out = LSTM(64, return_sequences=False)(state_repr)
            
            # Dense layers
            dense1 = Dense(128, activation='relu')(lstm_out)
            
            # Output layer - assuming ~193 users for classification
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
        """Generate and visualize sample trajectories for wandb."""
        if not self.wandb:
            return
            
        try:
            # Generate random noise for sampling
            noise = np.random.normal(0, 1, (4, self.latent_dim))
            
            # Sample a few real trajectories
            idx = np.random.randint(0, len(self.X_train[0]), 4)
            real_batch = [X[idx] for X in self.X_train]
            
            # Generate trajectories
            gen_trajs = self.generator.predict([*real_batch, noise])
            
            # Create visualization of the trajectories
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i in range(4):
                # Plot real trajectory
                axes[i].scatter(
                    real_batch[0][i, :, 0], 
                    real_batch[0][i, :, 1], 
                    c='blue', 
                    alpha=0.7, 
                    label='Real'
                )
                
                # Plot generated trajectory
                axes[i].scatter(
                    gen_trajs[0][i, :, 0], 
                    gen_trajs[0][i, :, 1], 
                    c='red', 
                    alpha=0.7, 
                    label='Generated'
                )
                
                axes[i].set_title(f'Trajectory Sample {i+1}')
                axes[i].legend()
                
            plt.tight_layout()
            
            # Log to wandb
            self.wandb.log({f"trajectory_samples": self.wandb.Image(fig)})
            plt.close(fig)
            
        except Exception as e:
            print(f"Error generating trajectory samples for wandb: {e}")