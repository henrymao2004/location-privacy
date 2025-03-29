import tensorflow as tf
import keras
import numpy as np
import random
from tensorflow.keras import layers
import tensorflow_probability as tfp
import os
import json
import warnings
import gc

# Add wandb import with try-except to handle cases where it might not be available
try:
    import wandb
except ImportError:
    warnings.warn("wandb not installed. WandB logging will be disabled.")
    wandb = None

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

from keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Reshape, Multiply
from keras.initializers import he_uniform
from keras.regularizers import l1

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from losses import d_bce_loss, trajLoss, TrajLossLayer, CustomTrajLoss, compute_advantage, compute_returns, compute_trajectory_ratio, compute_entropy_loss

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
        
        # RL parameters - refined based on formulation
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.c1 = 0.5  # value function coefficient (balanced for better critic training)
        self.c2 = 0.03  # entropy coefficient (higher for better exploration)
        self.ppo_epochs = 5  # Number of PPO epochs per batch
        
        # Generator-Discriminator balance parameter
        self.gen_updates_per_disc = 4  # Update generator this many times per discriminator update
        
        # Dynamic clip limits for reward components
        self.initial_clip_limits = {
            'spatial': 10.0,  # Broader range for spatial exploration
            'temporal': 12.0, # Allow more temporal flexibility
            'category': 10.0  # Allow more category flexibility
        }
        self.max_clip_limits = {
            'spatial': 15.0,  # Maximum allowable spatial distance
            'temporal': 20.0, # Maximum allowable temporal distance
            'category': 15.0  # Maximum allowable category distance
        }
        self.clip_increase_start_epoch = 15  # Start increasing clips earlier
        self.clip_increase_frequency = 5    # Adjust clip limits more frequently
        self.clip_increase_rate = 0.15      # Balanced increase rate
        self.current_clip_limits = self.initial_clip_limits.copy()
        self.current_epoch = 0  # Track current epoch for clip limit adjustment
        
        # Load or initialize TUL classifier for privacy rewards
        self.tul_classifier = self.load_tul_classifier()
        
        # Reward weights based on formulation
        # alpha * R_priv + beta * R_util + gamma * R_adv
        self.alpha = 0.6    # Privacy reward weight
        self.beta = 0.8     # Utility reward weight  
        self.gamma_adv = 0.2  # Adversarial reward weight
        
        # Utility reward component weights for w_1, w_2, w_3
        self.w1 = 0.5     # Spatial weight
        self.w2 = 0.3     # Temporal weight
        self.w3 = 0.2     # Semantic/category weight
        
        # Initial and target utility component weights for curriculum learning
        self.initial_component_weights = {
            'spatial': 0.7,     # Initial spatial weight 
            'temporal': 0.2,    # Initial temporal weight
            'category': 0.1     # Initial category weight
        }
        
        self.target_component_weights = {
            'spatial': 0.5,     # Target spatial weight
            'temporal': 0.3,    # Target temporal weight
            'category': 0.2     # Target category weight
        }
        
        # Current utility component weights (will be updated during training)
        self.current_component_weights = self.initial_component_weights.copy()
        
        # Curriculum learning parameters
        self.curriculum_start_epoch = 20     # Start curriculum at epoch 20
        self.curriculum_duration = 100       # Transition over 100 epochs
        
        # Adaptive reward balancing settings
        self.privacy_target = 0.6      # Target privacy protection level (TUL accuracy, lower is better)
        self.utility_target = 0.8      # Target utility preservation level
        self.alpha_min = 0.3          # Minimum privacy weight
        self.alpha_max = 0.8          # Maximum privacy weight
        self.beta_min = 0.5           # Minimum utility weight
        self.beta_max = 0.9           # Maximum utility weight
        
        # For tracking metrics and adjusting weights
        self.current_tul_accuracy = 1.0  # Start with worst privacy (100% identifiable)
        self.current_utility_score = 0.0 # Start with worst utility (0%)
        
        # For backwards compatibility with existing code
        self.w_priv = self.alpha
        self.w_util = self.beta
        self.w_adv = self.gamma_adv
        
        # Tracking variable for wandb usage
        self.use_wandb = False
        
        # Define optimizers with well-tuned learning rates
        self.actor_optimizer = Adam(0.0003, clipnorm=1.0)  # More stable learning rate
        self.critic_optimizer = Adam(0.0003, clipnorm=1.0) # Same learning rate as actor for stability
        self.discriminator_optimizer = Adam(0.0001, clipnorm=0.5)

        # Build networks
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.discriminator = self.build_discriminator()
        
        # Compile models
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer)
        self.critic.compile(loss='mse', optimizer=self.critic_optimizer)
        
        # Combined model for training
        self.setup_combined_model()

        # Track reward normalization factors for utility components
        self.utility_norm_stats = {
            'spatial_mean': 0.0,
            'spatial_std': 1.0,
            'temporal_mean': 0.0,
            'temporal_std': 1.0,
            'category_mean': 0.0,
            'category_std': 1.0,
            'update_count': 0
        }

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
        
        # Embedding layers for each feature - REDUCED DIMENSIONS
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=32, activation='relu', use_bias=True,  # Reduced from 128 to 32
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                stacked_latlon = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
                embeddings.append(stacked_latlon)
                inputs.append(i)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=16, activation='relu', use_bias=True,  # Reduced from 64 to 16
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_cat = [d(x) for x in unstacked]
                stacked_cat = Lambda(lambda x: tf.stack(x, axis=1))(dense_cat)
                embeddings.append(stacked_cat)
                inputs.append(i)
        
        # Combine all embeddings
        combined = Concatenate(axis=2)(embeddings)
        
        # Dense layer for noise input - REDUCED DIMENSIONS
        noise_dense = Dense(self.max_length * 16, activation='relu')(noise)  # Reduced from 32 to 16
        noise_reshape = Reshape((self.max_length, 16))(noise_dense)  # Adjust shape to match
        
        # Add noise to embeddings
        with_noise = Concatenate(axis=2)([combined, noise_reshape])
        
        # Apply Transformer blocks - REDUCED DEPTH
        x = with_noise
        
        # Stack of transformer blocks with reduced capacity
        num_transformer_blocks = 1  # Reduced from 3 to 1
        embed_dim = x.shape[-1]
        
        for i in range(num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=embed_dim, 
                num_heads=4,  # Reduced from 8 to 4
                ff_dim=embed_dim * 2,  # Reduced from 4x to 2x
                rate=0.05  # Reduced dropout
            )(x, training=True)
        
        # Output layers for each feature - SIMPLIFIED
        outputs = []
        
        # Create a shared dense layer with reduced size
        shared_dense = Dense(64, activation='relu', name='shared_output_dense')(x)  # Reduced from 128 to 64
        
        # Lat-lon output - simplified
        lat_lon_out = TimeDistributed(Dense(self.vocab_size['lat_lon'], activation='tanh', name='lat_lon_output'))(shared_dense)
        outputs.append(lat_lon_out)
        
        # Day output - simplified
        day_out = TimeDistributed(Dense(self.vocab_size['day'], activation='softmax', name='day_output'))(shared_dense)
        outputs.append(day_out)
        
        # Hour output - simplified
        hour_out = TimeDistributed(Dense(self.vocab_size['hour'], activation='softmax', name='hour_output'))(shared_dense)
        outputs.append(hour_out)
        
        # Category output - simplified
        cat_out = TimeDistributed(Dense(self.vocab_size['category'], activation='softmax', name='cat_output'))(shared_dense)
        outputs.append(cat_out)
        
        # Mask output (pass-through)
        outputs.append(mask)
        
        # Create the model
        model = Model(inputs + [noise], outputs, name='generator')
        return model

    def build_critic(self):
        """Build the critic model for value function estimation.
        
        Returns:
            A Keras model that predicts state values.
        """
        # Create input layers - same as generator without noise
        inputs = []
        embeddings = []
        
        # Input for each feature
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                i = Input(shape=(self.max_length, 1), name='critic_input_' + key)
                inputs.append(i)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='critic_input_' + key)
                # Simplify: use shared embedding without unstacking for speed
                emb = Dense(16, activation='relu', name='critic_emb_' + key)(i)  # Reduced from 32
                embeddings.append(emb)
                inputs.append(i)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='critic_input_' + key)
                # Simplify: use shared embedding without unstacking for speed
                emb = Dense(8, activation='relu', name='critic_emb_' + key)(i)  # Reduced from 16
                embeddings.append(emb)
                inputs.append(i)
        
        # Combine all embeddings
        combined = Concatenate(axis=2)(embeddings)
        
        # Apply mask for padding if provided
        mask = inputs[-1]  # Last input is mask
        masked = Multiply()([combined, mask])
        
        # Simplified architecture: Replace transformer with LSTM
        # This is faster than transformer blocks
        lstm = LSTM(32, return_sequences=False, name='critic_lstm')(masked)  # Reduced from 64
        
        # Value prediction head - single layer
        value = Dense(1, name='critic_value')(lstm)
        
        # Create the model
        model = Model(inputs, value, name='critic')
        return model

    def build_discriminator(self):
        """Build the discriminator model.
        
        Returns:
            A Keras model that classifies trajectories as real or fake.
        """
        # Create input layers - same structure as generator output
        inputs = []
        embeddings = []
        
        # Process each feature
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                mask = Input(shape=(self.max_length, 1), name='disc_input_mask')
                inputs.append(mask)
                continue
                
            i = Input(shape=(self.max_length, self.vocab_size[key]), name='disc_input_' + key)
            # Use simple dense layer instead of complex embedding
            x = Dense(16, activation='relu', name=f'disc_dense_{key}')(i)  # Reduced from 32
            embeddings.append(x)
            inputs.append(i)
        
        # Combine all embeddings
        combined = Concatenate(axis=2)(embeddings)
        
        # Apply mask
        masked = Multiply()([combined, mask])
        
        # Use simple LSTM instead of transformer for speed
        x = LSTM(32, return_sequences=False)(masked)  # Reduced from 64
        
        # Simple classification head
        x = Dense(16, activation='relu')(x)  # Reduced from 32
        validity = Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs, validity, name='discriminator')
        return model

    def setup_combined_model(self):
        """Create the combined actor-critic model for RL training.
        
        This function creates a combined model that:
        1. Takes trajectory inputs and noise
        2. Generates synthetic trajectories via the generator
        3. Passes them to the discriminator for adversarial training
        
        For PPO, the model structure is maintained, but policy updates happen
        through the custom update_actor method instead of direct model.fit calls.
        """
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
        pred = self.discriminator(gen_trajs)
        
        # Create the combined model
        self.combined = Model(inputs, pred)
        
        # Create a custom loss instance for trajectory optimization
        self.traj_loss = CustomTrajLoss()
        
        # Store references to input and output tensors
        self.input_tensors = inputs
        self.generated_trajectories = gen_trajs
        
        # Compile the model with the custom loss
        # For PPO, this only serves as a backup - main updates are done in update_actor
        self.combined.compile(loss=self.traj_loss, optimizer=self.actor_optimizer)
        
        # Store the generator outputs for later use in reward computation
        self.gen_trajs_symbolic = gen_trajs
        
        # Use the correct function to enable eager execution
        # tf.config.run_functions_eagerly(True) is deprecated for data ops
        try:
            tf.data.experimental.enable_debug_mode()
            print("TensorFlow data debug mode enabled")
        except Exception as e:
            print(f"Could not enable TF data debug mode: {e}")
            # Fallback to the traditional method
            tf.config.run_functions_eagerly(True)
            print("TensorFlow functions running eagerly")

    def update_clip_limits(self, epoch):
        """Update the clip limits for utility components based on training progress.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        # Only start increasing clip limits after specified epoch
        if epoch < self.clip_increase_start_epoch:
            if epoch % 20 == 0:  # Log occasionally before starting
                print(f"Epoch {epoch}: Clip limit updates will start at epoch {self.clip_increase_start_epoch}")
            return False
        
        # Track if limits changed in this update
        limits_changed = False
        reset_components = []
        
        # Check if it's time to increase clip limits
        epochs_since_start = epoch - self.clip_increase_start_epoch
        
        # Log on first eligible epoch
        if epochs_since_start == 0:
            print(f"Epoch {epoch}: Reached clip limit increase start epoch. Current limits - "
                  f"Temporal: {self.current_clip_limits['temporal']:.2f}, "
                  f"Category: {self.current_clip_limits['category']:.2f}")
        
        if epochs_since_start > 0 and epochs_since_start % self.clip_increase_frequency == 0:
            print(f"Epoch {epoch}: Eligible for clip limit increase (epochs since start: {epochs_since_start}, "
                  f"frequency: {self.clip_increase_frequency})")
            
            # Increase each clip limit
            for component in ['temporal', 'category']:
                old_limit = self.current_clip_limits[component]
                # Increase by clip_increase_rate (e.g., 20%)
                new_limit = self.current_clip_limits[component] * (1 + self.clip_increase_rate)
                # Cap at max limit
                self.current_clip_limits[component] = min(new_limit, self.max_clip_limits[component])
                
                # Check if limit actually changed
                if abs(old_limit - self.current_clip_limits[component]) > 1e-6:
                    limits_changed = True
                    reset_components.append(component)
                    print(f"  - {component.capitalize()} limit increased: {old_limit:.2f} -> {self.current_clip_limits[component]:.2f}")
                else:
                    print(f"  - {component.capitalize()} limit unchanged: {old_limit:.2f} (max: {self.max_clip_limits[component]:.2f})")
            
            # Reset normalization statistics for components that changed
            if reset_components:
                print(f"Resetting normalization statistics for components: {', '.join(reset_components)}")
                for component in reset_components:
                    if component == 'temporal':
                        self.utility_norm_stats['temporal_mean'] = 0.0
                        self.utility_norm_stats['temporal_std'] = 1.0
                    elif component == 'category':
                        self.utility_norm_stats['category_mean'] = 0.0
                        self.utility_norm_stats['category_std'] = 1.0
                
                # Reset update counter to quickly adapt to new statistics
                self.utility_norm_stats['update_count'] = 0
                print("Normalization statistics reset complete")
            
            print(f"Epoch {epoch}: Updated clip limits - Temporal: {self.current_clip_limits['temporal']:.2f}, "
                  f"Category: {self.current_clip_limits['category']:.2f}")
        elif epoch % 10 == 0:
            # Log status every 10 epochs for monitoring
            print(f"Epoch {epoch}: Current clip limits - Temporal: {self.current_clip_limits['temporal']:.2f}, "
                  f"Category: {self.current_clip_limits['category']:.2f}, Next update at epoch {self.clip_increase_start_epoch + ((epochs_since_start // self.clip_increase_frequency) + 1) * self.clip_increase_frequency}")
            
        return limits_changed
    
    def set_manual_clip_limits(self, temporal=None, category=None, spatial=None, reset_stats=True):
        """Manually set clip limits for utility components.
        
        Args:
            temporal: New clip limit for temporal component (None = no change)
            category: New clip limit for category component (None = no change)
            spatial: New clip limit for spatial component (None = no change)
            reset_stats: Whether to reset normalization statistics for changed components
            
        Returns:
            Dictionary of changes made
        """
        changes = {}
        reset_components = []
        
        # Update temporal clip limit if provided
        if temporal is not None:
            old_limit = self.current_clip_limits['temporal']
            # Ensure new limit is within bounds
            new_limit = min(max(temporal, 1.0), self.max_clip_limits['temporal'])
            self.current_clip_limits['temporal'] = new_limit
            changes['temporal'] = (old_limit, new_limit)
            print(f"Manual update: Temporal clip limit changed from {old_limit:.2f} to {new_limit:.2f}")
            if reset_stats and abs(new_limit - old_limit) > 1e-6:
                reset_components.append('temporal')
        
        # Update category clip limit if provided
        if category is not None:
            old_limit = self.current_clip_limits['category']
            # Ensure new limit is within bounds
            new_limit = min(max(category, 1.0), self.max_clip_limits['category'])
            self.current_clip_limits['category'] = new_limit
            changes['category'] = (old_limit, new_limit)
            print(f"Manual update: Category clip limit changed from {old_limit:.2f} to {new_limit:.2f}")
            if reset_stats and abs(new_limit - old_limit) > 1e-6:
                reset_components.append('category')
        
        # Update spatial clip limit if provided
        if spatial is not None:
            old_limit = self.current_clip_limits['spatial']
            # Ensure new limit is within bounds
            new_limit = min(max(spatial, 1.0), self.max_clip_limits['spatial'])
            self.current_clip_limits['spatial'] = new_limit
            changes['spatial'] = (old_limit, new_limit)
            print(f"Manual update: Spatial clip limit changed from {old_limit:.2f} to {new_limit:.2f}")
            if reset_stats and abs(new_limit - old_limit) > 1e-6:
                reset_components.append('spatial')
        
        # Reset normalization statistics for changed components if requested
        if reset_stats and reset_components:
            print(f"Resetting normalization statistics for components: {', '.join(reset_components)}")
            for component in reset_components:
                if component == 'temporal':
                    self.utility_norm_stats['temporal_mean'] = 0.0
                    self.utility_norm_stats['temporal_std'] = 1.0
                elif component == 'category':
                    self.utility_norm_stats['category_mean'] = 0.0
                    self.utility_norm_stats['category_std'] = 1.0
                elif component == 'spatial':
                    self.utility_norm_stats['spatial_mean'] = 0.0
                    self.utility_norm_stats['spatial_std'] = 1.0
            
            # Reset update counter to quickly adapt to new statistics
            self.utility_norm_stats['update_count'] = 0
            print("Normalization statistics reset complete")
        
        # Print summary if any changes were made
        if changes:
            print(f"Current clip limits after manual update - Temporal: {self.current_clip_limits['temporal']:.2f}, "
                  f"Category: {self.current_clip_limits['category']:.2f}, Spatial: {self.current_clip_limits['spatial']:.2f}")
            
        return changes
                  
    def update_component_weights(self, epoch):
        """Update utility component weights according to curriculum learning schedule.
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
        
        # Track if weights changed in this update
        weights_changed = False
        old_weights = self.current_component_weights.copy()
        
        # Only start curriculum after specified epoch
        if epoch < self.curriculum_start_epoch:
            return weights_changed
            
        # Check if we're still in curriculum phase
        if epoch >= self.curriculum_start_epoch + self.curriculum_duration:
            # We've reached the end of curriculum, use target weights
            self.current_component_weights = self.target_component_weights.copy()
            
            # Check if weights actually changed
            for component in ['spatial', 'temporal', 'category']:
                if abs(old_weights[component] - self.current_component_weights[component]) > 1e-6:
                    weights_changed = True
                    
            return weights_changed
            
        # Calculate progress through curriculum (0.0 to 1.0)
        progress = (epoch - self.curriculum_start_epoch) / self.curriculum_duration
        
        # Linearly interpolate between initial and target weights
        for component in ['spatial', 'temporal', 'category']:
            initial = self.initial_component_weights[component]
            target = self.target_component_weights[component]
            current = initial + progress * (target - initial)
            self.current_component_weights[component] = current
            
            # Check if weight changed significantly
            if abs(old_weights[component] - current) > 0.01:  # 1% change threshold
                weights_changed = True
            
        # Update legacy attributes for backward compatibility
        self.beta = self.current_component_weights['spatial']
        self.gamma_temporal = self.current_component_weights['temporal']
        self.chi = self.current_component_weights['category']
        
        # Log changes at regular intervals
        if epoch % 10 == 0 or (epoch - self.curriculum_start_epoch) % 50 == 0:
            print(f"Epoch {epoch}: Updated component weights - "
                  f"Spatial: {self.current_component_weights['spatial']:.3f}, "
                  f"Temporal: {self.current_component_weights['temporal']:.3f}, "
                  f"Category: {self.current_component_weights['category']:.3f}")
                  
        return weights_changed
    
    def compute_rewards(self, real_trajs, gen_trajs, tul_classifier):
        """Compute rewards for generated trajectories based on privacy, utility, and realism.
        
        Args:
            real_trajs: List of real trajectories (lat_lon, category, day, hour, mask)
            gen_trajs: List of generated trajectories (lat_lon, category, day, hour, mask)
            tul_classifier: TUL model for privacy evaluation
            
        Returns:
            rewards: Tensor of shape [batch_size, 1] containing rewards
            privacy_rewards: Privacy component of rewards
            utility_rewards: Utility component of rewards
            adversarial_rewards: Adversarial component of rewards
        """
        batch_size = real_trajs[0].shape[0]
        
        # Use reduced weights in early epochs to stabilize training
        # Gradual increase of privacy importance
        current_alpha = min(self.alpha, 0.2 + (self.alpha - 0.2) * min(1.0, self.current_epoch / 10.0))
        
        # 1. Compute privacy reward - skip in early epochs to reduce computation
        if self.current_epoch < 5:
            # Use placeholder values in early epochs to reduce computation
            privacy_rewards = tf.ones([batch_size, 1], dtype=tf.float32) * 0.5
            # Still track TUL accuracy to make sure we're on the right track
            if hasattr(self, 'current_tul_accuracy'):
                self.current_tul_accuracy = 0.8  # Placeholder value
        else:
            privacy_rewards = self._compute_privacy_reward(gen_trajs, tul_classifier)
        
        # 2. Compute utility reward with fewer calculations
        utility_rewards = self._compute_utility_reward(real_trajs, gen_trajs)
        
        # 3. Compute adversarial reward only occasionally in early training
        if self.current_epoch < 10 and self.current_epoch % 2 != 0:
            # Use cached or placeholder values in early epochs
            adversarial_rewards = tf.ones([batch_size, 1], dtype=tf.float32) * 0.5
        else:
            adversarial_rewards = self._compute_adversarial_reward(gen_trajs)
        
        # 4. Combine rewards with adaptive weights
        # R(T_i, T_i^orig) = α·R_priv + β·R_util + γ·R_adv
        # Use the current alpha that varies with epoch
        rewards = (
            current_alpha * privacy_rewards + 
            self.beta * utility_rewards + 
            self.gamma_adv * adversarial_rewards
        )
        
        # Apply adaptive reward balancing less frequently to reduce computation
        if self.current_epoch % 5 == 0 and hasattr(self, 'current_tul_accuracy') and hasattr(self, 'current_utility_score'):
            # Update alpha based on current privacy performance
            # Only adapt beyond initial epochs
            if self.current_epoch > 10:
                if self.current_tul_accuracy > 0 and self.privacy_target > 0:
                    adaptive_alpha = current_alpha * min(1, self.current_tul_accuracy / self.privacy_target)
                    adaptive_alpha = max(self.alpha_min, min(self.alpha_max, adaptive_alpha))
                else:
                    adaptive_alpha = current_alpha
                    
                # Update beta based on current utility performance
                if self.current_utility_score >= 0:
                    utility_gap = max(0, 1 - self.current_utility_score / self.utility_target)
                    adaptive_beta = self.beta * (1 - utility_gap)
                    adaptive_beta = max(self.beta_min, min(self.beta_max, adaptive_beta))
                else:
                    adaptive_beta = self.beta
                
                # Recompute combined reward with adaptive weights
                rewards = (
                    adaptive_alpha * privacy_rewards + 
                    adaptive_beta * utility_rewards + 
                    self.gamma_adv * adversarial_rewards
                )
                
                # Log the adaptive weights if using wandb and only occasionally
                if self.use_wandb and wandb is not None:
                    wandb.log({
                        'adaptive_alpha': adaptive_alpha,
                        'adaptive_beta': adaptive_beta,
                        'original_alpha': self.alpha,
                        'original_beta': self.beta
                    })
        
        # Normalize rewards for stable training, but with less computation
        # Use running stats to avoid recalculating mean/std every batch
        if not hasattr(self, 'reward_stats'):
            self.reward_stats = {'mean': 0.0, 'std': 1.0, 'count': 0}
        
        # Update running stats
        rewards_mean = tf.reduce_mean(rewards)
        rewards_std = tf.math.reduce_std(rewards) + 1e-8
        
        # Update running average of mean and std
        if self.reward_stats['count'] < 100:
            # Use exponential moving average for first 100 updates
            alpha = 0.1
            self.reward_stats['mean'] = (1 - alpha) * self.reward_stats['mean'] + alpha * rewards_mean.numpy()
            self.reward_stats['std'] = (1 - alpha) * self.reward_stats['std'] + alpha * rewards_std.numpy()
            self.reward_stats['count'] += 1
        
        # Use running stats for normalization
        normalized_rewards = (rewards - self.reward_stats['mean']) / self.reward_stats['std']
        
        return normalized_rewards, privacy_rewards, utility_rewards, adversarial_rewards
    
    def _compute_privacy_reward(self, gen_trajs, tul_classifier):
        """Compute privacy reward using Trajectory-User Linking (TUL) model.
        
        R_priv = -log p_TUL(u_i|T_i)
        
        Higher values indicate better privacy (lower probability of linking).
        """
        if tul_classifier is None:
            # If no TUL model available, return a default privacy reward
            batch_size = gen_trajs[0].shape[0]
            return tf.ones([batch_size, 1], dtype=tf.float32) * 0.5
            
        try:
            # Limit batch size for TUL classifier to avoid OOM errors
            batch_size = gen_trajs[0].shape[0]
            max_tul_batch = 64  # Max batch size for TUL classifier to avoid memory issues
            
            # Process in smaller batches if needed
            if batch_size > max_tul_batch:
                # Split processing into smaller batches
                all_privacy_rewards = []
                
                for i in range(0, batch_size, max_tul_batch):
                    end_idx = min(i + max_tul_batch, batch_size)
                    batch_gen_trajs = [traj[i:end_idx] for traj in gen_trajs]
                    
                    # Process this batch
                    batch_rewards = self._compute_privacy_reward_batch(batch_gen_trajs, tul_classifier)
                    all_privacy_rewards.append(batch_rewards)
                
                # Combine results
                privacy_rewards = tf.concat(all_privacy_rewards, axis=0)
                return privacy_rewards
            else:
                # Process the entire batch at once if it's small enough
                return self._compute_privacy_reward_batch(gen_trajs, tul_classifier)
                
        except Exception as e:
            print(f"Error computing privacy reward: {e}")
            # Return a default value in case of error
            batch_size = gen_trajs[0].shape[0]
            return tf.ones([batch_size, 1], dtype=tf.float32) * 0.5
            
    def _compute_privacy_reward_batch(self, gen_trajs, tul_classifier):
        """Compute privacy reward for a single batch that fits in memory.
        
        Args:
            gen_trajs: List of generated trajectories for a single batch
            tul_classifier: TUL model for privacy evaluation
            
        Returns:
            privacy_rewards: Privacy reward tensor for the batch
        """
        try:
            # Format data for TUL model
            gen_lat_lon = gen_trajs[0]  # [batch_size, seq_len, 2]
            gen_category = gen_trajs[1]  # [batch_size, seq_len, n_categories]
            gen_day = gen_trajs[2]      # [batch_size, seq_len, 7]
            gen_hour = gen_trajs[3]     # [batch_size, seq_len, 24]
            gen_mask = gen_trajs[4]     # [batch_size, seq_len, 1]
            
            # Use efficient operations that can be executed on GPU
            with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
                # Convert one-hot encodings to indices if needed by the MARC model
                day_indices = tf.argmax(gen_day, axis=-1)  # [batch_size, seq_len]
                hour_indices = tf.argmax(gen_hour, axis=-1)  # [batch_size, seq_len]
                category_indices = tf.argmax(gen_category, axis=-1)  # [batch_size, seq_len]
                
                # Reshape lat_lon if needed (MARC may expect a different format)
                batch_size, seq_len, _ = gen_lat_lon.shape
                lat_lon_expanded = tf.pad(gen_lat_lon, [[0, 0], [0, 0], [0, 38]])  # Pad to 40 dimensions
                
                # Convert to NumPy for prediction if needed by MARC
                # Some models work better with NumPy arrays for prediction
                day_indices_np = day_indices.numpy()
                hour_indices_np = hour_indices.numpy()
                category_indices_np = category_indices.numpy()
                lat_lon_expanded_np = lat_lon_expanded.numpy()
                
                # Create MARC-compatible inputs
                marc_inputs = [
                    day_indices_np,        # Day indices
                    hour_indices_np,       # Hour indices
                    category_indices_np,   # Category indices
                    lat_lon_expanded_np    # Expanded lat_lon
                ]
                
                # Reduce verbosity in prediction
                tf.get_logger().setLevel('ERROR')
                
                # Get user identification probabilities from TUL model
                try:
                    # First try the standard predict method
                    user_probs = tul_classifier.predict(marc_inputs, verbose=0)
                except Exception as e1:
                    print(f"Standard predict failed: {e1}, trying alternative methods")
                    try:
                        # Try calling the model directly
                        if hasattr(tul_classifier, 'model'):
                            # If it has a model attribute, call that
                            user_probs = tul_classifier.model(marc_inputs, training=False)
                        elif hasattr(tul_classifier, '__call__'):
                            # Try direct call
                            user_probs = tul_classifier(marc_inputs)
                        else:
                            raise ValueError("No working prediction method found")
                    except Exception as e2:
                        print(f"All prediction methods failed: {e2}")
                        return tf.ones([batch_size, 1], dtype=tf.float32) * 0.5
                
                # Convert user_probs to tensor if it's numpy
                if isinstance(user_probs, np.ndarray):
                    user_probs = tf.convert_to_tensor(user_probs, dtype=tf.float32)
                
                # Clip probabilities to avoid log(0)
                user_probs = tf.clip_by_value(user_probs, 1e-7, 1.0)
                
                # Compute max probability (probability of correctly identifying user)
                max_probs = tf.reduce_max(user_probs, axis=1, keepdims=True)
                
                # Compute privacy reward: -log(p)
                # Higher values = better privacy
                privacy_rewards = -tf.math.log(max_probs)
                
                # Update current TUL accuracy for adaptive reward balancing
                if hasattr(self, 'current_tul_accuracy'):
                    # Mean of max probabilities across batch
                    self.current_tul_accuracy = tf.reduce_mean(max_probs).numpy()
                    
                    # Log to wandb if enabled
                    if self.use_wandb and wandb is not None:
                        wandb.log({'tul_accuracy': self.current_tul_accuracy})
                
                return privacy_rewards
                
        except Exception as e:
            print(f"Error in _compute_privacy_reward_batch: {e}")
            # Return a default value in case of error
            batch_size = gen_trajs[0].shape[0]
            return tf.ones([batch_size, 1], dtype=tf.float32) * 0.5
    
    def _compute_utility_reward(self, real_trajs, gen_trajs):
        """Compute utility reward as weighted sum of spatial, temporal, and semantic components.
        
        R_util = -(w1 · d_spatial + w2 · d_temporal + w3 · d_semantic)
        
        Negative values indicate better utility (lower distances).
        """
        # Get components from trajectories
        real_lat_lon = real_trajs[0]
        gen_lat_lon = gen_trajs[0]
        real_day = real_trajs[2]
        gen_day = gen_trajs[2]
        real_hour = real_trajs[3]
        gen_hour = gen_trajs[3]
        real_cat = real_trajs[1]
        gen_cat = gen_trajs[1]
        mask = real_trajs[4]
        
        # 1. Compute spatial distance
        spatial_utility = self._compute_spatial_utility(real_lat_lon, gen_lat_lon, mask)
        
        # 2. Compute temporal distance
        temporal_utility = self._compute_temporal_utility(real_day, gen_day, real_hour, gen_hour, mask)
        
        # 3. Compute category/semantic distance
        category_utility = self._compute_category_utility(real_cat, gen_cat, mask)
        
        # Combine utility components with weights
        weighted_utility = -(
            self.w1 * spatial_utility +
            self.w2 * temporal_utility +
            self.w3 * category_utility
        )
        
        # Calculate and update overall utility score for adaptive weighting
        if hasattr(self, 'current_utility_score'):
            # Normalize to 0-1 range where 1 is perfect utility
            # Convert distances to similarities
            spatial_similarity = tf.exp(-spatial_utility)
            temporal_similarity = tf.exp(-temporal_utility)
            category_similarity = tf.exp(-category_utility)
            
            # Weighted average of similarities
            overall_similarity = (
                self.w1 * spatial_similarity +
                self.w2 * temporal_similarity +
                self.w3 * category_similarity
            ) / (self.w1 + self.w2 + self.w3)
            
            # Update current utility score (mean across batch)
            self.current_utility_score = tf.reduce_mean(overall_similarity).numpy()
            
            # Log to wandb if enabled
            if self.use_wandb and wandb is not None:
                wandb.log({
                    'overall_utility': self.current_utility_score,
                    'spatial_utility': tf.reduce_mean(spatial_similarity).numpy(),
                    'temporal_utility': tf.reduce_mean(temporal_similarity).numpy(),
                    'category_utility': tf.reduce_mean(category_similarity).numpy()
                })
        
        return weighted_utility
    
    def _compute_adversarial_reward(self, gen_trajs):
        """Compute adversarial reward using discriminator.
        
        R_adv = log D_φ(T_i)
        
        Higher values indicate better realism (higher probability of being real).
        """
        # Format trajectories for discriminator
        gen_inputs = [
            gen_trajs[0],  # lat_lon
            gen_trajs[1],  # category
            gen_trajs[2],  # day
            gen_trajs[3],  # hour
            gen_trajs[4]   # mask
        ]
        
        # Get discriminator probabilities
        d_probs = self.discriminator.predict(gen_inputs, verbose=False)
        
        # Clip probabilities to avoid log(0)
        d_probs = tf.clip_by_value(d_probs, 1e-7, 1.0)
        
        # Compute adversarial reward: log(p)
        return tf.math.log(d_probs)

    def train(self, epochs=2000, batch_size=32, sample_interval=10, early_stopping=True, patience=30, min_delta=0.001, start_epoch=0):
        """Train the model with progressive dataset size and reduced iterations.
        
        Args:
            epochs: Number of epochs to train for
            batch_size: Batch size for training
            sample_interval: Interval for generating samples
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            min_delta: Minimum improvement for early stopping
            start_epoch: Starting epoch number (for continuing training)
        """
        # Prepare data
        self.prepare_training_data()
        
        # Initialize weights and biases
        self.initialize_wandb()
        
        # Make sure we have properly initialized running statistics
        if not hasattr(self, 'utility_norm_stats'):
            self.utility_norm_stats = {
                'spatial_mean': 0.0,
                'spatial_std': 1.0,
                'temporal_mean': 0.0,
                'temporal_std': 1.0,
                'category_mean': 0.0,
                'category_std': 1.0,
                'update_count': 0
            }
        
        # Track current epoch
        self.current_epoch = start_epoch
        
        # Memory optimization: configure TensorFlow memory growth
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled for {len(gpus)} GPU(s)")
        except Exception as e:
            print(f"Could not configure GPU memory: {e}")
        
        # Set up early stopping
        best_reward = -float('inf')
        best_epoch = 0
        no_improvement = 0
        
        # Initialize metrics history for early stopping
        metrics_history = {
            'total_reward': [],
            'privacy_reward': [],
            'utility_reward': [],
        }
        
        # Setup for progressive training
        all_samples = len(self.X_train[0])
        print(f"Total available samples: {all_samples}")
        
        # Progressive training settings - gradually increase dataset size
        initial_percent = 0.1  # Start with 10% of data
        final_percent = 1.0    # End with 100% of data
        progressive_epochs = 20  # Number of epochs to reach full dataset
        
        # Progress tracking
        epoch_digits = len(str(epochs))
        
        # Start the training loop
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            
            # Skip every other epoch for the first 10 epochs (fast forward)
            if epoch < 10 and epoch % 2 != 0:
                print(f"Skipping epoch {epoch} (fast forward)")
                continue
                
            # Calculate percentage of data to use based on current epoch
            if epoch < progressive_epochs:
                data_percent = initial_percent + (final_percent - initial_percent) * (epoch / progressive_epochs)
            else:
                data_percent = final_percent
                
            # Calculate number of samples to use
            num_samples = int(all_samples * data_percent)
            
            # Print progress update
            print(f"Epoch {epoch+1:0{epoch_digits}d}/{epochs} - Using {num_samples} samples ({data_percent:.1%} of data)")
            
            # Update clip limits based on current epoch
            self.update_clip_limits(epoch)
            
            # Update component weights based on current epoch
            self.update_component_weights(epoch)
            
            # Shuffle indices for this epoch
            indices = np.random.permutation(all_samples)[:num_samples]
            epoch_data = [x[indices] for x in self.X_train]
            
            # Train on batches
            num_batches = num_samples // batch_size
            if num_batches == 0:
                num_batches = 1
                curr_batch_size = num_samples
            else:
                curr_batch_size = batch_size
                
            # Initialize metrics for this epoch
            epoch_metrics = {
                'g_loss': 0.0,
                'c_loss': 0.0,
                'd_loss': 0.0,
                'r_adv': 0.0,
                'r_util': 0.0,
                'r_priv': 0.0,
                'total_reward': 0.0,
            }
            
            # Train on all batches
            for batch_i in range(num_batches):
                start_idx = batch_i * curr_batch_size
                end_idx = min((batch_i + 1) * curr_batch_size, num_samples)
                batch_data = [d[start_idx:end_idx] for d in epoch_data]
                
                # Use simplified train_step with reduced PPO iterations
                metrics = self.train_step(batch_data, end_idx - start_idx)
                
                # Update epoch metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key] / num_batches
                
                if batch_i % max(1, num_batches // 5) == 0:
                    print(f"  Batch {batch_i+1}/{num_batches}: G_loss={metrics['g_loss']:.4f}, C_loss={metrics['c_loss']:.4f}")
            
            # Log metrics for this epoch
            log_data = {
                'epoch': epoch,
                'g_loss': epoch_metrics['g_loss'],
                'c_loss': epoch_metrics['c_loss'],
                'd_loss': epoch_metrics['d_loss'],
                'r_adv': epoch_metrics['r_adv'],
                'r_util': epoch_metrics['r_util'],
                'r_priv': epoch_metrics['r_priv'],
                'total_reward': epoch_metrics['total_reward'],
                'data_percent': data_percent
            }
            
            if self.use_wandb and wandb is not None:
                wandb.log(log_data)
                
            # Print epoch summary
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"D_real: {epoch_metrics.get('d_loss_real', 0):.4f}, D_fake: {epoch_metrics.get('d_loss_fake', 0):.4f}, G: {epoch_metrics['g_loss']:.4f}, C: {epoch_metrics['c_loss']:.4f}")
            print(f"Reward components - Adv: {epoch_metrics['r_adv']:.4f}, Util: {epoch_metrics['r_util']:.4f}, Priv: {epoch_metrics['r_priv']:.4f}")
            
            # Update history for early stopping
            metrics_history['total_reward'].append(epoch_metrics['total_reward'])
            metrics_history['privacy_reward'].append(epoch_metrics['r_priv'])
            metrics_history['utility_reward'].append(epoch_metrics['r_util'])
            
            # Save the model periodically
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                print(f"Model weights saved for epoch {epoch}")
                
            # Early stopping logic
            current_reward = epoch_metrics['total_reward']
            if current_reward > best_reward + min_delta:
                best_reward = current_reward
                best_epoch = epoch
                no_improvement = 0
                # Save best model
                self.save_checkpoint(epoch, best=True)
                print(f"New best model at epoch {epoch} with reward {best_reward:.4f}")
            else:
                no_improvement += 1
                if early_stopping and no_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
            # Periodic memory cleanup
            if epoch % 5 == 0:
                gc.collect()
                if tf.config.list_physical_devices('GPU'):
                    try:
                        tf.keras.backend.clear_session()
                    except:
                        pass
        
        # Training complete
        print(f"Training completed. Best model was at epoch {best_epoch} with reward {best_reward:.4f}")
        
        # Try to load the best model
        try:
            self.load_checkpoint(best_epoch, best=True)
            print(f"Loaded best model from epoch {best_epoch}")
        except:
            print("Could not load best model.")
            
        return {
            'best_epoch': best_epoch,
            'best_reward': best_reward,
            'epochs_trained': epoch - start_epoch + 1
        }

    def save_checkpoint(self, epoch, best=False, suffix=''):
        """Save model checkpoints.
        
        Args:
            epoch: Current epoch number
            best: Whether this is the best model so far
            suffix: Optional suffix to add to the filename (for multi-metric early stopping)
        """
        # Make sure the results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Build the prefix for the filename
        if best and suffix:
            prefix = f"best_{suffix}_"
        elif best:
            prefix = "best_"
        else:
            prefix = ""
        
        # Save model weights
        try:
            self.generator.save_weights(f'results/{prefix}generator_{epoch}.weights.h5')
            self.discriminator.save_weights(f'results/{prefix}discriminator_{epoch}.weights.h5')
            self.critic.save_weights(f'results/{prefix}critic_{epoch}.weights.h5')
            
            # Also save the most recent best model separately if this is a best model
            if best:
                self.generator.save_weights(f'results/{prefix}generator.weights.h5')
                self.discriminator.save_weights(f'results/{prefix}discriminator.weights.h5')
                self.critic.save_weights(f'results/{prefix}critic.weights.h5')
                
            print(f"Model weights saved for epoch {epoch}" + 
                  (" (best model)" if best and not suffix else "") +
                  (f" (best {suffix} model)" if best and suffix else ""))
        except Exception as e:
            print(f"Warning: Could not save weights for epoch {epoch}: {e}")
        
        # Now try to save the full models with architecture
        try:
            # Save the Keras models
            self.generator.save(f'results/{prefix}generator_architecture_{epoch}.keras')
            self.discriminator.save(f'results/{prefix}discriminator_architecture_{epoch}.keras')
            self.critic.save(f'results/{prefix}critic_architecture_{epoch}.keras')
            print(f"Model architectures saved for epoch {epoch}")
            
            # Also save the main model's configuration
            with open(f'results/{prefix}model_config_{epoch}.json', 'w') as f:
                json.dump(self.get_config(), f, indent=4)
            
        except Exception as e:
            print(f"Warning: Could not save full model architectures for epoch {epoch}: {e}")
            print("Only weights were saved. You'll need to recreate the model structure to load them.")

    def load_best_model(self):
        """Load the best model saved during training."""
        try:
            self.generator.load_weights('results/best_generator.weights.h5')
            self.discriminator.load_weights('results/best_discriminator.weights.h5')
            self.critic.load_weights('results/best_critic.weights.h5')
            print("Loaded best model weights")
        except Exception as e:
            print(f"Error loading best model: {e}")
            
    def load_checkpoint(self, epoch, best=False, suffix=''):
        """Load a specific checkpoint.
        
        Args:
            epoch: The epoch number to load
            best: Whether to load the best model
            suffix: Optional suffix in the filename (for multi-metric early stopping)
        """
        try:
            # Build the prefix for the filename
            if best and suffix:
                prefix = f"best_{suffix}_"
            elif best:
                prefix = "best_"
            elif suffix:
                prefix = f"{suffix}_"
            else:
                prefix = ""
                
            self.generator.load_weights(f'results/{prefix}generator_{epoch}.weights.h5')
            self.discriminator.load_weights(f'results/{prefix}discriminator_{epoch}.weights.h5')
            self.critic.load_weights(f'results/{prefix}critic_{epoch}.weights.h5')
            print(f"Loaded model weights from epoch {epoch}" + 
                  (" (best model)" if best and not suffix else "") +
                  (f" (best {suffix} model)" if best and suffix else ""))
        except Exception as e:
            print(f"Error loading model from epoch {epoch}: {e}")
            raise

    def update_actor(self, states, noise, advantages):
        """Update generator using PPO (Proximal Policy Optimization).
        
        Args:
            states: List of state tensors (should include all inputs including mask)
            noise: Random noise for the generator
            advantages: Advantage values from critic
            
        Returns:
            loss: The actor loss value
        """
        batch_size = states[0].shape[0]
        
        # Reduce batch size if very large to avoid OOM errors
        max_batch_size = 64  # Maximum batch size to process at once
        
        # Process in smaller batches if needed
        if batch_size > max_batch_size:
            # Split processing into smaller batches
            total_loss = 0
            num_batches = 0
            
            for i in range(0, batch_size, max_batch_size):
                end_idx = min(i + max_batch_size, batch_size)
                # Extract batch slices
                batch_states = [s[i:end_idx] for s in states]
                batch_noise = noise[i:end_idx] if isinstance(noise, tf.Tensor) else noise[i:end_idx]
                batch_advantages = advantages[i:end_idx]
                
                # Process this batch
                batch_loss = self._update_actor_batch(batch_states, batch_noise, batch_advantages)
                total_loss += batch_loss * (end_idx - i)
                num_batches += 1
            
            # Return average loss weighted by batch size
            return total_loss / batch_size
        else:
            # Process the entire batch at once if it's small enough
            return self._update_actor_batch(states, noise, advantages)
    
    def _update_actor_batch(self, states, noise, advantages):
        """Process a single batch for actor update that fits in memory."""
        # Convert noise to tensor if it's a numpy array
        if isinstance(noise, np.ndarray):
            noise = tf.convert_to_tensor(noise, dtype=tf.float32)
            
        # Prepare inputs for the generator - ensure all are tensors
        all_inputs = [tf.convert_to_tensor(x, dtype=tf.float32) if not isinstance(x, tf.Tensor) else x for x in states] + [noise]
        
        # Ensure advantages are properly formatted and normalized
        advantages = tf.cast(advantages, tf.float32)
        advantages = tf.clip_by_value(advantages, -10.0, 10.0)
        
        # Get the old predictions - predict once instead of calling repeatedly
        with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
            old_predictions = self.generator.predict(all_inputs, verbose=0)
            # Convert to tensors immediately
            old_predictions = [tf.convert_to_tensor(pred, dtype=tf.float32) for pred in old_predictions]
        
        # For PPO, we use a custom training loop to apply the PPO clipping objective
        with tf.GradientTape() as tape:
            # Forward pass through the generator
            new_predictions = self.generator(all_inputs, training=True)
            
            # Calculate the ratio of new and old policies using a simplified approach
            # Focus on key changes rather than computing detailed ratios
            ratios = []
            
            # For lat/lon coordinates (continuous values)
            coord_diff = tf.reduce_sum(tf.square(new_predictions[0] - old_predictions[0]), axis=[1, 2])
            coord_ratio = tf.exp(-0.01 * coord_diff)  # Scale factor to prevent extreme values
            ratios.append(tf.expand_dims(coord_ratio, axis=1))
            
            # For categorical outputs (simplified calculation)
            for i in range(1, 4):  # Indices for categorical features
                # Calculate KL divergence for each category
                old_probs = tf.clip_by_value(old_predictions[i], 1e-8, 1.0)
                new_probs = tf.clip_by_value(new_predictions[i], 1e-8, 1.0)
                
                # Use a simpler ratio calculation - just compare most likely categories
                old_max = tf.argmax(old_probs, axis=-1)
                new_max = tf.argmax(new_probs, axis=-1)
                # Count matches as ratio of 1.0, mismatches as 0.9
                match = tf.cast(tf.equal(old_max, new_max), tf.float32)
                cat_ratio = 0.9 + 0.1 * match
                # Take mean over sequence length
                seq_ratio = tf.reduce_mean(cat_ratio, axis=1)
                ratios.append(tf.expand_dims(seq_ratio, axis=1))
            
            # Combine ratios from all components by multiplication
            combined_ratio = ratios[0]
            for r in ratios[1:]:
                combined_ratio = combined_ratio * r
            
            # Clip the ratio as per PPO
            clipped_ratios = tf.clip_by_value(combined_ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            
            # Compute surrogate objective
            surrogate1 = combined_ratio * advantages
            surrogate2 = clipped_ratios * advantages
            surrogate_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Add entropy bonus for exploration (simplified for efficiency)
            entropy_loss = 0
            if self.c2 > 0:
                # Only calculate entropy for one categorical output to save computation
                probs = new_predictions[1]  # Use categories (index 1)
                entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=-1)
                entropy_loss = -self.c2 * tf.reduce_mean(entropy)
            
            # Total loss
            ppo_loss = surrogate_loss + entropy_loss
        
        # Get gradients and apply them with clipnorm
        grads = tape.gradient(ppo_loss, self.generator.trainable_variables)
        
        # Check for None gradients and replace with zeros
        grads = [tf.zeros_like(var) if grad is None else grad for grad, var in zip(grads, self.generator.trainable_variables)]
        
        # Clip gradients to prevent exploding gradients
        clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        # Apply gradients
        self.actor_optimizer.apply_gradients(zip(clipped_grads, self.generator.trainable_variables))
        
        # Return the loss value for tracking
        return ppo_loss.numpy()

    def compute_trajectory_ratio(self, new_predictions, old_predictions):
        """Compute the policy ratio between new and old policies for PPO.
        
        For trajectories, we need to calculate the ratio of probabilities under the new
        and old policies. We compute this separately for each component of the trajectory
        (coordinates, category, day, hour) and combine them.
        
        Args:
            new_predictions: List of tensors representing predictions from updated policy
            old_predictions: List of tensors representing predictions from previous policy
            
        Returns:
            ratios: Tensor of shape [batch_size, 1] with policy ratios
        """
        batch_size = new_predictions[0].shape[0]
        
        # Ensure all inputs are tensors
        for i, pred in enumerate(new_predictions):
            if not isinstance(pred, tf.Tensor):
                new_predictions[i] = tf.convert_to_tensor(pred, dtype=tf.float32)
                
        for i, pred in enumerate(old_predictions):
            if not isinstance(pred, tf.Tensor):
                old_predictions[i] = tf.convert_to_tensor(pred, dtype=tf.float32)
        
        # For continuous coordinates (lat/lon), use Gaussian log-likelihood differences
        # Assuming diagonal covariance with standard deviation 1.0
        coord_new = new_predictions[0]  # Shape: [batch_size, seq_len, 2]
        coord_old = old_predictions[0]  # Shape: [batch_size, seq_len, 2]
        
        # Squared difference between predictions (negative log-likelihood under Gaussian)
        sq_diff = tf.reduce_sum(tf.square(coord_new - coord_old), axis=-1, keepdims=True)
        
        # Convert to ratio (exponentiate negative log-likelihood difference)
        # Small differences -> ratio close to 1
        # Large differences -> ratio far from 1
        coord_ratio = tf.exp(-0.5 * sq_diff)  # Shape: [batch_size, seq_len, 1]
        
        # Process categorical distributions (category, day, hour)
        categorical_ratios = []
        for i in range(1, 4):  # Indices 1, 2, 3 for category, day, hour
            # Get probabilities under new and old policies
            new_probs = tf.clip_by_value(new_predictions[i], 1e-10, 1.0)
            old_probs = tf.clip_by_value(old_predictions[i], 1e-10, 1.0)
            
            # Calculate the KL divergence from old to new policy
            # We need this for each prediction in the sequence
            
            # For each category, calculate probability ratio
            prob_ratio = new_probs / old_probs  # [batch_size, seq_len, n_categories]
            
            # Weight by the new policy probabilities and sum across categories
            # This gives the expected ratio under the new policy
            expected_ratio = tf.reduce_sum(new_probs * prob_ratio, axis=-1, keepdims=True)
            categorical_ratios.append(expected_ratio)
            
        # Combine ratios from all components
        # Start with coordinate ratio
        combined_ratio = coord_ratio
        
        # Multiply by each categorical ratio
        for ratio in categorical_ratios:
            combined_ratio = combined_ratio * ratio
        
        # Apply mask to handle variable-length sequences
        mask = tf.cast(new_predictions[4], tf.float32)  # Shape: [batch_size, seq_len, 1]
        masked_ratio = combined_ratio * mask
        
        # Compute the cumulative product along the sequence length
        # We want the product of ratios for all steps in the trajectory
        log_ratio = tf.math.log(masked_ratio + 1e-10)
        sum_log_ratio = tf.reduce_sum(log_ratio, axis=1)  # Sum logs instead of multiplying ratios
        cumulative_ratio = tf.exp(sum_log_ratio)  # Convert back to ratio
        
        # Normalize by valid sequence length to get geometric mean
        seq_lengths = tf.reduce_sum(mask, axis=1) + 1e-10
        normalized_ratio = tf.pow(cumulative_ratio, 1.0 / seq_lengths)
        
        # Clip for stability
        final_ratio = tf.clip_by_value(normalized_ratio, 0.1, 10.0)
        
        return final_ratio

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
            marc_model.load_weights('MARC/weights/MARC_Weight.h5')
            
            # Add a predict method if not available
            if not hasattr(marc_model, 'predict'):
                def predict_wrapper(inputs, verbose=0):
                    """Wrapper to add predict method to MARC model.
                    
                    Args:
                        inputs: The input trajectory data
                        verbose: Verbosity level
                        
                    Returns:
                        The model's predictions
                    """
                    # Suppress TensorFlow warnings and info messages if verbose is False
                    if verbose == 0 or verbose is False:
                        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
                    
                    try:
                        # Try using the underlying model directly if available
                        if hasattr(marc_model, 'model') and hasattr(marc_model.model, 'predict'):
                            return marc_model.model.predict(inputs, verbose=verbose)
                        # Try calling the model directly
                        elif hasattr(marc_model, '__call__'):
                            return marc_model(inputs)
                        # Try specialized evaluation method if available
                        elif hasattr(marc_model, 'evaluate_trajectory'):
                            return marc_model.evaluate_trajectory(inputs)
                        else:
                            raise AttributeError("MARC model has no prediction capability")
                    finally:
                        # Reset verbosity level
                        if verbose == 0 or verbose is False:
                            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
                
                marc_model.predict = predict_wrapper
            
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

    def _compute_spatial_utility(self, real_lat_lon, gen_lat_lon, mask):
        """Compute spatial distance between real and generated trajectories.
        
        Args:
            real_lat_lon: Real trajectory coordinates, shape [batch_size, seq_len, 2]
            gen_lat_lon: Generated trajectory coordinates, shape [batch_size, seq_len, 2]
            mask: Sequence mask, shape [batch_size, seq_len, 1]
            
        Returns:
            spatial_distance: Tensor of shape [batch_size, 1] containing average Haversine distance
        """
        # Calculate squared Euclidean distance
        diff = gen_lat_lon - real_lat_lon
        squared_diff = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)
        
        # Mask the distances (zero out padding)
        mask_repeated = tf.repeat(mask, 1, axis=2)  # Repeat mask to match squared_diff shape
        masked_squared_diff = squared_diff * mask_repeated
        
        # Sum the masked distances
        total_masked_distance = tf.reduce_sum(masked_squared_diff, axis=1)
        
        # Normalize by sequence length
        seq_lengths = tf.reduce_sum(mask, axis=[1, 2])
        seq_lengths = tf.clip_by_value(seq_lengths, 1.0, float('inf'))  # Avoid division by zero
        avg_distance = total_masked_distance / tf.reshape(seq_lengths, [-1, 1])
        
        # Clip to reasonable range
        clipped_distance = tf.clip_by_value(avg_distance, 0.0, self.current_clip_limits['spatial'])
        
        return clipped_distance
        
    def _compute_temporal_utility(self, real_day, gen_day, real_hour, gen_hour, mask):
        """Compute temporal distance between real and generated trajectories.
        
        Args:
            real_day: Real trajectory day one-hot encodings, shape [batch_size, seq_len, 7]
            gen_day: Generated trajectory day probabilities, shape [batch_size, seq_len, 7]
            real_hour: Real trajectory hour one-hot encodings, shape [batch_size, seq_len, 24]
            gen_hour: Generated trajectory hour probabilities, shape [batch_size, seq_len, 24]
            mask: Sequence mask, shape [batch_size, seq_len, 1]
            
        Returns:
            temporal_distance: Tensor of shape [batch_size, 1] containing temporal distance
        """
        # Calculate day distance using cross-entropy
        day_cross_entropy = tf.reduce_sum(
            -real_day * tf.math.log(tf.clip_by_value(gen_day, 1e-10, 1.0)), 
            axis=-1, 
            keepdims=True
        )
        
        # Calculate hour distance using cross-entropy
        hour_cross_entropy = tf.reduce_sum(
            -real_hour * tf.math.log(tf.clip_by_value(gen_hour, 1e-10, 1.0)), 
            axis=-1, 
            keepdims=True
        )
        
        # Combine day and hour distances with equal weighting
        temporal_distance = 0.5 * day_cross_entropy + 0.5 * hour_cross_entropy
        
        # Mask the distances (zero out padding)
        masked_temporal_distance = temporal_distance * mask
        
        # Sum the masked distances
        total_masked_distance = tf.reduce_sum(masked_temporal_distance, axis=1)
        
        # Normalize by sequence length
        seq_lengths = tf.reduce_sum(mask, axis=[1, 2])
        seq_lengths = tf.clip_by_value(seq_lengths, 1.0, float('inf'))  # Avoid division by zero
        avg_distance = total_masked_distance / tf.reshape(seq_lengths, [-1, 1])
        
        # Clip to reasonable range
        clipped_distance = tf.clip_by_value(avg_distance, 0.0, self.current_clip_limits['temporal'])
        
        return clipped_distance
        
    def _compute_category_utility(self, real_cat, gen_cat, mask):
        """Compute semantic/category distance between real and generated trajectories.
        
        Args:
            real_cat: Real trajectory category one-hot encodings, shape [batch_size, seq_len, n_categories]
            gen_cat: Generated trajectory category probabilities, shape [batch_size, seq_len, n_categories]
            mask: Sequence mask, shape [batch_size, seq_len, 1]
            
        Returns:
            category_distance: Tensor of shape [batch_size, 1] containing category distance
        """
        # Calculate category distance using cross-entropy
        category_cross_entropy = tf.reduce_sum(
            -real_cat * tf.math.log(tf.clip_by_value(gen_cat, 1e-10, 1.0)), 
            axis=-1, 
            keepdims=True
        )
        
        # Mask the distances (zero out padding)
        masked_category_distance = category_cross_entropy * mask
        
        # Sum the masked distances
        total_masked_distance = tf.reduce_sum(masked_category_distance, axis=1)
        
        # Normalize by sequence length
        seq_lengths = tf.reduce_sum(mask, axis=[1, 2])
        seq_lengths = tf.clip_by_value(seq_lengths, 1.0, float('inf'))  # Avoid division by zero
        avg_distance = total_masked_distance / tf.reshape(seq_lengths, [-1, 1])
        
        # Clip to reasonable range
        clipped_distance = tf.clip_by_value(avg_distance, 0.0, self.current_clip_limits['category'])
        
        return clipped_distance

    def train_step(self, real_trajs, batch_size=256):
        # Make sure batch_size is reasonable
        if batch_size > 128:
            print(f"Warning: Large batch size ({batch_size}) detected. Using 64 instead to avoid memory issues.")
            batch_size = 64
            
        # Generate trajectories
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Use generator in prediction mode with lower verbosity
        with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
            gen_trajs = self.generator.predict([*real_trajs, noise], verbose=0)
        
        # Ensure consistent data types but avoid unnecessary conversions
        if not all(isinstance(tensor, tf.Tensor) for tensor in real_trajs):
            real_trajs = [tf.convert_to_tensor(tensor, tf.float32) for tensor in real_trajs]
        if not all(isinstance(tensor, tf.Tensor) for tensor in gen_trajs):
            gen_trajs = [tf.convert_to_tensor(tensor, tf.float32) for tensor in gen_trajs]
        
        # Update the custom loss with the current real and generated trajectories
        self.traj_loss.set_trajectories(real_trajs, gen_trajs)
        
        # Simplified reward computation (faster)
        # Skip full reward calculation in early epochs
        if self.current_epoch < 5:
            # Use placeholder rewards in very early epochs
            batch_shape = (batch_size, 1)
            rewards = tf.zeros(batch_shape, dtype=tf.float32)
            privacy_rewards = tf.ones(batch_shape, dtype=tf.float32) * 0.5
            utility_rewards = tf.ones(batch_shape, dtype=tf.float32) * -1.0
            adversarial_rewards = tf.ones(batch_shape, dtype=tf.float32) * -0.5
        else:
            # Compute rewards using the TUL classifier
            rewards, privacy_rewards, utility_rewards, adversarial_rewards = self.compute_rewards(
                real_trajs, gen_trajs, self.tul_classifier
            )
        
        # Calculate original utility components (simplified)
        spatial_loss_orig = self._compute_spatial_utility(real_trajs[0], gen_trajs[0], real_trajs[4])
        # Simplified temporal and category computation for early epochs to save time
        if self.current_epoch < 10:
            # Use simpler calculations in early epochs
            temporal_loss_orig = tf.reduce_mean(tf.square(real_trajs[2] - gen_trajs[2])) * 3.0  # Simple approximation
            category_loss_orig = tf.reduce_mean(tf.square(real_trajs[1] - gen_trajs[1])) * 2.0  # Simple approximation
        else:
            # Use full calculations in later epochs
            temporal_loss_orig = self._compute_temporal_utility(real_trajs[2], gen_trajs[2], real_trajs[3], gen_trajs[3], real_trajs[4])
            category_loss_orig = self._compute_category_utility(real_trajs[1], gen_trajs[1], real_trajs[4])
        
        # Logging is expensive - do it only periodically
        if self.use_wandb and wandb is not None and self.current_epoch % 5 == 0:
            wandb.log({
                'mean_total_reward': tf.reduce_mean(rewards).numpy(),
                'mean_privacy_reward': tf.reduce_mean(privacy_rewards).numpy(),
                'mean_utility_reward': tf.reduce_mean(utility_rewards).numpy(),
                'mean_adversarial_reward': tf.reduce_mean(adversarial_rewards).numpy()
            })
        
        # Compute advantages and returns for PPO efficiently
        with tf.device('/GPU:0' if len(tf.config.list_physical_devices('GPU')) > 0 else '/CPU:0'):
            values = self.critic.predict(real_trajs, verbose=0)
            values = tf.cast(values, tf.float32)
            advantages = compute_advantage(rewards, values, self.gamma, self.gae_lambda)
            returns = compute_returns(rewards, self.gamma)
        
        # Ensure returns has the same batch size as what the critic expects
        if returns.shape[0] != batch_size:
            returns = tf.reshape(returns, [batch_size, 1])
        
        # Update critic using returns with custom gradient calculation
        with tf.GradientTape() as critic_tape:
            pred_values = self.critic(real_trajs, training=True)
            critic_loss = tf.reduce_mean(tf.square(returns - pred_values))
            
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads = [tf.zeros_like(var) if grad is None else grad 
                       for grad, var in zip(critic_grads, self.critic.trainable_variables)]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        c_loss = critic_loss.numpy()
        
        # Use just 1 PPO iteration during early training, up to a maximum of 2
        # This significantly speeds up training
        ppo_epochs = 1
        if self.current_epoch >= 20:  # Only use multiple iterations after epoch 20
            ppo_epochs = min(2, self.ppo_epochs)
        
        a_loss = 0
        for _ in range(ppo_epochs):
            a_loss += self.update_actor(real_trajs, noise, advantages)
        a_loss /= max(1, ppo_epochs)
        
        # Update discriminator less frequently
        # In early epochs, update every 3 steps
        # In later epochs, update every 2 steps
        d_update_freq = 3 if self.current_epoch < 20 else 2
        
        if self.current_epoch % d_update_freq == 0:  # Only update sometimes
            # Discriminator update with explicit gradient calculation
            with tf.GradientTape() as disc_tape_real:
                real_pred = self.discriminator(real_trajs, training=True)
                real_labels = tf.ones_like(real_pred)
                d_loss_real_val = tf.keras.losses.binary_crossentropy(real_labels, real_pred)
                d_loss_real_val = tf.reduce_mean(d_loss_real_val)
                
            with tf.GradientTape() as disc_tape_fake:
                fake_pred = self.discriminator(gen_trajs, training=True)
                fake_labels = tf.zeros_like(fake_pred)
                d_loss_fake_val = tf.keras.losses.binary_crossentropy(fake_labels, fake_pred)
                d_loss_fake_val = tf.reduce_mean(d_loss_fake_val)
            
            # Apply gradients for real samples
            disc_grads_real = disc_tape_real.gradient(d_loss_real_val, self.discriminator.trainable_variables)
            disc_grads_real = [tf.zeros_like(var) if grad is None else grad 
                              for grad, var in zip(disc_grads_real, self.discriminator.trainable_variables)]
            
            # Apply gradients for fake samples
            disc_grads_fake = disc_tape_fake.gradient(d_loss_fake_val, self.discriminator.trainable_variables)
            disc_grads_fake = [tf.zeros_like(var) if grad is None else grad 
                              for grad, var in zip(disc_grads_fake, self.discriminator.trainable_variables)]
            
            # Combine gradients
            disc_grads = [(real_grad + fake_grad) * 0.5 for real_grad, fake_grad in zip(disc_grads_real, disc_grads_fake)]
            
            # Apply combined gradients
            self.discriminator_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
            
            d_loss_real = d_loss_real_val.numpy()
            d_loss_fake = d_loss_fake_val.numpy()
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
        else:
            # Skip discriminator update this step
            d_loss_real = 0
            d_loss_fake = 0
            d_loss = 0
        
        # Get TUL accuracy as a measure of privacy (only in later epochs)
        tul_accuracy = None
        if hasattr(self, 'current_tul_accuracy') and self.current_epoch >= 5:
            tul_accuracy = self.current_tul_accuracy
        
        # Return a consistent set of metrics
        return {
            'g_loss': a_loss,  # Actor (generator) loss
            'c_loss': c_loss,  # Critic loss
            'd_loss_real': d_loss_real,  # Discriminator loss on real data
            'd_loss_fake': d_loss_fake,  # Discriminator loss on fake data
            'r_adv': tf.reduce_mean(adversarial_rewards).numpy(),  # Mean adversarial reward
            'r_util': tf.reduce_mean(utility_rewards).numpy(),  # Mean utility reward
            'r_priv': tf.reduce_mean(privacy_rewards).numpy(),  # Mean privacy reward
            'total_reward': tf.reduce_mean(rewards).numpy(),  # Mean total reward
            'rewards_mean': tf.reduce_mean(rewards).numpy(),  # Same as total_reward for compatibility
            'rewards_std': tf.math.reduce_std(rewards).numpy(),  # Standard deviation of rewards
            'spatial_loss_orig': tf.reduce_mean(spatial_loss_orig).numpy(),  # Original spatial loss
            'temporal_loss_orig': tf.reduce_mean(temporal_loss_orig).numpy(),  # Original temporal loss
            'category_loss_orig': tf.reduce_mean(category_loss_orig).numpy(),  # Original category loss
            'spatial_component': tf.reduce_mean(self.w1 * spatial_loss_orig).numpy(),  # Weighted spatial component
            'temporal_component': tf.reduce_mean(self.w2 * temporal_loss_orig).numpy(),  # Weighted temporal component
            'category_component': tf.reduce_mean(self.w3 * category_loss_orig).numpy(),  # Weighted category component
            'gen_updates': ppo_epochs,  # Number of generator updates
            'tul_accuracy': tul_accuracy  # Current TUL accuracy (privacy metric)
        }

    def prepare_training_data(self):
        """Load and prepare training data."""
        # Training data
        x_train = np.load('data/final_train.npy', allow_pickle=True)
        self.x_train = x_train
        
        # Padding
        X_train = [pad_sequences(f, self.max_length, padding='pre', dtype='float64') 
                  for f in x_train]
        self.X_train = X_train
        
        # Check if we need to rebuild the model with correct input shapes
        needs_rebuild = False
        actual_shapes = {}
        
        for i, key in enumerate(self.keys):
            if key != 'mask':
                actual_shape = X_train[i].shape
                print(f"Data shape for {key}: {actual_shape}")
                if key == 'category' and actual_shape[2] != self.vocab_size[key]:
                    print(f"Mismatch for {key}: expected {self.vocab_size[key]}, got {actual_shape[2]}")
                    self.vocab_size[key] = actual_shape[2]
                    needs_rebuild = True
                actual_shapes[key] = actual_shape[2]
        
        # Rebuild the model if needed
        if needs_rebuild:
            print("Rebuilding model with correct input shapes...")
            # Save optimizer state
            optimizer_weights = None
            if hasattr(self, 'actor_optimizer') and hasattr(self.actor_optimizer, 'get_weights'):
                optimizer_weights = self.actor_optimizer.get_weights()
            
            # Rebuild models
            self.generator = self.build_generator()
            self.critic = self.build_critic()
            self.discriminator = self.build_discriminator()
            
            # Compile models
            self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer)
            self.critic.compile(loss='mse', optimizer=self.critic_optimizer)
            
            self.setup_combined_model()
            
            # Restore optimizer state if available
            if optimizer_weights is not None:
                self.actor_optimizer.set_weights(optimizer_weights)
                
            print("Model rebuilt successfully!")
    
    def initialize_wandb(self):
        """Initialize weights & biases for experiment tracking."""
        # Skip if wandb is not available
        if wandb is None:
            print("WandB not available. Logging will be disabled.")
            self.use_wandb = False
            return
            
        try:
            # Set flag for wandb usage
            self.use_wandb = True
            
            # Login and initialize
            wandb.login()  # This will use the API key from environment variable or prompt for login
            wandb.init(project="rl-transformer-trajgan", entity="xutao-henry-mao-vanderbilt-university",
                       config={
                           "epochs": 2000,
                           "batch_size": 32,
                           "latent_dim": self.latent_dim,
                           "max_length": self.max_length,
                           "gamma": self.gamma,
                           "gae_lambda": self.gae_lambda,
                           "clip_epsilon": self.clip_epsilon,
                           "w_adv": self.w_adv,
                           "w_util": self.w_util, 
                           "w_priv": self.w_priv,
                           "gen_updates_per_disc": self.gen_updates_per_disc,
                           # Component weights for utility calculation
                           "w1_spatial": self.w1,
                           "w2_temporal": self.w2,
                           "w3_category": self.w3,
                           # Curriculum learning parameters
                           "initial_spatial": self.initial_component_weights['spatial'],
                           "initial_temporal": self.initial_component_weights['temporal'],
                           "initial_category": self.initial_component_weights['category'],
                           "target_spatial": self.target_component_weights['spatial'],
                           "target_temporal": self.target_component_weights['temporal'],
                           "target_category": self.target_component_weights['category'],
                           "curriculum_start_epoch": self.curriculum_start_epoch,
                           "curriculum_duration": self.curriculum_duration,
                           # Reward weights
                           "alpha": self.alpha,
                           "beta": self.beta,
                           "gamma_adv": self.gamma_adv,
                           # Adaptive reward balancing
                           "privacy_target": self.privacy_target,
                           "utility_target": self.utility_target,
                           "alpha_min": self.alpha_min,
                           "alpha_max": self.alpha_max,
                           "beta_min": self.beta_min,
                           "beta_max": self.beta_max,
                           # Learning rates
                           "actor_lr": 0.0003,
                           "critic_lr": 0.0003,
                           "discriminator_lr": 0.0001,
                           # Dynamic clip limit parameters
                           "initial_clip_limits": self.initial_clip_limits,
                           "max_clip_limits": self.max_clip_limits,
                           "clip_increase_start_epoch": self.clip_increase_start_epoch,
                           "clip_increase_frequency": self.clip_increase_frequency,
                           "clip_increase_rate": self.clip_increase_rate
                       })

            # Create custom wandb panels for reward analysis
            wandb.define_metric("epoch")
            wandb.define_metric("reward_components/*", step_metric="epoch")
            wandb.define_metric("utility_components/*", step_metric="epoch")
            wandb.define_metric("reward_effectiveness/*", step_metric="epoch")
            wandb.define_metric("normalization/*", step_metric="epoch")
            wandb.define_metric("balance/*", step_metric="epoch")
            wandb.define_metric("clip_limits/*", step_metric="epoch")
            wandb.define_metric("curriculum/*", step_metric="epoch")
            wandb.define_metric("early_stopping/*", step_metric="epoch")
            
            print("WandB initialized successfully")
        except Exception as e:
            print(f"Error initializing WandB: {e}")
            print("WandB logging will be disabled")
            self.use_wandb = False