import tensorflow as tf
import keras
import numpy as np
import random
from tensorflow.keras import layers
import tensorflow_probability as tfp
import os
import json
import warnings

# Add wandb import with try-except to handle cases where it might not be available
try:
    import wandb
except ImportError:
    warnings.warn("wandb not installed. WandB logging will be disabled.")
    wandb = None

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

from keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, Reshape
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
        
        # RL parameters - adjusting for better performance
        self.gamma = 0.99  # discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.clip_epsilon = 0.2  # PPO clip parameter
        self.c1 = 0.3  # value function coefficient (reduced further to prevent over-conservatism)
        self.c2 = 0.01  # entropy coefficient (increased for more exploration in utility space)
        self.ppo_epochs = 4  # Number of PPO epochs per batch (increased back to encourage better policy updates)
        
        # Generator-Discriminator balance parameter - increase gen updates for better learning
        self.gen_updates_per_disc = 5  # Update generator this many times per discriminator update (increased further)
        
        # Dynamic clip limits - updated based on baseline analysis
        self.initial_clip_limits = {
            'spatial': 8.0,    # Increased to allow more spatial exploration
            'temporal': 8.0,   # Increased to allow more temporal flexibility
            'category': 8.0    # Increased to allow more category flexibility
        }
        self.max_clip_limits = {
            'spatial': 12.0,     # Increased to allow better learning
            'temporal': 20.0,    # Increased to allow better learning
            'category': 15.0     # Increased to allow better learning
        }
        self.clip_increase_start_epoch = 20  # Start increasing clips earlier (was 30)
        self.clip_increase_frequency = 8     # Increase clip limits more frequently (was 10)
        self.clip_increase_rate = 0.2        # Faster increase rate (was 0.15)
        self.current_clip_limits = self.initial_clip_limits.copy()
        self.current_epoch = 0  # Track current epoch for clip limit adjustment
        
        # Load or initialize TUL classifier for privacy rewards
        self.tul_classifier = self.load_tul_classifier()
        
        # Balanced reward weights - significantly adjusted to prioritize utility
        self.w_adv = 0.3    # Reduce adversarial weight
        self.w_util = 1.5   # Significantly increase utility weight
        self.w_priv = 0.2   # Keep privacy weight lower to avoid overwhelming utility
            
        # Initial utility component weights - more balanced from the start
        self.initial_component_weights = {
            'spatial': 0.8,     # Keep high emphasis on spatial
            'temporal': 0.7,    # Higher initial weight for temporal
            'category': 0.6     # Higher initial weight for category
        }
        
        # Target utility component weights - more balanced for better utility metrics
        self.target_component_weights = {
            'spatial': 0.7,     # Slightly lower spatial emphasis to balance with others
            'temporal': 0.8,    # Higher temporal weight for better temporal patterns
            'category': 0.7     # Higher category weight for better category matching
        }
        
        # Curriculum learning parameters - faster utility convergence
        self.curriculum_start_epoch = 20      # Start curriculum earlier
        self.curriculum_duration = 150        # Shorter transition period for faster adaptation
        
        # Current utility component weights (will be updated during training)
        self.current_component_weights = self.initial_component_weights.copy()
        
        # For backwards compatibility with existing code
        self.beta = self.initial_component_weights['spatial']
        self.gamma_temporal = self.initial_component_weights['temporal']
        self.chi = self.initial_component_weights['category']
        self.alpha = 0.2    # Privacy weight
        
        # Tracking variable for wandb usage
        self.use_wandb = False
        
        # Define optimizers with adjusted learning rates based on baseline performance
        self.actor_optimizer = Adam(0.0005, clipnorm=0.8)  # Increased from 0.0003
        self.critic_optimizer = Adam(0.0005, clipnorm=0.8)  # Increased from 0.0003
        self.discriminator_optimizer = Adam(0.0001, clipnorm=0.5)  # Increased from 0.00005

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
        
        # Embedding layers for each feature
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=128, activation='relu', use_bias=True,  # Increased to 128 (was 100)
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                stacked_latlon = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
                embeddings.append(stacked_latlon)
                inputs.append(i)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, activation='relu', use_bias=True,  # Increased to 64 (was less)
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_cat = [d(x) for x in unstacked]
                stacked_cat = Lambda(lambda x: tf.stack(x, axis=1))(dense_cat)
                embeddings.append(stacked_cat)
                inputs.append(i)
        
        # Combine all embeddings
        combined = Concatenate(axis=2)(embeddings)
        
        # Dense layer for noise input
        noise_dense = Dense(self.max_length * 32, activation='relu')(noise)  # Increased to 32 (was smaller)
        noise_reshape = Reshape((self.max_length, 32))(noise_dense)  # Adjust shape to match
        
        # Add noise to embeddings
        with_noise = Concatenate(axis=2)([combined, noise_reshape])
        
        # Apply Transformer blocks - deeper network
        x = with_noise
        
        # Stack of transformer blocks with increased capacity
        num_transformer_blocks = 3  # Reduced from 5
        embed_dim = x.shape[-1]
        
        for i in range(num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=embed_dim, 
                num_heads=8,  # Increased from fewer heads
                ff_dim=embed_dim * 4,  # Increased multiplier 
                rate=0.1 if i < num_transformer_blocks - 1 else 0.05  # Less dropout in final layer
            )(x, training=True)
        
        # Output layers for each feature
        outputs = []
        
        # Lat-lon output - improved precision
        lat_lon_out = TimeDistributed(Dense(self.vocab_size['lat_lon'], activation='tanh'))(x)  # tanh for bounded outputs
        outputs.append(lat_lon_out)
        
        # Day output
        day_out = TimeDistributed(Dense(self.vocab_size['day'], activation='softmax'))(x)
        outputs.append(day_out)
        
        # Hour output
        hour_out = TimeDistributed(Dense(self.vocab_size['hour'], activation='softmax'))(x)
        outputs.append(hour_out)
        
        # Category output
        cat_out = TimeDistributed(Dense(self.vocab_size['category'], activation='softmax'))(x)
        outputs.append(cat_out)
        
        # Mask output (pass-through)
        outputs.append(mask)
        
        # Create the model
        model = Model(inputs + [noise], outputs, name='generator')
        return model

    def build_critic(self):
        # Input Layer - similar to generator
        inputs = []
        embeddings = []
        
        # Prepare inputs for each feature
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                mask = Input(shape=(self.max_length, 1), name='input_mask')
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=128, activation='relu', use_bias=True,  # Increased to 128 (was 100)
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                stacked_latlon = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
                embeddings.append(stacked_latlon)
                inputs.append(i)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, activation='relu', use_bias=True,  # Increased to 64 (was less)
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_cat = [d(x) for x in unstacked]
                stacked_cat = Lambda(lambda x: tf.stack(x, axis=1))(dense_cat)
                embeddings.append(stacked_cat)
                inputs.append(i)
        
        # Combine all embeddings
        combined = Concatenate(axis=2)(embeddings)
        
        # Apply transformer blocks - similar to generator
        x = combined
        embed_dim = x.shape[-1]
        
        # Stack of transformer blocks with residual connections
        num_transformer_blocks = 4  # Slightly fewer than generator
        
        for i in range(num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=embed_dim, 
                num_heads=6,  # 6 heads 
                ff_dim=embed_dim * 3,  # 3x multiplier
                rate=0.1
            )(x, training=True)
        
        # Global average pooling to get sequence-level representation
        x = GlobalAveragePooling1D()(x)
        
        # Value output - deeper network
        x = Dense(256, activation='relu')(x)  # Increased from smaller size
        x = Dropout(0.2)(x)  # Added dropout for regularization
        x = Dense(128, activation='relu')(x)  # Additional layer
        x = Dropout(0.1)(x)  # Added dropout
        value = Dense(1)(x)
        
        # Create model
        model = Model(inputs, value, name='critic')
        return model

    def build_discriminator(self):
        # Create inputs for each feature - similar to generator and critic
        inputs = []
        embeddings = []
        
        # Prepare inputs for each feature
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                mask = Input(shape=(self.max_length, 1), name='input_mask')
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=128, activation='relu', use_bias=True,  # Increased to 128 (was 100) 
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_latlon = [d(x) for x in unstacked]
                stacked_latlon = Lambda(lambda x: tf.stack(x, axis=1))(dense_latlon)
                embeddings.append(stacked_latlon)
                inputs.append(i)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                unstacked = Lambda(lambda x: tf.unstack(x, axis=1))(i)
                d = Dense(units=64, activation='relu', use_bias=True,  # Increased to 64 (was less)
                         kernel_initializer=he_uniform(seed=1), name='emb_' + key)
                dense_cat = [d(x) for x in unstacked]
                stacked_cat = Lambda(lambda x: tf.stack(x, axis=1))(dense_cat)
                embeddings.append(stacked_cat)
                inputs.append(i)
        
        # Combine all embeddings
        combined = Concatenate(axis=2)(embeddings)
        
        # Apply transformer blocks
        x = combined
        embed_dim = x.shape[-1]
        
        # Stack of transformer blocks - deeper network for discriminator
        num_transformer_blocks = 6  # More layers for discriminator
        
        for i in range(num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=embed_dim, 
                num_heads=8,  # More attention heads
                ff_dim=embed_dim * 4,  # Larger feed-forward network
                rate=0.15  # Slightly more dropout
            )(x, training=True)
        
        # Global average pooling to get sequence-level representation
        x = GlobalAveragePooling1D()(x)
        
        # Multi-layer classification network
        x = Dense(256, activation='relu', kernel_regularizer=l1(0.0001))(x)  # Added L1 regularization
        x = Dropout(0.3)(x)  # More dropout
        x = Dense(128, activation='relu', kernel_regularizer=l1(0.0001))(x)
        x = Dropout(0.2)(x)
        validity = Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = Model(inputs, validity, name='discriminator')
        return model

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
        # Fix: Include all 5 inputs for the discriminator, including the mask (which is at gen_trajs[4])
        pred = self.discriminator(gen_trajs)
        
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
                  f"Spatial: {self.beta:.3f}, Temporal: {self.gamma_temporal:.3f}, Category: {self.chi:.3f}")
                  
        return weights_changed
    
    def compute_rewards(self, real_trajs, gen_trajs, tul_classifier):
        """Compute reward signals for generator actions.
        
        Args:
            real_trajs: List of real trajectory tensors
            gen_trajs: List of generated trajectory tensors
            tul_classifier: TUL classifier model or None
            
        Returns:
            rewards: Dictionary containing reward components
        """
        # Calculate adversarial rewards - no need to create a new list, use gen_trajs directly
        
        # Get discriminator predictions - pass the entire gen_trajs list directly
        d_pred = self.discriminator.predict(gen_trajs, verbose=0)
        
        # Calculate adversarial reward (higher when discriminator is fooled)
        adv_reward = -tf.math.log(1.0 - d_pred + 1e-8)
        
        # Calculate utility rewards for different components
        utility_rewards = {
            'spatial': self._compute_spatial_utility(real_trajs[0], gen_trajs[0], real_trajs[4]),
            'temporal': self._compute_temporal_utility(
                real_trajs[2], gen_trajs[2],  # day
                real_trajs[3], gen_trajs[3],  # hour
                real_trajs[4]  # mask
            ),
            'category': self._compute_category_utility(real_trajs[1], gen_trajs[1], real_trajs[4])
        }
        
        # Store original rewards for debugging and analysis
        utility_rewards_orig = {k: tf.identity(v) for k, v in utility_rewards.items()}
        
        # Apply clipping and normalization to utility rewards
        for component, reward in utility_rewards.items():
            # Clip rewards based on current limits
            clip_limit = self.current_clip_limits[component]
            
            # First simple clipping to prevent extreme values
            # This is essential to avoid outliers corrupting normalization
            reward_clipped = tf.clip_by_value(reward, -clip_limit, clip_limit)
            
            # Extract stats for normalization
            stat_key = f"{component}_mean"
            std_key = f"{component}_std"
            
            # Update running statistics with exponential moving average
            # Always update statistics to maintain good estimates
            current_mean = tf.reduce_mean(reward_clipped)
            current_std = tf.math.reduce_std(reward_clipped) + 1e-8  # Add epsilon to avoid division by zero
            
            if self.utility_norm_stats['update_count'] == 0:
                # First update
                self.utility_norm_stats[stat_key] = current_mean
                self.utility_norm_stats[std_key] = current_std
            else:
                # Use a smaller alpha for more stable estimates
                alpha = 0.03
                self.utility_norm_stats[stat_key] = (1 - alpha) * self.utility_norm_stats[stat_key] + alpha * current_mean
                self.utility_norm_stats[std_key] = (1 - alpha) * self.utility_norm_stats[std_key] + alpha * current_std
            
            # Apply normalization based on statistics
            if self.utility_norm_stats['update_count'] > 30:
                # Get mean and std from running statistics
                mean = self.utility_norm_stats[stat_key]
                std = self.utility_norm_stats[std_key]
                
                # Apply normalization only if standard deviation is significant
                if std > 0.1:  # Threshold to avoid over-normalization of already small values
                    # Calculate normalized rewards
                    normalized_reward = (reward_clipped - mean) / std
                    
                    # Blend normalized rewards with original clipped rewards
                    # Start with more weight on original, gradually shift to normalized
                    blend_factor = min(0.8, (self.utility_norm_stats['update_count'] - 30) / 100.0)
                    reward_processed = (blend_factor * normalized_reward + 
                                      (1 - blend_factor) * reward_clipped)
                    
                    # Apply final clipping after normalization
                    reward_processed = tf.clip_by_value(reward_processed, -clip_limit, clip_limit)
                else:
                    # If std is very small, just use clipped rewards
                    reward_processed = reward_clipped
            else:
                # In early training, just use clipped rewards
                reward_processed = reward_clipped
            
            # Store processed rewards
            utility_rewards[component] = reward_processed
        
        # Increment update counter
        self.utility_norm_stats['update_count'] += 1
        
        # Log current utility stats
        if self.utility_norm_stats['update_count'] % 50 == 0:
            print(f"Utility stats after {self.utility_norm_stats['update_count']} updates:")
            for key, value in self.utility_norm_stats.items():
                if key != 'update_count':
                    if isinstance(value, (tf.Tensor, tf.Variable)):
                        print(f"  {key}: {value.numpy():.4f}")
                    else:
                        print(f"  {key}: {value:.4f}")
        
        # Calculate weighted utility combination - apply progressive boosting to underperforming components
        # Get average values for each component to identify underperforming ones
        avg_spatial = tf.reduce_mean(tf.abs(utility_rewards['spatial']))
        avg_temporal = tf.reduce_mean(tf.abs(utility_rewards['temporal']))
        avg_category = tf.reduce_mean(tf.abs(utility_rewards['category']))
        
        # Compute component ratios to identify relative performance
        total_avg = avg_spatial + avg_temporal + avg_category + 1e-8
        ratios = {
            'spatial': avg_spatial / total_avg,
            'temporal': avg_temporal / total_avg,
            'category': avg_category / total_avg
        }
        
        # Apply adaptive boosting to underperforming components
        boost_factors = {}
        for component, ratio in ratios.items():
            # Boost components with lower ratios (underperforming)
            if ratio < 0.25:  # Significant underperformance
                boost_factors[component] = 1.4  # Strong boost
            elif ratio < 0.33:  # Moderate underperformance
                boost_factors[component] = 1.2  # Moderate boost
            else:
                boost_factors[component] = 1.0  # No boost
        
        # Apply boosting to component weights
        boosted_weights = {
            component: self.current_component_weights[component] * boost_factors.get(component, 1.0)
            for component in self.current_component_weights
        }
        
        # Combine utility rewards with boosted component weights
        combined_utility = (
            boosted_weights['spatial'] * utility_rewards['spatial'] +
            boosted_weights['temporal'] * utility_rewards['temporal'] +
            boosted_weights['category'] * utility_rewards['category']
        )
        
        # Compute privacy reward if TUL classifier is available
        privacy_reward = tf.zeros_like(adv_reward)
        if tul_classifier is not None:
            privacy_reward = self._compute_privacy_reward(gen_trajs, tul_classifier)
            # Clip privacy reward
            privacy_reward = tf.clip_by_value(privacy_reward, -5.0, 5.0)
        
        # Apply reward weights and combine all reward components
        # Add utility bias term to ensure minimum utility contribution
        utility_bias = 0.2  # Small bias to ensure utility signal always present
        final_reward = (
            self.w_adv * adv_reward +
            self.w_util * (combined_utility + utility_bias) +
            self.w_priv * privacy_reward
        )
        
        # Final clipping for stability
        final_reward = tf.clip_by_value(final_reward, -20.0, 20.0)
        
        # Return all reward components for logging and analysis
        return {
            'total': final_reward,
            'adversarial': adv_reward,
            'utility': combined_utility,
            'utility_biased': combined_utility + utility_bias,
            'spatial': utility_rewards['spatial'],
            'temporal': utility_rewards['temporal'],
            'category': utility_rewards['category'],
            'spatial_orig': utility_rewards_orig['spatial'],
            'temporal_orig': utility_rewards_orig['temporal'],
            'category_orig': utility_rewards_orig['category'],
            'spatial_boost': boosted_weights['spatial'],
            'temporal_boost': boosted_weights['temporal'],
            'category_boost': boosted_weights['category'],
            'privacy': privacy_reward
        }, utility_rewards_orig  # Return original rewards for logging and adaptive clipping

    def _compute_spatial_utility(self, real_latlon, gen_latlon, mask):
        """Compute spatial utility reward between real and generated lat/lon.
        
        Args:
            real_latlon: Real trajectory lat/lon tensor [batch_size, max_length, 2]
            gen_latlon: Generated trajectory lat/lon tensor [batch_size, max_length, 2]
            mask: Mask tensor [batch_size, max_length, 1]
            
        Returns:
            spatial_reward: Spatial utility reward tensor [batch_size, 1]
        """
        # Calculate trajectory length from mask
        traj_length = tf.reduce_sum(mask, axis=1) + 1e-8
        
        # Calculate point-wise distances between real and generated points
        diff = gen_latlon - real_latlon
        squared_diff = diff * diff
        
        # Apply mask to focus on valid points only
        mask_repeated = tf.repeat(mask, 2, axis=2)
        masked_squared_diff = squared_diff * mask_repeated
        
        # Sum across spatial dimensions and sequence length to get MSE
        batch_size = tf.shape(real_latlon)[0]
        spatial_mse = tf.reduce_sum(tf.reduce_sum(
            masked_squared_diff, axis=1), axis=1, keepdims=True) / traj_length
        
        # Calculate trajectory shape similarity (direction preservation)
        # This captures how well the generated trajectory preserves directional changes
        
        # 1. Calculate displacement vectors for consecutive points
        # For real trajectory
        real_shifted = tf.concat([real_latlon[:, 1:, :], tf.zeros_like(real_latlon[:, :1, :])], axis=1)
        real_vectors = real_shifted - real_latlon
        
        # For generated trajectory
        gen_shifted = tf.concat([gen_latlon[:, 1:, :], tf.zeros_like(gen_latlon[:, :1, :])], axis=1)
        gen_vectors = gen_shifted - gen_latlon
        
        # Create a mask for valid vector pairs (both points must be valid)
        mask_shifted = tf.concat([mask[:, 1:, :], tf.zeros_like(mask[:, :1, :])], axis=1)
        vector_mask = mask * mask_shifted  # Only points with valid next points
        vector_mask_repeated = tf.repeat(vector_mask, 2, axis=2)
        
        # Apply mask to vectors
        real_vectors_masked = real_vectors * vector_mask_repeated
        gen_vectors_masked = gen_vectors * vector_mask_repeated
        
        # 2. Calculate vector magnitudes
        real_magnitudes = tf.sqrt(tf.reduce_sum(real_vectors_masked * real_vectors_masked, axis=2, keepdims=True) + 1e-8)
        gen_magnitudes = tf.sqrt(tf.reduce_sum(gen_vectors_masked * gen_vectors_masked, axis=2, keepdims=True) + 1e-8)
        
        # 3. Normalize vectors to get directions
        real_directions = real_vectors_masked / (real_magnitudes + 1e-8)
        gen_directions = gen_vectors_masked / (gen_magnitudes + 1e-8)
        
        # 4. Calculate dot product to get cosine similarity
        cosine_sim = tf.reduce_sum(real_directions * gen_directions, axis=2, keepdims=True)
        
        # Apply vector mask again to ensure we're only considering valid points
        cosine_sim_masked = cosine_sim * tf.squeeze(vector_mask, axis=2, name='squeezed_vector_mask')[:, :, tf.newaxis]
        
        # Average cosine similarity across the trajectory
        valid_vector_count = tf.reduce_sum(vector_mask, axis=[1, 2]) + 1e-8
        avg_cosine_sim = tf.reduce_sum(cosine_sim_masked, axis=1) / valid_vector_count[:, tf.newaxis]
        
        # Convert to a directional similarity score (-1 to 1)
        # 1 means perfect directional alignment, -1 means opposite directions
        direction_score = avg_cosine_sim
        
        # Higher reward for lower error (negative MSE) and better direction preservation
        # Balance point-wise accuracy with trajectory shape preservation
        point_weight = 0.7
        direction_weight = 0.3
        
        # Compute the spatial reward combining both aspects
        spatial_reward_points = -spatial_mse
        spatial_reward_direction = direction_score  # Already in -1 to 1 range
        
        # Combine the rewards
        spatial_reward = (
            point_weight * spatial_reward_points + 
            direction_weight * spatial_reward_direction
        )
        
        # Apply scaling based on point density - reward denser trajectories
        point_density = traj_length / tf.cast(tf.shape(mask)[1], tf.float32)
        density_factor = 1.0 + 0.5 * point_density  # More weight for denser trajectories
        scaled_spatial_reward = spatial_reward * density_factor
        
        return scaled_spatial_reward

    def _compute_temporal_utility(self, real_day, gen_day, real_hour, gen_hour, mask):
        """Compute temporal utility reward for day and hour distributions.
        
        Args:
            real_day: Real trajectory day tensor [batch_size, max_length, 7]
            gen_day: Generated trajectory day tensor [batch_size, max_length, 7]
            real_hour: Real trajectory hour tensor [batch_size, max_length, 24]
            gen_hour: Generated trajectory hour tensor [batch_size, max_length, 24]
            mask: Mask tensor [batch_size, max_length, 1]
            
        Returns:
            temporal_reward: Temporal utility reward tensor [batch_size, 1]
        """
        # Calculate trajectory length from mask
        traj_length = tf.reduce_sum(mask, axis=1) + 1e-8
        
        # Calculate cross-entropy for days and hours (point-wise accuracy)
        gen_day_clipped = tf.clip_by_value(gen_day, 1e-7, 1.0)
        day_ce = tf.keras.losses.categorical_crossentropy(real_day, gen_day_clipped)
        
        gen_hour_clipped = tf.clip_by_value(gen_hour, 1e-7, 1.0)
        hour_ce = tf.keras.losses.categorical_crossentropy(real_hour, gen_hour_clipped)
        
        # Apply mask to focus on valid points only
        mask_flat = tf.squeeze(mask, axis=2)
        day_ce_masked = day_ce * mask_flat
        hour_ce_masked = hour_ce * mask_flat
        
        # Calculate average CE loss per trajectory
        day_ce_avg = tf.reduce_sum(day_ce_masked, axis=1, keepdims=True) / traj_length
        hour_ce_avg = tf.reduce_sum(hour_ce_masked, axis=1, keepdims=True) / traj_length
        
        # Compute trajectory-level distribution similarity
        # This captures if the overall temporal pattern (regardless of order) is preserved
        
        # 1. Calculate trajectory-level distribution of days and hours
        # For real trajectory
        real_day_dist = tf.reduce_sum(real_day * mask_flat[:, :, tf.newaxis], axis=1)
        real_day_dist = real_day_dist / (tf.reduce_sum(real_day_dist, axis=1, keepdims=True) + 1e-8)
        
        real_hour_dist = tf.reduce_sum(real_hour * mask_flat[:, :, tf.newaxis], axis=1)
        real_hour_dist = real_hour_dist / (tf.reduce_sum(real_hour_dist, axis=1, keepdims=True) + 1e-8)
        
        # For generated trajectory
        gen_day_dist = tf.reduce_sum(gen_day * mask_flat[:, :, tf.newaxis], axis=1)
        gen_day_dist = gen_day_dist / (tf.reduce_sum(gen_day_dist, axis=1, keepdims=True) + 1e-8)
        
        gen_hour_dist = tf.reduce_sum(gen_hour * mask_flat[:, :, tf.newaxis], axis=1)
        gen_hour_dist = gen_hour_dist / (tf.reduce_sum(gen_hour_dist, axis=1, keepdims=True) + 1e-8)
        
        # 2. Calculate JS divergence (symmetric KL) between distributions
        # For days
        gen_day_dist_clipped = tf.clip_by_value(gen_day_dist, 1e-7, 1.0)
        real_day_dist_clipped = tf.clip_by_value(real_day_dist, 1e-7, 1.0)
        
        m_day_dist = 0.5 * (gen_day_dist_clipped + real_day_dist_clipped)
        kl_day_real_m = tf.reduce_sum(real_day_dist_clipped * tf.math.log(real_day_dist_clipped / m_day_dist + 1e-8), axis=1, keepdims=True)
        kl_day_gen_m = tf.reduce_sum(gen_day_dist_clipped * tf.math.log(gen_day_dist_clipped / m_day_dist + 1e-8), axis=1, keepdims=True)
        js_day = 0.5 * (kl_day_real_m + kl_day_gen_m)
        
        # For hours
        gen_hour_dist_clipped = tf.clip_by_value(gen_hour_dist, 1e-7, 1.0)
        real_hour_dist_clipped = tf.clip_by_value(real_hour_dist, 1e-7, 1.0)
        
        m_hour_dist = 0.5 * (gen_hour_dist_clipped + real_hour_dist_clipped)
        kl_hour_real_m = tf.reduce_sum(real_hour_dist_clipped * tf.math.log(real_hour_dist_clipped / m_hour_dist + 1e-8), axis=1, keepdims=True)
        kl_hour_gen_m = tf.reduce_sum(gen_hour_dist_clipped * tf.math.log(gen_hour_dist_clipped / m_hour_dist + 1e-8), axis=1, keepdims=True)
        js_hour = 0.5 * (kl_hour_real_m + kl_hour_gen_m)
        
        # 3. Calculate temporal sequence consistency
        # Create one-hot encoding of most likely day and hour for each point
        real_day_idx = tf.argmax(real_day, axis=2)
        gen_day_idx = tf.argmax(gen_day, axis=2)
        
        real_hour_idx = tf.argmax(real_hour, axis=2)
        gen_hour_idx = tf.argmax(gen_hour, axis=2)
        
        # Calculate match ratios for consecutive points
        # For days
        day_match = tf.cast(tf.equal(real_day_idx, gen_day_idx), tf.float32)
        day_match_masked = day_match * mask_flat
        
        # For hours
        hour_match = tf.cast(tf.equal(real_hour_idx, gen_hour_idx), tf.float32)
        hour_match_masked = hour_match * mask_flat
        
        # Calculate consecutive match consistency
        # This rewards maintaining the same temporal pattern/sequence
        day_match_shift = tf.concat([day_match_masked[:, 1:], tf.zeros_like(day_match_masked[:, :1])], axis=1)
        hour_match_shift = tf.concat([hour_match_masked[:, 1:], tf.zeros_like(hour_match_masked[:, :1])], axis=1)
        
        day_consistency = tf.reduce_sum(day_match_masked * day_match_shift, axis=1, keepdims=True) / (traj_length - 1 + 1e-8)
        hour_consistency = tf.reduce_sum(hour_match_masked * hour_match_shift, axis=1, keepdims=True) / (traj_length - 1 + 1e-8)
        
        # Combine all components with appropriate weights
        # 1. Point-wise accuracy (negative CE)
        point_accuracy = -(0.35 * day_ce_avg + 0.65 * hour_ce_avg)  # Hours are more important than days
        
        # 2. Distribution match (negative JS divergence)
        dist_match = -(0.35 * js_day + 0.65 * js_hour)
        
        # 3. Sequence consistency
        seq_consistency = 0.35 * day_consistency + 0.65 * hour_consistency
        
        # Combine all temporal aspects with weights
        temporal_reward = (
            0.4 * point_accuracy +   # Point accuracy
            0.4 * dist_match +       # Distribution match 
            0.2 * seq_consistency    # Sequence consistency
        )
        
        # Apply scaling based on trajectory temporal diversity
        # Compute day and hour diversity metrics from real data
        day_entropy = -tf.reduce_sum(real_day_dist_clipped * tf.math.log(real_day_dist_clipped + 1e-8), axis=1, keepdims=True)
        hour_entropy = -tf.reduce_sum(real_hour_dist_clipped * tf.math.log(real_hour_dist_clipped + 1e-8), axis=1, keepdims=True)
        
        # Normalize entropy to 0-1 range for scaling
        day_entropy_norm = day_entropy / tf.math.log(tf.cast(7.0, tf.float32))  # 7 days
        hour_entropy_norm = hour_entropy / tf.math.log(tf.cast(24.0, tf.float32))  # 24 hours
        
        # More diverse temporal patterns should get higher rewards
        complexity_factor = 1.0 + 0.3 * (0.4 * day_entropy_norm + 0.6 * hour_entropy_norm)
        scaled_temporal_reward = temporal_reward * complexity_factor
        
        return scaled_temporal_reward

    def _compute_category_utility(self, real_cat, gen_cat, mask):
        """Compute category utility reward between real and generated categories.
        
        Args:
            real_cat: Real trajectory category tensor [batch_size, max_length, num_categories]
            gen_cat: Generated trajectory category tensor [batch_size, max_length, num_categories]
            mask: Mask tensor [batch_size, max_length, 1]
            
        Returns:
            category_reward: Category utility reward tensor [batch_size, 1]
        """
        # Calculate trajectory length from mask
        traj_length = tf.reduce_sum(mask, axis=1) + 1e-8
        
        # Calculate point-wise cross-entropy for categories
        gen_cat_clipped = tf.clip_by_value(gen_cat, 1e-7, 1.0)
        cat_ce = tf.keras.losses.categorical_crossentropy(real_cat, gen_cat_clipped)
        
        # Apply mask to focus on valid points only
        mask_flat = tf.squeeze(mask, axis=2)
        cat_ce_masked = cat_ce * mask_flat
        
        # Calculate average CE loss per trajectory
        cat_ce_avg = tf.reduce_sum(cat_ce_masked, axis=1, keepdims=True) / traj_length
        
        # Point-wise accuracy component (negative CE)
        point_accuracy = -cat_ce_avg
        
        # Calculate trajectory-level category distribution
        # For real trajectory
        real_cat_dist = tf.reduce_sum(real_cat * mask_flat[:, :, tf.newaxis], axis=1)
        real_cat_dist = real_cat_dist / (tf.reduce_sum(real_cat_dist, axis=1, keepdims=True) + 1e-8)
        
        # For generated trajectory
        gen_cat_dist = tf.reduce_sum(gen_cat * mask_flat[:, :, tf.newaxis], axis=1)
        gen_cat_dist = gen_cat_dist / (tf.reduce_sum(gen_cat_dist, axis=1, keepdims=True) + 1e-8)
        
        # Calculate JS divergence for category distributions
        gen_cat_dist_clipped = tf.clip_by_value(gen_cat_dist, 1e-7, 1.0)
        real_cat_dist_clipped = tf.clip_by_value(real_cat_dist, 1e-7, 1.0)
        
        m_cat_dist = 0.5 * (gen_cat_dist_clipped + real_cat_dist_clipped)
        kl_cat_real_m = tf.reduce_sum(real_cat_dist_clipped * tf.math.log(real_cat_dist_clipped / m_cat_dist + 1e-8), axis=1, keepdims=True)
        kl_cat_gen_m = tf.reduce_sum(gen_cat_dist_clipped * tf.math.log(gen_cat_dist_clipped / m_cat_dist + 1e-8), axis=1, keepdims=True)
        js_cat = 0.5 * (kl_cat_real_m + kl_cat_gen_m)
        
        # Distribution match component (negative JS divergence)
        dist_match = -js_cat
        
        # Calculate category sequence transitions
        # Get most likely category for each point
        real_cat_idx = tf.argmax(real_cat, axis=2)
        gen_cat_idx = tf.argmax(gen_cat, axis=2)
        
        # First, calculate the accuracy of category prediction
        cat_match = tf.cast(tf.equal(real_cat_idx, gen_cat_idx), tf.float32)
        cat_match_masked = cat_match * mask_flat
        cat_match_ratio = tf.reduce_sum(cat_match_masked, axis=1, keepdims=True) / traj_length
        
        # Calculate transition matrices for real and generated trajectories
        # This captures the sequential patterns of category changes
        
        # For real trajectory
        num_categories = tf.shape(real_cat)[2]
        
        # Create one-hot encodings for current and next categories
        real_cat_one_hot = tf.one_hot(real_cat_idx, num_categories)
        real_cat_next_one_hot = tf.concat([real_cat_one_hot[:, 1:], tf.zeros_like(real_cat_one_hot[:, :1])], axis=1)
        
        # Apply mask for valid transitions (both current and next point must be valid)
        mask_next = tf.concat([mask_flat[:, 1:], tf.zeros_like(mask_flat[:, :1])], axis=1)
        transition_mask = mask_flat * mask_next
        
        # Apply mask to one-hot encodings
        real_cat_one_hot_masked = real_cat_one_hot * mask_flat[:, :, tf.newaxis]
        real_cat_next_one_hot_masked = real_cat_next_one_hot * mask_next[:, :, tf.newaxis]
        
        # Compute transition probabilities
        # For each category i, calculate P(next=j | current=i) for all j
        # We'll simplify and just measure if the transitions match in pattern
        
        # Calculate consecutive match consistency
        cat_match_shift = tf.concat([cat_match_masked[:, 1:], tf.zeros_like(cat_match_masked[:, :1])], axis=1)
        transition_consistency = tf.reduce_sum(cat_match_masked * cat_match_shift * transition_mask, axis=1, keepdims=True) / (tf.reduce_sum(transition_mask, axis=1, keepdims=True) + 1e-8)
        
        # Compute category frequency similarity
        # This captures how well the frequency of each category's appearances is preserved
        
        # Calculate category frequency profiles
        real_cat_freq = tf.reduce_sum(real_cat_one_hot_masked, axis=1) / (traj_length[:, tf.newaxis] + 1e-8)
        gen_cat_freq = tf.reduce_sum(tf.one_hot(gen_cat_idx, num_categories) * mask_flat[:, :, tf.newaxis], axis=1) / (traj_length[:, tf.newaxis] + 1e-8)
        
        # Calculate L1 distance between frequency profiles
        freq_diff = tf.abs(gen_cat_freq - real_cat_freq)
        freq_l1_dist = tf.reduce_sum(freq_diff, axis=1, keepdims=True)
        
        # Convert to similarity measure (higher is better)
        freq_similarity = 1.0 - tf.minimum(freq_l1_dist, 1.0)  # Cap at 1.0 for extreme cases
        
        # Combine all components with weights
        category_reward = (
            0.4 * point_accuracy +        # Point-wise accuracy
            0.3 * dist_match +            # Distribution match
            0.15 * transition_consistency + # Transition pattern match
            0.15 * freq_similarity        # Frequency profile match
        )
        
        # Apply scaling based on category diversity
        cat_entropy = -tf.reduce_sum(real_cat_dist_clipped * tf.math.log(real_cat_dist_clipped + 1e-8), axis=1, keepdims=True)
        
        # Normalize entropy to 0-1 range
        cat_entropy_norm = cat_entropy / tf.math.log(tf.cast(tf.cast(num_categories, tf.float32), tf.float32))
        
        # Scale reward by diversity (higher for more diverse category distributions)
        diversity_factor = 1.0 + 0.4 * cat_entropy_norm
        scaled_category_reward = category_reward * diversity_factor
        
        return scaled_category_reward

    def train_step(self, real_trajs, batch_size=256):
        # Generate trajectories
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        gen_trajs = self.generator.predict([*real_trajs, noise])
        
        # Ensure consistent data types
        real_trajs = [tf.cast(tensor, tf.float32) for tensor in real_trajs]
        gen_trajs = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
        
        # Update the custom loss with the current real and generated trajectories
        self.traj_loss.set_trajectories(real_trajs, gen_trajs)
        
        # Compute full rewards using the TUL classifier
        rewards_dict, utility_rewards_orig = self.compute_rewards(real_trajs, gen_trajs, self.tul_classifier)
        rewards = rewards_dict['total']  # Extract total rewards
        
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
        
        # Update discriminator with two-step process - first basic update
        d_loss_real = self.discriminator.train_on_batch(
            real_trajs,  # Pass all inputs including mask
            np.ones((batch_size, 1))
        )
        d_loss_fake = self.discriminator.train_on_batch(
            gen_trajs,  # Pass all inputs including mask
            np.zeros((batch_size, 1))
        )
        
        # Determine if any utility components are underperforming based on original rewards
        utility_component_below_threshold = False
        for component, reward in utility_rewards_orig.items():
            mean_reward = tf.reduce_mean(reward)
            if mean_reward < -5.0:  # If a component has very negative rewards
                utility_component_below_threshold = True
                print(f"Utility component {component} is underperforming with mean reward {mean_reward:.4f}")
                break
        
        # Update generator (actor) multiple times for each discriminator update
        # Use more updates if utility components are underperforming
        g_updates = self.gen_updates_per_disc
        if utility_component_below_threshold:
            g_updates += 2  # Add extra updates to improve utility
            print(f"Increasing generator updates to {g_updates} due to underperforming utility")
        
        g_loss_total = 0
        for i in range(g_updates):
            # Generate new noise for each update to increase diversity
            noise_gen = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # For the first update, use the already computed advantages
            if i == 0:
                g_loss = self.update_actor(real_trajs, gen_trajs, advantages)
            else:
                # For subsequent updates, generate new trajectories and compute new rewards/advantages
                gen_trajs_new = self.generator.predict([*real_trajs, noise_gen])
                gen_trajs_new = [tf.cast(tensor, tf.float32) for tensor in gen_trajs_new]
                
                # Compute rewards for the new trajectories
                rewards_dict_new, _ = self.compute_rewards(real_trajs, gen_trajs_new, self.tul_classifier)
                rewards_new = rewards_dict_new['total']
                
                # Compute advantages for the new trajectories
                values_new = self.critic.predict(real_trajs[:4])
                values_new = tf.cast(values_new, tf.float32)
                advantages_new = compute_advantage(rewards_new, values_new, self.gamma, self.gae_lambda)
                
                # Adaptively weight utility-focused advantages
                if utility_component_below_threshold and i >= 2:
                    # For later updates, focus more on utility components
                    utility_advantages = compute_advantage(
                        rewards_dict_new['utility_biased'], values_new, self.gamma, self.gae_lambda)
                    # Blend utility and total advantages
                    blend_factor = 0.6  # 60% utility, 40% total rewards
                    advantages_new = (blend_factor * utility_advantages + 
                                     (1.0 - blend_factor) * advantages_new)
                    print(f"Update {i}: Using utility-focused advantages (blend factor: {blend_factor})")
                
                # Update generator with new advantages
                g_loss = self.update_actor(real_trajs, gen_trajs_new, advantages_new)
                
                # Update critic with new returns if this isn't the last generator update
                if i < g_updates - 1:
                    returns_new = compute_returns(rewards_new, self.gamma)
                    if returns_new.shape[0] != batch_size:
                        returns_new = tf.reshape(returns_new, [batch_size, 1])
                    c_loss_new = self.critic.train_on_batch(real_trajs[:4], returns_new)
                    # Average critic losses
                    c_loss = (c_loss + c_loss_new) / 2
            
            # Track total generator loss
            g_loss_total += g_loss
        
        # Average the generator loss over all updates
        g_loss = g_loss_total / g_updates
        
        # After generator updates, do a focused discriminator update to maintain balance
        if utility_component_below_threshold:
            # Generate new trajectories for an extra discriminator update
            noise_disc = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_trajs_disc = self.generator.predict([*real_trajs, noise_disc])
            
            # Extra discriminator update focused on newest generated samples
            d_loss_fake_extra = self.discriminator.train_on_batch(
                gen_trajs_disc,
                np.zeros((batch_size, 1))
            )
            
            # Average the fake losses
            d_loss_fake = (d_loss_fake + d_loss_fake_extra) / 2
        
        # Add the reward components to the metrics
        metrics = {
            "d_loss_real": d_loss_real, 
            "d_loss_fake": d_loss_fake, 
            "g_loss": g_loss, 
            "c_loss": c_loss,
            "gen_updates": g_updates  # Track number of generator updates
        }
        
        # Add reward component metrics
        for key, value in rewards_dict.items():
            if isinstance(value, tf.Tensor):
                metrics[key] = value.numpy().mean()
            else:
                metrics[key] = value
        
        # Add original utility metrics for monitoring clip limits
        for component in ['spatial', 'temporal', 'category']:
            orig_key = f"{component}_orig"
            if orig_key in rewards_dict and isinstance(rewards_dict[orig_key], tf.Tensor):
                metrics[f"{component}_loss_orig"] = tf.reduce_mean(tf.abs(rewards_dict[orig_key])).numpy()
                
        return metrics

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
                           "beta_initial": self.initial_component_weights['spatial'],
                           "gamma_temporal_initial": self.initial_component_weights['temporal'],
                           "chi_initial": self.initial_component_weights['category'],
                           "beta_target": self.target_component_weights['spatial'],
                           "gamma_temporal_target": self.target_component_weights['temporal'],
                           "chi_target": self.target_component_weights['category'],
                           "curriculum_start_epoch": self.curriculum_start_epoch,
                           "curriculum_duration": self.curriculum_duration,
                           "alpha": self.alpha,
                           # Store the learning rates directly rather than trying to extract from optimizer
                           "actor_lr": 0.0003,  # Hardcoded value from __init__
                           "critic_lr": 0.0003,  # Hardcoded value from __init__
                           "discriminator_lr": 0.00005,  # Hardcoded value from __init__
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

    def train(self, epochs=2000, batch_size=32, sample_interval=10, early_stopping=True, patience=30, min_delta=0.001, start_epoch=0):
        """Train the model.
        
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
            
        # Initialize metrics history for early stopping
        metrics_history = {
            'g_loss': [],
            'd_loss': [],
            'c_loss': [],
            'spatial_loss_orig': [],
            'temporal_loss_orig': [],
            'category_loss_orig': [],
            'tul_accuracy': []
        }
        
        # Initialize best metrics for early stopping
        best_metrics = {
            'g_loss': float('inf'),
            'balanced_score': 0.0,
            'utility_improvement': 0.0,
            'privacy_score': 0.0
        }
        
        # Initialize best epochs
        best_epochs = {
            'g_loss': -1,
            'balanced_score': -1,
            'utility_improvement': -1,
            'privacy_score': -1
        }
        
        # Initialize no improvement counters
        no_improvement_counts = {
            'g_loss': 0,
            'balanced_score': 0,
            'utility_improvement': 0,
            'privacy_score': 0
        }
        
        # Early stopping window size
        history_window = 20
        
        # Minimum epochs before allowing early stopping
        # This ensures we don't stop before curriculum learning and clip limit adjustments take effect
        min_epochs_before_stopping = max(
            self.curriculum_start_epoch + self.curriculum_duration,
            self.clip_increase_start_epoch + 2 * self.clip_increase_frequency
        ) + 20  # Add margin for adjustments to stabilize
        
        # Add method to analyze clip limit impact
        def analyze_clip_limit_impact(epoch):
            """Analyze and log the impact of clip limits on training."""
            if 'temporal_loss_orig' in metrics and 'category_loss_orig' in metrics:
                temporal_pct = metrics['temporal_loss_orig'] / self.current_clip_limits['temporal'] * 100
                category_pct = metrics['category_loss_orig'] / self.current_clip_limits['category'] * 100
                
                temporal_clipped = temporal_pct >= 99.0
                category_clipped = category_pct >= 99.0
                
                if temporal_clipped or category_clipped:
                    components_clipped = []
                    if temporal_clipped:
                        components_clipped.append(f"Temporal ({temporal_pct:.1f}%)")
                    if category_clipped:
                        components_clipped.append(f"Category ({category_pct:.1f}%)")
                    
                    print(f"\nEpoch {epoch}: WARNING - Components hitting clip limits: {', '.join(components_clipped)}")
                    print(f"  - This may indicate gradient saturation and limited learning for these components")
                    
                    if epoch >= self.clip_increase_start_epoch:
                        next_update = self.clip_increase_start_epoch + ((epochs_since_start // self.clip_increase_frequency) + 1) * self.clip_increase_frequency
                        print(f"  - Next clip limit increase scheduled for epoch {next_update}")
                        if next_update - epoch > 10:
                            print(f"  - Consider manually adjusting clip limits or decreasing update frequency")
                else:
                    # All components have headroom
                    print(f"\nEpoch {epoch}: Utility components within limits - "
                          f"Temporal: {temporal_pct:.1f}%, Category: {category_pct:.1f}%")
        
        # Track epochs since last clip limit analysis
        last_clip_analysis = -1
        analysis_frequency = 10  # Check every 10 epochs
        
        X_train = self.X_train
        
        # Training loop
        print(f"Starting training for {epochs} epochs from epoch {start_epoch}")
        for epoch in range(start_epoch, start_epoch + epochs):
            # Update clip limits based on current epoch
            clip_limits_changed = self.update_clip_limits(epoch)
            
            # Update component weights based on curriculum
            weights_changed = self.update_component_weights(epoch)
            
            # Special checks at critical points to ensure components aren't clip-limited
            if epoch in [75, 125, 175, 225]:
                # Check if we need to manually adjust limits
                if hasattr(self, 'last_epoch_metrics') and 'temporal_loss_orig' in self.last_epoch_metrics:
                    temporal_pct = self.last_epoch_metrics['temporal_loss_orig'] / self.current_clip_limits['temporal']
                    category_pct = self.last_epoch_metrics['category_loss_orig'] / self.current_clip_limits['category']
                    
                    changes_made = False
                    new_limits = {}
                    
                    # If temporal component is hitting >90% of clip limit, increase it
                    if temporal_pct > 0.9:
                        old_limit = self.current_clip_limits['temporal']
                        new_limit = min(old_limit * 1.3, self.max_clip_limits['temporal'])
                        if new_limit > old_limit:
                            new_limits['temporal'] = new_limit
                            changes_made = True
                    
                    # If category component is hitting >90% of clip limit, increase it
                    if category_pct > 0.9:
                        old_limit = self.current_clip_limits['category']
                        new_limit = min(old_limit * 1.3, self.max_clip_limits['category'])
                        if new_limit > old_limit:
                            new_limits['category'] = new_limit
                            changes_made = True
                    
                    # Apply changes if needed
                    if changes_made:
                        print(f"\nEpoch {epoch}: Special checkpoint - adjusting clip limits")
                        if 'temporal' in new_limits:
                            print(f"  - Increasing temporal limit from {self.current_clip_limits['temporal']:.2f} to {new_limits['temporal']:.2f}")
                            self.current_clip_limits['temporal'] = new_limits['temporal']
                        if 'category' in new_limits:
                            print(f"  - Increasing category limit from {self.current_clip_limits['category']:.2f} to {new_limits['category']:.2f}")
                            self.current_clip_limits['category'] = new_limits['category']
                            
                        # Reset statistics for affected components
                        if 'temporal' in new_limits:
                            self.utility_norm_stats['temporal_mean'] = 0.0
                            self.utility_norm_stats['temporal_std'] = 1.0
                        if 'category' in new_limits:
                            self.utility_norm_stats['category_mean'] = 0.0
                            self.utility_norm_stats['category_std'] = 1.0
                        self.utility_norm_stats['update_count'] = 0
                        
                        clip_limits_changed = True
            
            # Reset early stopping counters if training conditions changed significantly
            if clip_limits_changed or weights_changed:
                significant_change = False
                
                if clip_limits_changed:
                    print(f"Epoch {epoch}: Clip limits changed - resetting utility and balance early stopping counters")
                    significant_change = True
                    
                if weights_changed:
                    print(f"Epoch {epoch}: Component weights changed significantly - resetting utility early stopping counter")
                    significant_change = True
                
                if significant_change:
                    # Reset counters that would be affected by these changes
                    no_improvement_counts['balanced_score'] = 0
                    no_improvement_counts['utility_improvement'] = 0
                    # Also reset metrics history to prevent false trends
                    for key in ['spatial_loss_orig', 'temporal_loss_orig', 'category_loss_orig']:
                        metrics_history[key] = []
                    
                    if self.use_wandb:
                        wandb_metrics["early_stopping/counters_reset"] = True
                        wandb_metrics["early_stopping/reset_reason"] = "clip_limits" if clip_limits_changed else "curriculum"
            
            # Sample batch
            idx = np.random.randint(0, len(X_train[0]), batch_size)
            batch = [X[idx] for X in X_train]
            
            # Training step
            metrics = self.train_step(batch, batch_size)
            
            # Store metrics for next epoch's checks
            self.last_epoch_metrics = {
                'temporal_loss_orig': metrics.get('temporal_loss_orig', 0),
                'category_loss_orig': metrics.get('category_loss_orig', 0)
            }
            
            # Analyze clip limit impact
            if all(k in metrics for k in ['temporal_loss_orig', 'category_loss_orig']):
                # Check if components are hitting clip limits
                temporal_pct = metrics['temporal_loss_orig'] / self.current_clip_limits['temporal'] * 100
                category_pct = metrics['category_loss_orig'] / self.current_clip_limits['category'] * 100
                
                temporal_clipped = temporal_pct >= 99.0
                category_clipped = category_pct >= 99.0
                
                # Only log warnings every 10 epochs to avoid spam
                if epoch % 10 == 0 or temporal_clipped or category_clipped:
                    components_clipped = []
                    if temporal_clipped:
                        components_clipped.append(f"Temporal ({temporal_pct:.1f}%)")
                    if category_clipped:
                        components_clipped.append(f"Category ({category_pct:.1f}%)")
                    
                    if components_clipped:
                        print(f"\nEpoch {epoch}: WARNING - Components hitting clip limits: {', '.join(components_clipped)}")
                        print(f"  - This may indicate gradient saturation and limited learning for these components")
                        
                        if epoch >= self.clip_increase_start_epoch:
                            epochs_since_start = epoch - self.clip_increase_start_epoch
                            next_update = self.clip_increase_start_epoch + ((epochs_since_start // self.clip_increase_frequency) + 1) * self.clip_increase_frequency
                            print(f"  - Next clip limit increase scheduled for epoch {next_update}")
                            
                            # Manual intervention suggestion if next update is far away
                            if next_update - epoch > 10 and (temporal_clipped or category_clipped):
                                if temporal_clipped:
                                    suggested_limit = round(self.current_clip_limits['temporal'] * 1.2, 1)
                                    print(f"  - Consider manually increasing temporal clip limit to {suggested_limit}")
                                if category_clipped:
                                    suggested_limit = round(self.current_clip_limits['category'] * 1.2, 1)
                                    print(f"  - Consider manually increasing category clip limit to {suggested_limit}")
                    elif epoch % 10 == 0:
                        # Report healthy status every 10 epochs
                        print(f"\nEpoch {epoch}: Utility components within limits - "
                              f"Temporal: {temporal_pct:.1f}%, Category: {category_pct:.1f}%")
            
            # Initialize metrics dictionary for logging
            wandb_metrics = {
                "epoch": epoch,
                "d_loss_real": metrics['d_loss_real'],
                "d_loss_fake": metrics['d_loss_fake'],
                "g_loss": metrics['g_loss'],
                "c_loss": metrics['c_loss'],
                "gen_updates": metrics['gen_updates']
            }
            
            # Add clip limit metrics
            wandb_metrics["clip_limits/temporal"] = self.current_clip_limits['temporal']
            wandb_metrics["clip_limits/category"] = self.current_clip_limits['category']
            wandb_metrics["clip_limits/spatial"] = self.current_clip_limits['spatial']
            
            # Log curriculum learning weights
            wandb_metrics["curriculum/spatial_weight"] = self.current_component_weights['spatial']
            wandb_metrics["curriculum/temporal_weight"] = self.current_component_weights['temporal']
            wandb_metrics["curriculum/category_weight"] = self.current_component_weights['category']
            
            # Calculate curriculum progress
            if epoch >= self.curriculum_start_epoch and epoch < self.curriculum_start_epoch + self.curriculum_duration:
                progress = (epoch - self.curriculum_start_epoch) / self.curriculum_duration
                wandb_metrics["curriculum/progress"] = progress
            elif epoch >= self.curriculum_start_epoch + self.curriculum_duration:
                wandb_metrics["curriculum/progress"] = 1.0
            else:
                wandb_metrics["curriculum/progress"] = 0.0
            
            # Log reward component metrics
            reward_component_keys = [
                'r_adv_raw', 'r_util_raw', 'r_priv_raw',
                'r_adv_normalized', 'r_util_normalized', 
                'r_adv', 'r_util', 'r_priv',
                'rewards_mean', 'rewards_std', 'rewards_min', 'rewards_max'
            ]
            for key in reward_component_keys:
                if key in metrics:
                    wandb_metrics[f"reward_components/{key}"] = metrics[key]
            
            # Log utility component metrics - both original and normalized
            utility_component_keys = [
                'spatial_loss', 'temporal_loss', 'category_loss',
                'spatial_component', 'temporal_component', 'category_component',
                'spatial_loss_orig', 'temporal_loss_orig', 'category_loss_orig'
            ]
            for key in utility_component_keys:
                if key in metrics:
                    wandb_metrics[f"utility_components/{key}"] = metrics[key]
            
            # Log normalization statistics
            normalization_keys = [
                'spatial_mean', 'spatial_std', 
                'temporal_mean', 'temporal_std',
                'category_mean', 'category_std'
            ]
            for key in normalization_keys:
                if key in metrics:
                    wandb_metrics[f"normalization/{key}"] = metrics[key]
                    
            # Log reward effectiveness metrics
            if 'tul_accuracy' in metrics:
                wandb_metrics["reward_effectiveness/tul_accuracy"] = metrics['tul_accuracy']
                
            # Calculate reward component contributions
            if all(k in metrics for k in ['r_adv', 'r_util', 'r_priv']):
                total = abs(metrics['r_adv']) + abs(metrics['r_util']) + abs(metrics['r_priv'])
                if total > 0:
                    adv_contribution = abs(metrics['r_adv']) / total
                    util_contribution = abs(metrics['r_util']) / total
                    priv_contribution = abs(metrics['r_priv']) / total
                    
                    wandb_metrics["reward_effectiveness/adv_contribution"] = adv_contribution
                    wandb_metrics["reward_effectiveness/util_contribution"] = util_contribution
                    wandb_metrics["reward_effectiveness/priv_contribution"] = priv_contribution
            
            # Calculate training balance metrics
            d_avg_loss = (metrics['d_loss_real'] + metrics['d_loss_fake']) / 2
            g_to_d_ratio = metrics['g_loss'] / (d_avg_loss + 1e-8)
            d_real_to_fake_ratio = metrics['d_loss_real'] / (metrics['d_loss_fake'] + 1e-8)
            
            wandb_metrics["balance/g_to_d_loss_ratio"] = g_to_d_ratio
            wandb_metrics["balance/d_real_to_fake_ratio"] = d_real_to_fake_ratio
            
            # Calculate utility component balance
            if all(k in metrics for k in ['spatial_component', 'temporal_component', 'category_component']):
                util_total = (abs(metrics['spatial_component']) + 
                             abs(metrics['temporal_component']) + 
                             abs(metrics['category_component']) + 1e-8)
                
                spatial_contribution = abs(metrics['spatial_component']) / util_total
                temporal_contribution = abs(metrics['temporal_component']) / util_total
                category_contribution = abs(metrics['category_component']) / util_total
                
                wandb_metrics["balance/spatial_contribution"] = spatial_contribution
                wandb_metrics["balance/temporal_contribution"] = temporal_contribution
                wandb_metrics["balance/category_contribution"] = category_contribution
            
            # Log % of clip limit used
            if all(k in metrics for k in ['temporal_loss_orig', 'category_loss_orig']):
                temporal_pct_used = metrics['temporal_loss_orig'] / self.current_clip_limits['temporal']
                category_pct_used = metrics['category_loss_orig'] / self.current_clip_limits['category']
                spatial_pct_used = metrics['spatial_loss_orig'] / self.current_clip_limits['spatial']
                
                wandb_metrics["clip_limits/temporal_pct_used"] = temporal_pct_used
                wandb_metrics["clip_limits/category_pct_used"] = category_pct_used
                wandb_metrics["clip_limits/spatial_pct_used"] = spatial_pct_used
            
            # Update metrics history
            for key in metrics_history:
                if key == 'd_loss':
                    metrics_history[key].append(d_avg_loss)
                elif key in metrics:
                    metrics_history[key].append(metrics[key])
            
            # Keep only the most recent entries
            for key in metrics_history:
                metrics_history[key] = metrics_history[key][-history_window:]
            
            # ------------ Multi-metric early stopping logic ------------
            if early_stopping and epoch >= history_window:
                # 1. Traditional generator loss metric
                current_g_loss = metrics['g_loss']
                if current_g_loss < best_metrics['g_loss'] - min_delta:
                    best_metrics['g_loss'] = current_g_loss
                    best_epochs['g_loss'] = epoch
                    no_improvement_counts['g_loss'] = 0
                    # Save best model for this metric
                    self.save_checkpoint(epoch, best=True, suffix='g_loss')
                    if self.use_wandb:
                        wandb_metrics["early_stopping/best_g_loss"] = current_g_loss
                        wandb_metrics["early_stopping/best_g_loss_epoch"] = epoch
                else:
                    no_improvement_counts['g_loss'] += 1
                
                # 2. Balanced reward score (rewards should be balanced, not dominated by one component)
                # Calculate balance score: higher when components are more equal
                balance_score = 0
                if all(k in metrics for k in ['r_adv', 'r_util', 'r_priv']):
                    # Perfect balance would be 0.33, 0.33, 0.33
                    ideal_contribution = 1.0 / 3.0
                    balance_score = 1.0 - (
                        abs(adv_contribution - ideal_contribution) +
                        abs(util_contribution - ideal_contribution) +
                        abs(priv_contribution - ideal_contribution)
                    ) / 2.0  # Normalize to [0, 1]
                    
                    if self.use_wandb:
                        wandb_metrics["early_stopping/balance_score"] = balance_score
                    
                    if balance_score > best_metrics['balanced_score'] + min_delta:
                        best_metrics['balanced_score'] = balance_score
                        best_epochs['balanced_score'] = epoch
                        no_improvement_counts['balanced_score'] = 0
                        # Save best model for this metric
                        self.save_checkpoint(epoch, best=True, suffix='balance')
                        if self.use_wandb:
                            wandb_metrics["early_stopping/best_balance_score"] = balance_score
                            wandb_metrics["early_stopping/best_balance_epoch"] = epoch
                    else:
                        no_improvement_counts['balanced_score'] += 1
                
                # 3. Utility improvement score
                # Calculate trend in utility losses (looking for decreasing trend)
                utility_trend = 0
                if len(metrics_history['spatial_loss_orig']) >= history_window:
                    # Calculate average improvement over window
                    spatial_improvement = metrics_history['spatial_loss_orig'][0] - metrics_history['spatial_loss_orig'][-1]
                    temporal_improvement = 0
                    category_improvement = 0
                    
                    # Only consider temporal/category if they're not hitting clip limits consistently
                    # Use a more relaxed threshold (90% instead of 99%) to allow more learning
                    if metrics['temporal_loss_orig'] < self.current_clip_limits['temporal'] * 0.9:
                        temporal_improvement = metrics_history['temporal_loss_orig'][0] - metrics_history['temporal_loss_orig'][-1]
                    else:
                        # Still give partial credit even when hitting clip limits
                        # This helps the model continue learning even with clipped values
                        temporal_improvement = 0.2 * (metrics_history['temporal_loss_orig'][0] - metrics_history['temporal_loss_orig'][-1])
                    
                    if metrics['category_loss_orig'] < self.current_clip_limits['category'] * 0.9:
                        category_improvement = metrics_history['category_loss_orig'][0] - metrics_history['category_loss_orig'][-1]
                    else:
                        # Still give partial credit even when hitting clip limits
                        category_improvement = 0.2 * (metrics_history['category_loss_orig'][0] - metrics_history['category_loss_orig'][-1])
                    
                    # Weight improvements based on current curriculum weights, but ensure temporal and category
                    # are contributing even if they're smaller than spatial
                    utility_trend = (
                        self.current_component_weights['spatial'] * spatial_improvement +
                        max(self.current_component_weights['temporal'], 0.3) * temporal_improvement +
                        max(self.current_component_weights['category'], 0.3) * category_improvement
                    )
                    
                    if self.use_wandb:
                        wandb_metrics["early_stopping/utility_trend"] = utility_trend
                    
                    if utility_trend > best_metrics['utility_improvement'] + min_delta:
                        best_metrics['utility_improvement'] = utility_trend
                        best_epochs['utility_improvement'] = epoch
                        no_improvement_counts['utility_improvement'] = 0
                        # Save best model for this metric
                        self.save_checkpoint(epoch, best=True, suffix='utility')
                        if self.use_wandb:
                            wandb_metrics["early_stopping/best_utility_trend"] = utility_trend
                            wandb_metrics["early_stopping/best_utility_epoch"] = epoch
                    else:
                        no_improvement_counts['utility_improvement'] += 1
                
                # 4. Privacy score
                privacy_score = 0
                if 'tul_accuracy' in metrics:
                    # Lower TUL accuracy is better for privacy
                    privacy_score = 1.0 - metrics['tul_accuracy']
                    
                    if self.use_wandb:
                        wandb_metrics["early_stopping/privacy_score"] = privacy_score
                    
                    if privacy_score > best_metrics['privacy_score'] + min_delta:
                        best_metrics['privacy_score'] = privacy_score
                        best_epochs['privacy_score'] = epoch
                        no_improvement_counts['privacy_score'] = 0
                        # Save best model for this metric
                        self.save_checkpoint(epoch, best=True, suffix='privacy')
                        if self.use_wandb:
                            wandb_metrics["early_stopping/best_privacy_score"] = privacy_score
                            wandb_metrics["early_stopping/best_privacy_epoch"] = epoch
                    else:
                        no_improvement_counts['privacy_score'] += 1
                
                # Calculate a weighted combined score for overall model quality
                combined_score = 0
                if all(k in metrics for k in ['r_adv', 'r_util', 'r_priv']) and 'tul_accuracy' in metrics:
                    # Weight the different metrics based on importance
                    g_loss_norm = min(1.0, 300.0 / (metrics['g_loss'] + 1e-8))  # Normalize generator loss
                    
                    combined_score = (
                        0.3 * g_loss_norm +  # Generator loss
                        0.3 * balance_score +  # Reward balance
                        0.3 * min(1.0, utility_trend * 10) +  # Utility improvement trend
                        0.1 * privacy_score  # Privacy score
                    )
                    
                    if self.use_wandb:
                        wandb_metrics["early_stopping/combined_score"] = combined_score
                
                # Check if multiple criteria suggest stopping
                stop_count = sum(1 for count in no_improvement_counts.values() if count >= patience)
                if self.use_wandb:
                    wandb_metrics["early_stopping/criteria_suggesting_stop"] = stop_count
                
                # Don't trigger early stopping before clip limits have a chance to increase
                min_epochs_before_stopping = max(self.clip_increase_start_epoch + 20, 
                                               self.curriculum_start_epoch + 50)
                
                # If at least 3 criteria have not improved for patience epochs, stop training
                # But only after giving curriculum learning and clip limits a chance to take effect
                if stop_count >= 3 and epoch >= min_epochs_before_stopping:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    print(f"No improvement in {stop_count} out of {len(no_improvement_counts)} criteria for {patience} epochs")
                    
                    # Find the best overall model based on the most recent best epoch
                    best_overall_epoch = max(best_epochs.values())
                    best_metric = [k for k, v in best_epochs.items() if v == best_overall_epoch][0]
                    
                    print(f"Loading best overall model from epoch {best_overall_epoch} (best {best_metric})")
                    self.load_checkpoint(best_overall_epoch, suffix=best_metric)
                    
                    # Log early stopping in wandb
                    if self.use_wandb:
                        wandb.run.summary["stopped_early"] = True
                        wandb.run.summary["total_epochs"] = epoch + 1
                        wandb.run.summary["best_overall_epoch"] = best_overall_epoch
                        wandb.run.summary["best_overall_metric"] = best_metric
                        
                        for metric, best_epoch in best_epochs.items():
                            wandb.run.summary[f"best_{metric}_epoch"] = best_epoch
                    
                    break
                elif stop_count >= 3 and epoch < min_epochs_before_stopping:
                    # Log that we're delaying early stopping to allow curriculum and clip limits to take effect
                    print(f"\nEarly stopping criteria met ({stop_count}/4), but continuing until epoch {min_epochs_before_stopping}")
                    print(f"Allowing curriculum learning (starts at {self.curriculum_start_epoch}) and " +
                          f"clip limit increases (start at {self.clip_increase_start_epoch}) to take effect")
                    if self.use_wandb:
                        wandb_metrics["early_stopping/stopping_delayed"] = True
                        wandb_metrics["early_stopping/min_epochs_required"] = min_epochs_before_stopping
            
            # Log all metrics to wandb
            if self.use_wandb:
                wandb.log(wandb_metrics)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"D_real: {metrics['d_loss_real']:.4f}, D_fake: {metrics['d_loss_fake']:.4f}, G: {metrics['g_loss']:.4f}, C: {metrics['c_loss']:.4f}")
                print(f"Reward components - Adv: {metrics.get('r_adv', 0):.4f}, Util: {metrics.get('r_util', 0):.4f}, Priv: {metrics.get('r_priv', 0):.4f}")
                print(f"Original utility losses - Spatial: {metrics.get('spatial_loss_orig', 0):.4f}, Temporal: {metrics.get('temporal_loss_orig', 0):.4f}, Category: {metrics.get('category_loss_orig', 0):.4f}")
                print(f"Normalized utility losses - Spatial: {metrics.get('spatial_loss', 0):.4f}, Temporal: {metrics.get('temporal_loss', 0):.4f}, Category: {metrics.get('category_loss', 0):.4f}")
                print(f"Clip limits - Spatial: {self.current_clip_limits['spatial']:.2f}, Temporal: {self.current_clip_limits['temporal']:.2f}, Category: {self.current_clip_limits['category']:.2f}")
                print(f"Component weights - Spatial: {self.current_component_weights['spatial']:.3f}, Temporal: {self.current_component_weights['temporal']:.3f}, Category: {self.current_component_weights['category']:.3f}")
                if 'tul_accuracy' in metrics:
                    print(f"TUL Classifier Accuracy: {metrics['tul_accuracy']:.4f}")
                
                if early_stopping and epoch >= history_window:
                    print(f"Early stopping status - G Loss: {no_improvement_counts['g_loss']}/{patience}, " +
                          f"Balance: {no_improvement_counts.get('balanced_score', 0)}/{patience}, " +
                          f"Utility: {no_improvement_counts.get('utility_improvement', 0)}/{patience}, " +
                          f"Privacy: {no_improvement_counts.get('privacy_score', 0)}/{patience}")
            
            # Save checkpoints
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                
                # Save to wandb
                if self.use_wandb:
                    wandb.save(f'results/generator_{epoch}.weights.h5')
                    wandb.save(f'results/discriminator_{epoch}.weights.h5')
                    wandb.save(f'results/critic_{epoch}.weights.h5')
        
        # Close wandb run
        if self.use_wandb:
            wandb.finish()
        
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
            
    def load_checkpoint(self, epoch, suffix=''):
        """Load a specific checkpoint.
        
        Args:
            epoch: The epoch number to load
            suffix: Optional suffix in the filename (for multi-metric early stopping)
        """
        try:
            # Build the prefix for the filename
            if suffix:
                prefix = f"best_{suffix}_"
            else:
                prefix = ""
                
            self.generator.load_weights(f'results/{prefix}generator_{epoch}.weights.h5')
            self.discriminator.load_weights(f'results/{prefix}discriminator_{epoch}.weights.h5')
            self.critic.load_weights(f'results/{prefix}critic_{epoch}.weights.h5')
            print(f"Loaded model weights from epoch {epoch}" + 
                  (f" (best {suffix} model)" if suffix else ""))
        except Exception as e:
            print(f"Error loading model from epoch {epoch}: {e}")
            raise

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
            
            # Analyze advantage distribution
            adv_mean = np.mean(advantages_np)
            adv_std = np.std(advantages_np) + 1e-8  # Avoid division by zero
            adv_min = np.min(advantages_np)
            adv_max = np.max(advantages_np)
            
            # Apply normalization to stabilize training
            # Center and scale advantages for more stable policy updates
            advantages_normalized = (advantages_np - adv_mean) / adv_std
            
            # Convert to positive weights for sample_weight (must be positive)
            # Use softmax-like normalization to maintain relative ordering
            # but ensure all values are positive and sum to batch_size
            advantages_exp = np.exp(np.clip(advantages_normalized, -5.0, 5.0))  # exp(clipped) for numerical stability
            advantages_weight = advantages_exp / (np.mean(advantages_exp) + 1e-8) # Normalize to reasonable values
            
            # Apply minimum weight threshold to ensure all samples contribute
            min_weight = 0.1  # Minimum sample weight
            advantages_weight = np.maximum(advantages_weight, min_weight)
            
            # Apply non-linear weighting to emphasize high-advantage samples
            # This gives more weight to samples with higher advantages
            # Square function gives more weight to positive outliers
            advantages_weight = np.sign(advantages_weight) * np.square(np.abs(advantages_weight))
            
            # Final normalization to keep mean close to 1.0
            advantages_weight = advantages_weight / (np.mean(advantages_weight) + 1e-8)
            
            # Ensure it has the right shape
            if len(advantages_weight) != batch_size:
                print(f"Warning: Reshaping advantages from {len(advantages_weight)} to {batch_size}")
                # If shapes don't match, use uniform weights
                advantages_weight = np.ones(batch_size)
                
            # Final safety check - replace any NaN or inf values
            advantages_weight = np.nan_to_num(advantages_weight, nan=1.0, posinf=2.0, neginf=0.1)
                
        except Exception as e:
            print(f"Error processing advantages: {e}")
            # Fallback to uniform weights
            advantages_weight = np.ones(batch_size)
            
        # Print statistics about advantage values used for training
        print(f"Advantage stats - min: {adv_min:.4f}, max: {adv_max:.4f}, " +
              f"mean: {adv_mean:.4f}, std: {adv_std:.4f}")
        print(f"Weight stats - min: {np.min(advantages_weight):.4f}, max: {np.max(advantages_weight):.4f}, " +
              f"mean: {np.mean(advantages_weight):.4f}, std: {np.std(advantages_weight):.4f}")
        
        # Apply gradient clipping to combined model for more stable updates
        # This is done by setting clipnorm on the optimizer
        # We'll set this dynamically based on the advantage statistics
        clipnorm = 0.5 + 0.1 * np.log(1.0 + adv_std)  # Adaptive clip norm based on advantage spread
        self.actor_optimizer.clipnorm = clipnorm
        
        # Train the combined model with sample weights from advantages
        loss = self.combined.train_on_batch(
            all_inputs, 
            targets,
            sample_weight=advantages_weight
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
            marc_model.load_weights('MARC/weights/MARC_Weight.h5')
            
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