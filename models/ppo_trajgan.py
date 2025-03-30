import tensorflow as tf
import numpy as np
import os
import time
import json
import wandb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Embedding, TimeDistributed, 
    Lambda, Layer, GlobalAveragePooling1D, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import tensorflow_probability as tfp

# Import custom components
from models.transformer_components import (
    TransformerEncoderLayer, TransformerDecoderLayer, 
    PositionalEncoding, create_padding_mask, create_look_ahead_mask, 
    PaddingMaskLayer, LookAheadMaskLayer, CombinedMaskLayer
)
from models.critic import CriticNetwork
from models.reward_function import TrajRewardFunction
from losses import d_bce_loss, trajLoss, compute_advantage

class TileLayer(tf.keras.layers.Layer):
    def __init__(self, multiples, **kwargs):
        super(TileLayer, self).__init__(**kwargs)
        self.multiples = multiples
    
    def call(self, inputs):
        # For noise_tiled we need dynamic multiples based on max_length
        if len(self.multiples) == 3 and self.multiples[1] == 0:
            # Special case for tiling noise across sequence length
            batch_size = tf.shape(inputs)[0]
            sequence_length = self.multiples[2]  # This will be replaced at runtime
            return tf.tile(inputs, [1, sequence_length, 1])
        else:
            return tf.tile(inputs, self.multiples)
    
    def get_config(self):
        config = super().get_config()
        config.update({"multiples": self.multiples})
        return config

class TransformerTrajGAN:
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor):
        """
        Initialize the Transformer-based TrajGAN model.
        
        Args:
            latent_dim: Dimension of the latent space
            keys: List of trajectory components (lat_lon, day, hour, category, mask)
            vocab_size: Dictionary with vocab size for each component
            max_length: Maximum sequence length
            lat_centroid: Latitude centroid for normalization
            lon_centroid: Longitude centroid for normalization
            scale_factor: Scale factor for normalization
        """
        # Store model parameters
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.keys = keys
        self.vocab_size = vocab_size
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        
        # Model hyperparameters
        self.d_model = 128
        self.num_heads = 4
        self.dff = 512
        self.dropout_rate = 0.1
        
        # Optimizer parameters
        self.gen_lr = 0.0001
        self.disc_lr = 0.0001
        
        # Skip serialization flag (helps with keras model saving/loading)
        self._skip_serialization = True
        
        print("Building generator...")
        self.generator = self.build_generator()
        
        print("Building discriminator...")
        self.discriminator = self.build_discriminator()
        
        # Initialize weights for stable training
        for layer in self.generator.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = tf.keras.initializers.he_normal(seed=42)
        
        # Compile the discriminator
        self.discriminator.compile(
            loss=d_bce_loss(None),
            optimizer=Adam(self.disc_lr),
            metrics=['accuracy']
        )
        
        print("Building combined GAN model...")
        self.build_gan_model()
        
        # Create params directory if it doesn't exist
        if not os.path.exists("params"):
            os.makedirs("params")
        
        # Save model structures as text summaries instead of JSON
        print("Saving model summaries...")
        try:
            with open("params/transformer_generator_summary.txt", "w") as f:
                self.generator.summary(print_fn=lambda x: f.write(x + '\n'))
                
            with open("params/transformer_discriminator_summary.txt", "w") as f:
                self.discriminator.summary(print_fn=lambda x: f.write(x + '\n'))
        except Exception as e:
            print(f"Warning: Could not save model summaries: {e}")
            print("Continuing without saving summaries.")
    
    def build_generator(self):
        """
        Build the Transformer-based trajectory generator.
        
        Returns:
            model: Keras model for the generator
        """
        # Print actual vocab sizes for debugging
        print(f"Vocabulary sizes in build_generator:")
        for key, size in self.vocab_size.items():
            print(f"  {key}: {size}")
            
        # Input layers
        inputs = []
        for idx, key in enumerate(self.keys):
            if key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name=f'input_{key}')
            elif key == 'mask':
                i = Input(shape=(self.max_length, 1), name=f'input_{key}')
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name=f'input_{key}')
            inputs.append(i)
        
        # Add noise input
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        inputs.append(noise)
        
        # Encoder preprocessing
        # Process inputs to a common dimension space
        processed_inputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                x = TimeDistributed(Dense(64, activation='relu'))(inputs[idx])
            else:
                x = TimeDistributed(Dense(64, activation='relu'))(inputs[idx])
            processed_inputs.append(x)
            
        # Concatenate processed inputs
        concat_inputs = Concatenate(axis=2)(processed_inputs)
        
        # Project to d_model dimension
        encoder_inputs = Dense(self.d_model, activation='relu')(concat_inputs)
        
        # Add positional encoding
        encoder_inputs = PositionalEncoding(self.max_length, self.d_model)(encoder_inputs)
        
        # Create encoder padding mask
        mask_sum = Lambda(lambda x: tf.reduce_sum(x, axis=-1))(inputs[4])  # mask input
        padding_mask = PaddingMaskLayer()(mask_sum)
        
        # Transformer encoder layers
        enc_output = encoder_inputs
        for i in range(3):  # 3 encoder layers
            enc_output = TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate
            )(enc_output, training=True, mask=padding_mask)
        
        # Add noise to encoder output for each position in the sequence
        noise_expanded = Lambda(lambda x: tf.expand_dims(x, 1))(noise)
        noise_tiled = TileLayer(multiples=[1, self.max_length, 1])(noise_expanded)
        enc_output_with_noise = Concatenate(axis=2)([enc_output, noise_tiled])
        enc_output_with_noise = Dense(self.d_model, activation='relu')(enc_output_with_noise)
        
        # Decoder inputs (shifted right)
        # We will use the same encoder output with different heads for generation
        
        # Output heads for each trajectory component
        outputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                # Just pass through the mask
                outputs.append(inputs[idx])
            elif key == 'lat_lon':
                # Continuous output for lat/lon
                latlon_logits = TimeDistributed(Dense(2, activation='tanh'))(enc_output_with_noise)
                # Scale the output
                scaled_latlon = Lambda(lambda x: x * self.scale_factor)(latlon_logits)
                outputs.append(scaled_latlon)
            else:
                # Categorical output
                logits = TimeDistributed(Dense(self.vocab_size[key]))(enc_output_with_noise)
                prob_output = TimeDistributed(tf.keras.layers.Softmax())(logits)
                outputs.append(prob_output)
                
        # Create model
        generator = Model(inputs=inputs, outputs=outputs, name='generator')
        
        # Print model summary for debugging
        print("Generator model summary:")
        generator.summary(print_fn=lambda x: print(f"  {x}"))
        
        return generator
    
    def build_discriminator(self):
        """
        Build the Transformer-based trajectory discriminator.
        
        Returns:
            model: Keras model for the discriminator
        """
        # Print actual vocab sizes for debugging
        print(f"Vocabulary sizes in build_discriminator:")
        for key, size in self.vocab_size.items():
            print(f"  {key}: {size}")
            
        # Input layers (without mask)
        inputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name=f'input_{key}')
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name=f'input_{key}')
            inputs.append(i)
        
        # Process inputs to a common dimension space
        processed_inputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            elif key == 'lat_lon':
                x = TimeDistributed(Dense(64, activation='relu'))(inputs[idx])
            else:
                x = TimeDistributed(Dense(64, activation='relu'))(inputs[idx])
            processed_inputs.append(x)
            
        # Concatenate processed inputs
        concat_inputs = Concatenate(axis=2)(processed_inputs)
        
        # Project to d_model dimension
        encoder_inputs = Dense(self.d_model, activation='relu')(concat_inputs)
        
        # Add positional encoding
        encoder_inputs = PositionalEncoding(self.max_length, self.d_model)(encoder_inputs)
        
        # Transformer encoder layers
        x = encoder_inputs
        for i in range(3):  # 3 encoder layers
            x = TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate
            )(x, training=True)
        
        # Global pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer
        validity = Dense(1, activation='sigmoid')(x)
        
        # Create model
        discriminator = Model(inputs=inputs, outputs=validity, name='discriminator')
        
        # Print model summary for debugging
        print("Discriminator model summary:")
        discriminator.summary(print_fn=lambda x: print(f"  {x}"))
        
        return discriminator
    
    def build_gan_model(self):
        """Build the combined GAN model (generator + discriminator)"""
        # Generator inputs
        inputs = []
        for idx, key in enumerate(self.keys):
            if key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name=f'input_{key}')
            elif key == 'mask':
                i = Input(shape=(self.max_length, 1), name=f'input_{key}')
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name=f'input_{key}')
            inputs.append(i)
        
        # Add noise input
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        inputs.append(noise)
        
        # Generate trajectories
        gen_trajs = self.generator(inputs)
        
        # Freeze discriminator for generator training
        self.discriminator.trainable = False
        
        # Discriminator takes generated trajectories
        validity = self.discriminator(gen_trajs[:4])  # Exclude mask
        
        # Define a custom loss layer to handle Keras tensors properly
        class TrajLossWrapper(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super(TrajLossWrapper, self).__init__(**kwargs)
            
            def call(self, inputs):
                # This layer doesn't modify the input during forward pass
                return inputs
            
            def get_config(self):
                config = super().get_config()
                return config
        
        # Apply custom loss wrapper
        validity_with_loss = TrajLossWrapper()(validity)
        
        # Combined model
        self.combined = Model(inputs=inputs, outputs=validity_with_loss, name='combined_model')
        
        # Custom loss that considers both adversarial loss and trajectory properties
        # Use a simple binary crossentropy loss for compilation
        # The actual custom loss logic will be implemented in the train method
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=Adam(self.gen_lr)
        )
        
        # Save combined model summary
        try:
            if not os.path.exists("params"):
                os.makedirs("params")
                
            with open("params/combined_model_summary.txt", "w") as f:
                self.combined.summary(print_fn=lambda x: f.write(x + '\n'))
        except Exception as e:
            print(f"Warning: Could not save combined model summary: {e}")
            print("Continuing without saving summary.")

    # Add get_config method for serialization
    def get_config(self):
        """Return configuration for serialization."""
        config = {
            "latent_dim": self.latent_dim,
            "max_length": self.max_length,
            "keys": self.keys,
            "vocab_size": self.vocab_size,
            "lat_centroid": float(self.lat_centroid),
            "lon_centroid": float(self.lon_centroid),
            "scale_factor": float(self.scale_factor),
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "gen_lr": float(self.gen_lr),
            "disc_lr": float(self.disc_lr)
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        # Extract required parameters only
        init_config = {
            "latent_dim": config["latent_dim"],
            "keys": config["keys"],
            "vocab_size": config["vocab_size"],
            "max_length": config["max_length"],
            "lat_centroid": config["lat_centroid"],
            "lon_centroid": config["lon_centroid"],
            "scale_factor": config["scale_factor"]
        }
        return cls(**init_config)

class RL_Enhanced_Transformer_TrajGAN(TransformerTrajGAN):
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor):
        """
        Initialize the RL-enhanced Transformer TrajGAN model with PPO.
        
        Args:
            Same as parent class TransformerTrajGAN
        """
        print("Initializing PPO-Enhanced Transformer TrajGAN")
        
        # Initialize parent GAN model
        super().__init__(latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor)
        
        # Get category size for critic network
        category_size = vocab_size.get('category', 100)
        
        # Initialize critic network for PPO
        print("Building critic network...")
        self.critic = CriticNetwork(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            max_length=self.max_length,
            category_size=category_size
        )
        
        # Initialize reward function
        print("Initializing reward function...")
        self.reward_function = TrajRewardFunction()
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.critic_discount = 0.5
        self.entropy_beta = 0.02  # Increased from 0.01 to encourage exploration
        self.gamma = 0.99  # Discount factor
        
        # PPO optimization parameters
        self.ppo_epochs = 4  # Number of PPO updates per batch (increased from implicit 1)
        self.policy_optimizer = Adam(learning_rate=0.0001)
        
        # For logging training stats
        self.training_stats = {
            'g_losses': [],
            'd_losses': [],
            'value_losses': [],
            'policy_losses': [],
            'rewards': [],
            'utility_rewards': [],  # Track utility rewards separately
            'epochs': []
        }
        
        # Checkpoint management
        self.best_reward = -float('inf')
        self.best_epoch = 0
        self.best_utility = 0.0
        
        # Initialize TUL classifier
        self.tul_classifier = None
        
        # Create checkpoint directory
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
            
        print("RL-Enhanced Transformer TrajGAN initialized successfully")
    
    def set_tul_classifier(self, tul_classifier):
        """Set the TUL classifier for privacy reward computation"""
        self.tul_classifier = tul_classifier
        self.reward_function.tul_classifier = tul_classifier
    
    def compute_ppo_loss(self, states, actions, old_log_probs, advantages, values, returns):
        """
        Compute PPO loss for policy optimization.
        
        Args:
            states: Trajectory states
            actions: Taken actions
            old_log_probs: Log probabilities of actions under old policy
            advantages: Advantage estimates
            values: Value estimates
            returns: Discounted returns
            
        Returns:
            policy_loss: Loss for policy update
            value_loss: Loss for value function update
            entropy: Entropy of the policy
        """
        # Ensure all inputs are TensorFlow tensors
        states_tf = [tf.convert_to_tensor(s, dtype=tf.float32) for s in states]
        actions_tf = [tf.convert_to_tensor(a, dtype=tf.float32) for a in actions]
        old_log_probs_tf = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        advantages_tf = tf.convert_to_tensor(advantages, dtype=tf.float32)
        values_tf = tf.convert_to_tensor(values, dtype=tf.float32)
        returns_tf = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Generate trajectory with current policy
        noise = tf.random.normal((tf.shape(states_tf[0])[0], self.latent_dim))
        states_with_noise = states_tf + [noise]
        current_outputs = self.generator(states_with_noise)
        
        # Extract action components
        gen_latlon = current_outputs[0]
        gen_category = current_outputs[1]
        gen_day = current_outputs[2]
        gen_hour = current_outputs[3]
        
        # Extract targets
        target_latlon = actions_tf[0]
        target_category = actions_tf[1]
        target_day = actions_tf[2]
        target_hour = actions_tf[3]
        
        # Process using Keras operations instead of direct TensorFlow ops
        
        # For lat/lon (assuming normal distribution)
        latlon_log_probs = tf.reduce_sum(
            tfp.distributions.Normal(gen_latlon, 0.1).log_prob(target_latlon),
            axis=-1
        )
        
        # For categorical outputs - ensure we avoid log(0)
        gen_category_safe = tf.clip_by_value(gen_category, 1e-8, 1.0)
        gen_day_safe = tf.clip_by_value(gen_day, 1e-8, 1.0)
        gen_hour_safe = tf.clip_by_value(gen_hour, 1e-8, 1.0)
        
        category_log_probs = tf.reduce_sum(
            target_category * tf.math.log(gen_category_safe),
            axis=-1
        )
        
        day_log_probs = tf.reduce_sum(
            target_day * tf.math.log(gen_day_safe),
            axis=-1
        )
        
        hour_log_probs = tf.reduce_sum(
            target_hour * tf.math.log(gen_hour_safe),
            axis=-1
        )
        
        # Combine log probs with mask
        mask = states_tf[4][:,:,0]  # Extract mask
        
        masked_latlon_logprobs = tf.reduce_sum(latlon_log_probs * mask, axis=1)
        masked_category_logprobs = tf.reduce_sum(category_log_probs * mask, axis=1)
        masked_day_logprobs = tf.reduce_sum(day_log_probs * mask, axis=1)
        masked_hour_logprobs = tf.reduce_sum(hour_log_probs * mask, axis=1)
        
        # Sum up all components
        combined_log_probs = (
            masked_latlon_logprobs + 
            masked_category_logprobs + 
            masked_day_logprobs + 
            masked_hour_logprobs
        )
        
        # Calculate ratio for clipped objective
        ratio = tf.exp(combined_log_probs - old_log_probs_tf)
        
        # Clipped objective
        clipped_ratio = tf.clip_by_value(
            ratio, 
            1 - self.clip_epsilon, 
            1 + self.clip_epsilon
        )
        surrogate1 = ratio * advantages_tf
        surrogate2 = clipped_ratio * advantages_tf
        policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        
        # Value loss (MSE)
        # Use the critic model on converted tensor inputs
        # Create proper time input for critic with shape [batch_size, max_length, 2]
        
        # Extract day and hour features
        day_tensor = states_tf[1]  # One-hot day features
        hour_tensor = states_tf[2]  # One-hot hour features
        
        # Convert one-hot to indices and normalize to [0,1] range
        batch_size = tf.shape(day_tensor)[0]
        max_length = tf.shape(day_tensor)[1]
        
        # Manual conversion from one-hot to indices without using argmax directly
        # Create time features with shape [batch, seq_len, 2]
        time_features = tf.zeros((batch_size, max_length, 2), dtype=tf.float32)
        
        # Use another approach to avoid tf.argmax output_type issue
        # First get indices as integers using tf.int64 output_type
        day_indices = tf.argmax(day_tensor, axis=2, output_type=tf.int64)
        hour_indices = tf.argmax(hour_tensor, axis=2, output_type=tf.int64)
        
        # Convert to float manually
        day_indices_float = tf.cast(day_indices, tf.float32)
        hour_indices_float = tf.cast(hour_indices, tf.float32)
        
        # Normalize
        day_normalized = day_indices_float / 7.0  # 7 days in a week
        hour_normalized = hour_indices_float / 24.0  # 24 hours in a day
        
        # Reshape to [batch, seq_len, 1] for each feature
        day_reshaped = tf.expand_dims(day_normalized, axis=2)  # Shape: [batch, seq_len, 1]
        hour_reshaped = tf.expand_dims(hour_normalized, axis=2)  # Shape: [batch, seq_len, 1]
        
        # Concatenate to get final time features with shape [batch, seq_len, 2]
        time_features = tf.concat([day_reshaped, hour_reshaped], axis=2)
        
        value_pred = self.critic.model([
            states_tf[0],    # lat_lon
            states_tf[3],    # category
            time_features,   # time (day, hour indices)
            states_tf[4]     # mask
        ])
        value_loss = tf.reduce_mean(tf.square(returns_tf - value_pred))
        
        # Entropy bonus for exploration
        entropy = -tf.reduce_mean(
            tf.reduce_sum(gen_category_safe * tf.math.log(gen_category_safe), axis=-1)
        )
        
        return policy_loss, value_loss, entropy
    
    def optimize_policy(self, states, actions, old_log_probs, advantages, values, returns):
        """
        Optimize the policy using the PPO algorithm.
        
        Args:
            states: List of trajectory state components
            actions: List of trajectory action components
            old_log_probs: Log probs of actions under old policy
            advantages: Advantage estimates
            values: Value estimates
            returns: Discounted returns
        """
        policy_losses = []
        value_losses = []
        entropy_values = []
        
        # Run multiple optimization epochs for each batch (PPO update steps)
        for _ in range(self.ppo_epochs):
            # Compute the loss using our updated compute_ppo_loss method
            with tf.GradientTape() as tape:
                policy_loss, value_loss, entropy = self.compute_ppo_loss(
                    states, actions, old_log_probs, advantages, values, returns
                )
                
                # Total loss with entropy bonus and value function loss
                total_loss = (
                    policy_loss + 
                    self.critic_discount * value_loss - 
                    self.entropy_beta * entropy
                )
            
            # Get trainable variables
            trainable_vars = self.generator.trainable_variables
            
            # Calculate and apply gradients
            gradients = tape.gradient(total_loss, trainable_vars)
            
            # Clip gradients to prevent exploding gradients
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            
            # Apply gradients
            self.policy_optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Store losses for tracking
            policy_losses.append(float(policy_loss.numpy()))
            value_losses.append(float(value_loss.numpy()))
            entropy_values.append(float(entropy.numpy()))
        
        # Return average losses over PPO epochs
        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropy_values)
    
    def train(self, epochs=200, batch_size=32, sample_interval=10, early_stopping=True, patience=10, min_delta=0.001, use_wandb=False):
        """
        Train the RL-enhanced model using PPO.
        
        Args:
            epochs: Number of training epochs
            batch_size: Size of training batches
            sample_interval: Interval for saving checkpoints
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            min_delta: Minimum delta for improvement in early stopping
            use_wandb: Whether to use Weights & Biases for logging
            
        Returns:
            stats: Dictionary of training statistics
        """
        # Initialize WandB logging if requested
        if use_wandb:
            try:
                # Initialize wandb with the specified username
                wandb.init(
                    project="privacy-traj-gan", 
                    name="rl-enhanced-trajgan",
                    entity="xutao-henry-mao-vanderbilt-university"
                )
                print("Weights & Biases logging initialized")
            except Exception as e:
                print(f"Warning: Could not initialize Weights & Biases: {e}")
                print("Training will continue without W&B logging")
                use_wandb = False
        else:
            print("Weights & Biases logging disabled")
            
        # Training data
        x_train = np.load('data/final_train.npy', allow_pickle=True)
        
        # Padding to reach the maxlength
        X_train = [tf.keras.preprocessing.sequence.pad_sequences(
            f, self.max_length, padding='pre', dtype='float64'
        ) for f in x_train]
        
        # Training loop
        best_reward = -float('inf')
        best_epoch = 0
        best_utility = 0.0
        no_improvement_count = 0
        
        # Determine the number of users for TUL model
        num_users = 193  # Default value
        if self.tul_classifier is not None and hasattr(self.tul_classifier, 'num_users'):
            num_users = self.tul_classifier.num_users
            print(f"TUL classifier has {num_users} users configured")
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # ========================
            # 1. Train Discriminator
            # ========================
            
            # Select a random batch of trajectories
            idx = np.random.randint(0, X_train[0].shape[0], batch_size)
            
            # Real trajectories
            real_trajs = []
            real_trajs.append(X_train[0][idx])  # latlon
            real_trajs.append(X_train[1][idx])  # day
            real_trajs.append(X_train[2][idx])  # hour
            real_trajs.append(X_train[3][idx])  # category
            real_trajs.append(X_train[4][idx])  # mask
            
            # Print shapes for debugging - only in first epoch
            if epoch == 1:
                print("Input shapes for generator:")
                for i, key in enumerate(self.keys):
                    print(f"  {key}: {real_trajs[i].shape}, expected: (batch, seq_len, {self.vocab_size[key]})")
            
            # Random noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs.append(noise)
            
            # Generate synthetic trajectories
            try:
                gen_trajs = self.generator.predict(real_trajs)
                
                # Print shapes of generated outputs - only in first epoch
                if epoch == 1:
                    print("Generator output shapes:")
                    for i, key in enumerate(self.keys):
                        print(f"  {key}: {gen_trajs[i].shape}")
                
            except Exception as e:
                print(f"Error in generator.predict: {e}")
                print("Input shapes:")
                for i, comp in enumerate(real_trajs):
                    key = self.keys[i] if i < len(self.keys) else "noise"
                    print(f"  {key}: {comp.shape}")
                raise
            
            # Labels for real and fake trajectories
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            
            # Train discriminator
            d_loss_real = self.discriminator.train_on_batch(real_trajs[:4], real_labels)
            d_loss_fake = self.discriminator.train_on_batch(gen_trajs[:4], fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ========================
            # 2. Train Generator with GAN loss
            # ========================
            
            # New noise for generator training
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs[5] = noise
            
            # Manual GAN update with custom loss computation
            # First, create TensorFlow tensors from numpy arrays
            real_trajs_tf = [tf.convert_to_tensor(comp, dtype=tf.float32) for comp in real_trajs]
            real_labels_tf = tf.convert_to_tensor(real_labels, dtype=tf.float32)
            
            # Use gradient tape to track operations for gradient computation
            with tf.GradientTape() as tape:
                # Generate trajectories
                gen_trajs_tf = self.generator(real_trajs_tf)
                
                # Discriminator prediction
                validity = self.discriminator(gen_trajs_tf[:4])
                
                # Basic adversarial loss (binary cross-entropy)
                g_loss_bce = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(real_labels_tf, validity)
                )
                
                # Compute trajectory-specific losses using Keras ops instead of TensorFlow ops
                # This can be simplified for now to just use BCE loss
                g_loss = g_loss_bce
            
            # Get generator trainable variables
            trainable_vars = self.generator.trainable_variables
            
            # Compute and apply gradients
            gradients = tape.gradient(g_loss, trainable_vars)
            self.combined.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Convert loss to numpy for logging
            g_loss = float(g_loss.numpy())
            
            # ========================
            # 3. PPO-based Policy Update
            # ========================
            
            # Generate trajectories with current policy for PPO
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs[5] = noise
            gen_trajs_for_ppo = self.generator.predict(real_trajs)
            
            # Get discriminator scores for reward calculation
            disc_scores = self.discriminator.predict(gen_trajs_for_ppo[:4])
            
            # Compute rewards using reward function
            # Create user IDs within the valid range (0 to num_users-1)
            user_ids = np.arange(batch_size) % num_users
            rewards = np.zeros((batch_size, 1))
            utility_rewards = np.zeros((batch_size, 1))
            
            # Print debug info in first epoch
            if epoch == 1:
                print(f"Input shapes: {[comp.shape[:2] for comp in gen_trajs_for_ppo[:4]]}")
            
            for i in range(batch_size):
                # Extract individual trajectory components
                traj_components = [comp[i:i+1] for comp in gen_trajs_for_ppo]
                orig_components = [comp[i:i+1] for comp in real_trajs[:5]]  # Exclude noise
                
                # Compute reward
                rewards[i, 0] = self.reward_function.compute_reward(
                    traj_components, 
                    orig_components, 
                    user_ids[i], 
                    disc_scores[i]
                )
                
                # Also compute utility reward separately for tracking
                utility_rewards[i, 0] = self.reward_function.compute_utility_reward(
                    traj_components,
                    orig_components
                )
            
            # Compute advantages and returns for PPO
            # Create proper inputs for critic network with correct shapes
            # Critic expects [lat_lon, category, time (day+hour combined), mask]
            day = gen_trajs_for_ppo[1]
            hour = gen_trajs_for_ppo[2]
            
            # Concatenate day and hour to create time input for critic
            # The critic expects time to have shape [batch, seq_len, 2]
            # Create time features that match what the critic network expects
            time_features = np.zeros((batch_size, self.max_length, 2))
            
            # Extract the maximum value index for day and hour (one-hot to index)
            for b in range(batch_size):
                for t in range(self.max_length):
                    # Handle empty or padded positions
                    if np.sum(day[b, t]) > 0:
                        day_idx = np.argmax(day[b, t])
                    else:
                        day_idx = 0
                        
                    if np.sum(hour[b, t]) > 0:
                        hour_idx = np.argmax(hour[b, t])
                    else:
                        hour_idx = 0
                        
                    # Normalize to [0,1] range
                    time_features[b, t, 0] = day_idx / 7.0  # Normalize to [0,1]
                    time_features[b, t, 1] = hour_idx / 24.0  # Normalize to [0,1]
            
            critic_inputs = [
                gen_trajs_for_ppo[0],            # lat_lon
                gen_trajs_for_ppo[3],            # category
                time_features,                   # time (normalized day, hour indices)
                real_trajs[4]                    # mask
            ]
            
            # Get values from critic
            values = self.critic.predict_value(critic_inputs)
            
            # Compute advantages (simplified for single-step trajectories)
            advantages = rewards - values
            returns = rewards
            
            # Add utility-focused advantage component
            # This encourages the model to optimize specifically for utility
            utility_advantages = utility_rewards - np.mean(utility_rewards)
            combined_advantages = advantages + 0.5 * utility_advantages  # Weighted combination
            
            # Compute log probabilities of actions under old policy
            # (Simplified, assuming we're using the actions we just generated)
            # In a real implementation, you'd track the actual log probs
            old_log_probs = np.zeros((batch_size,))  # Placeholder
            
            # Perform PPO update with multiple optimization steps
            policy_loss, value_loss, entropy = self.optimize_policy(
                states=real_trajs[:5],  # Exclude noise
                actions=gen_trajs_for_ppo[:4],  # Exclude mask 
                old_log_probs=old_log_probs,
                advantages=combined_advantages,  # Use combined advantages
                values=values,
                returns=returns
            )
            
            # Update critic network with properly shaped inputs
            critic_loss = self.critic.train_on_batch(critic_inputs, returns)
            
            # ========================
            # 4. Logging and Checkpoints
            # ========================
            
            # Calculate average rewards
            avg_reward = np.mean(rewards)
            avg_utility = np.mean(utility_rewards)
            
            # Log metrics for this epoch
            epoch_time = time.time() - start_time
            print(f"[Epoch {epoch}/{epochs}] "
                  f"D loss: {d_loss[0]:.4f}, acc: {d_loss[1]:.4f} | "
                  f"G loss: {g_loss:.4f} | "
                  f"Policy loss: {policy_loss:.4f} | "
                  f"Value loss: {value_loss:.4f} | "
                  f"Avg reward: {avg_reward:.4f} | "
                  f"Avg utility: {avg_utility:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Log to WandB if enabled
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "d_loss": d_loss[0],
                    "d_accuracy": d_loss[1],
                    "g_loss": g_loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "avg_reward": avg_reward,
                    "avg_utility": avg_utility,
                    "critic_loss": critic_loss,
                    "epoch_time": epoch_time
                })
            
            # Save for monitoring progress
            self.training_stats['g_losses'].append(g_loss)
            self.training_stats['d_losses'].append(d_loss[0])
            self.training_stats['value_losses'].append(value_loss)
            self.training_stats['policy_losses'].append(policy_loss)
            self.training_stats['rewards'].append(avg_reward)
            self.training_stats['utility_rewards'].append(avg_utility)
            self.training_stats['epochs'].append(epoch)
            
            # Check if this is the best model based on combined metrics
            # We prioritize utility but still consider overall reward
            combined_score = avg_utility * 0.7 + avg_reward * 0.3
            
            if combined_score > best_reward:
                best_reward = combined_score
                best_epoch = epoch
                best_utility = avg_utility
                no_improvement_count = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best model with combined score {combined_score:.4f} (utility: {avg_utility:.4f})")
            else:
                no_improvement_count += 1
            
            # Save checkpoint periodically
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                print('Model checkpoint saved')
            
            # Early stopping
            if early_stopping and no_improvement_count >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                print(f"Best model was at epoch {best_epoch} with combined score {best_reward:.4f} (utility: {best_utility:.4f})")
                break
        
        # End WandB session if it was used
        if use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: Error ending W&B session: {e}")
        
        # Return best model info
        self.best_reward = best_reward
        self.best_epoch = best_epoch
        self.best_utility = best_utility
        
        return {
            'best_epoch': best_epoch,
            'best_reward': best_reward,
            'best_utility': best_utility,
            'training_stats': self.training_stats
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoints"""
        if is_best:
            self.generator.save_weights(f"checkpoints/generator_best.weights.h5")
            self.discriminator.save_weights(f"checkpoints/discriminator_best.weights.h5")
            self.critic.save_weights(f"checkpoints/critic_best.weights.h5")
        else:
            self.generator.save_weights(f"checkpoints/generator_{epoch}.weights.h5")
            self.discriminator.save_weights(f"checkpoints/discriminator_{epoch}.weights.h5")
            self.critic.save_weights(f"checkpoints/critic_{epoch}.weights.h5")
    
    def load_checkpoint(self, epoch=None, load_best=False):
        """Load model checkpoints"""
        if load_best:
            self.generator.load_weights(f"checkpoints/generator_best.weights.h5")
            self.discriminator.load_weights(f"checkpoints/discriminator_best.weights.h5")
            self.critic.load_weights(f"checkpoints/critic_best.weights.h5")
            print("Loaded best model checkpoints")
        else:
            self.generator.load_weights(f"checkpoints/generator_{epoch}.weights.h5")
            self.discriminator.load_weights(f"checkpoints/discriminator_{epoch}.weights.h5")
            self.critic.load_weights(f"checkpoints/critic_{epoch}.weights.h5")
            print(f"Loaded model checkpoints from epoch {epoch}")

    def generate_trajectories(self, real_trajs, num_samples=1):
        """
        Generate synthetic trajectories from real ones.
        
        Args:
            real_trajs: List of trajectory components from real data
            num_samples: Number of synthetic samples to generate per real trajectory
            
        Returns:
            gen_trajs: List of generated trajectories
        """
        batch_size = real_trajs[0].shape[0]
        
        # Create a list to hold all generated trajectories
        all_gen_trajs = []
        
        for _ in range(num_samples):
            # Generate random noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Add noise to real trajectories
            trajs_with_noise = real_trajs + [noise]
            
            # Generate synthetic trajectories
            gen_trajs = self.generator.predict(trajs_with_noise)
            
            # Store generated trajectories
            all_gen_trajs.append(gen_trajs)
        
        return all_gen_trajs

    # Override get_config method to include RL specific parameters
    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        # Add RL-specific parameters
        config.update({
            "clip_epsilon": self.clip_epsilon,
            "critic_discount": self.critic_discount,
            "entropy_beta": self.entropy_beta,
            "gamma": self.gamma
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create instance from configuration."""
        # Extract required parameters only
        init_config = {
            "latent_dim": config["latent_dim"],
            "keys": config["keys"],
            "vocab_size": config["vocab_size"],
            "max_length": config["max_length"],
            "lat_centroid": config["lat_centroid"],
            "lon_centroid": config["lon_centroid"],
            "scale_factor": config["scale_factor"]
        }
        return cls(**init_config) 