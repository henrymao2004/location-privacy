import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Embedding, TimeDistributed, 
    Lambda, Layer, GlobalAveragePooling1D, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
import tensorflow_probability as tfp
import numpy as np
import os
import time
import wandb

# Import custom components
from models.transformer_components import (
    TransformerEncoderLayer, TransformerDecoderLayer, 
    PositionalEncoding, create_padding_mask, create_look_ahead_mask
)
from models.critic import CriticNetwork
from models.reward_function import TrajRewardFunction
from losses import d_bce_loss, trajLoss, compute_advantage

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
        
        # Build generator and discriminator
        self.generator = self.build_generator()
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
        
        # Initialize GAN combined model
        self.build_gan_model()
        
        # Save model architecture to JSON files
        if not os.path.exists("params"):
            os.makedirs("params")
        
        model_json = self.generator.to_json()
        with open("params/transformer_generator.json", "w") as json_file:
            json_file.write(model_json)
            
        model_json = self.discriminator.to_json()
        with open("params/transformer_discriminator.json", "w") as json_file:
            json_file.write(model_json)
    
    def build_generator(self):
        """
        Build the Transformer-based trajectory generator.
        
        Returns:
            model: Keras model for the generator
        """
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
        padding_mask = Lambda(lambda x: create_padding_mask(tf.reduce_sum(x, axis=-1)))(inputs[4])  # mask input
        
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
        noise_tiled = Lambda(lambda x: tf.tile(x, [1, self.max_length, 1]))(noise_expanded)
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
        
        return Model(inputs=inputs, outputs=outputs, name='generator')
    
    def build_discriminator(self):
        """
        Build the Transformer-based trajectory discriminator.
        
        Returns:
            model: Keras model for the discriminator
        """
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
        
        return Model(inputs=inputs, outputs=validity, name='discriminator')
    
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
        
        # Combined model
        self.combined = Model(inputs=inputs, outputs=validity)
        
        # Custom loss that considers both adversarial loss and trajectory properties
        self.combined.compile(
            loss=trajLoss(inputs, gen_trajs),
            optimizer=Adam(self.gen_lr)
        )


class RL_Enhanced_Transformer_TrajGAN(TransformerTrajGAN):
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor):
        """
        Initialize the RL-enhanced Transformer TrajGAN model with PPO.
        
        Args:
            Same as parent class TransformerTrajGAN
        """
        # Initialize parent GAN model
        super().__init__(latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor)
        
        # Initialize critic network for PPO
        self.critic = CriticNetwork(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dff=self.dff,
            max_length=self.max_length
        )
        
        # Initialize reward function
        self.reward_function = TrajRewardFunction()
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.critic_discount = 0.5
        self.entropy_beta = 0.01
        self.gamma = 0.99  # Discount factor
        
        # Policy optimization parameters
        self.policy_optimizer = Adam(learning_rate=0.0001)
        
        # For logging training stats
        self.training_stats = {
            'g_losses': [],
            'd_losses': [],
            'value_losses': [],
            'policy_losses': [],
            'rewards': [],
            'epochs': []
        }
        
        # Checkpoint management
        self.best_reward = -float('inf')
        self.best_epoch = 0
        
        # Initialize TUL classifier
        self.tul_classifier = None
        
        # Create checkpoint directory
        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')
    
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
        # Generate trajectory with current policy
        noise = tf.random.normal((states[0].shape[0], self.latent_dim))
        states_with_noise = states + [noise]
        current_outputs = self.generator(states_with_noise)
        
        # Extract action components
        gen_latlon = current_outputs[0]
        gen_category = current_outputs[1]
        gen_day = current_outputs[2]
        gen_hour = current_outputs[3]
        
        # Extract targets
        target_latlon = actions[0]
        target_category = actions[1]
        target_day = actions[2]
        target_hour = actions[3]
        
        # Compute log probabilities for each action component
        
        # For lat/lon (assuming normal distribution)
        latlon_dist = tfp.distributions.Normal(gen_latlon, 0.1)
        latlon_log_probs = tf.reduce_sum(latlon_dist.log_prob(target_latlon), axis=-1)
        
        # For categorical outputs
        category_log_probs = tf.reduce_sum(target_category * tf.math.log(gen_category + 1e-8), axis=-1)
        day_log_probs = tf.reduce_sum(target_day * tf.math.log(gen_day + 1e-8), axis=-1)
        hour_log_probs = tf.reduce_sum(target_hour * tf.math.log(gen_hour + 1e-8), axis=-1)
        
        # Combine log probs
        mask = states[4][:,:,0]  # Extract mask
        combined_log_probs = (
            tf.reduce_sum(latlon_log_probs * mask, axis=1) +
            tf.reduce_sum(category_log_probs * mask, axis=1) +
            tf.reduce_sum(day_log_probs * mask, axis=1) +
            tf.reduce_sum(hour_log_probs * mask, axis=1)
        )
        
        # Calculate ratio for clipped objective
        ratio = tf.exp(combined_log_probs - old_log_probs)
        
        # Clipped objective
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -tf.reduce_mean(
            tf.minimum(ratio * advantages, clipped_ratio * advantages)
        )
        
        # Value loss (MSE)
        value_pred = self.critic.model(states)
        value_loss = tf.reduce_mean(tf.square(returns - value_pred))
        
        # Entropy bonus for exploration
        entropy = tf.reduce_mean(
            -tf.reduce_sum(gen_category * tf.math.log(gen_category + 1e-8), axis=-1)
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
        # Tape the operations for gradient calculation
        with tf.GradientTape() as tape:
            policy_loss, value_loss, entropy = self.compute_ppo_loss(
                states, actions, old_log_probs, advantages, values, returns
            )
            # Total loss with entropy bonus
            total_loss = policy_loss + self.critic_discount * value_loss - self.entropy_beta * entropy
        
        # Get trainable variables
        trainable_vars = self.generator.trainable_variables
        
        # Calculate and apply gradients
        gradients = tape.gradient(total_loss, trainable_vars)
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        self.policy_optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return policy_loss, value_loss, entropy
    
    def train(self, epochs=200, batch_size=32, sample_interval=10, early_stopping=True, patience=10, min_delta=0.001):
        """
        Train the RL-enhanced model using PPO.
        
        Args:
            epochs: Number of training epochs
            batch_size: Size of training batches
            sample_interval: Interval for saving checkpoints
            early_stopping: Whether to use early stopping
            patience: Patience for early stopping
            min_delta: Minimum delta for improvement in early stopping
            
        Returns:
            stats: Dictionary of training statistics
        """
        # Initialize WandB logging
        wandb.init(project="privacy-traj-gan", name="rl-enhanced-trajgan")
        
        # Training data
        x_train = np.load('data/final_train.npy', allow_pickle=True)
        
        # Padding to reach the maxlength
        X_train = [tf.keras.preprocessing.sequence.pad_sequences(
            f, self.max_length, padding='pre', dtype='float64'
        ) for f in x_train]
        
        # Training loop
        best_reward = -float('inf')
        best_epoch = 0
        no_improvement_count = 0
        
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
            
            # Random noise
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            real_trajs.append(noise)
            
            # Generate synthetic trajectories
            gen_trajs = self.generator.predict(real_trajs)
            
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
            
            # GAN update - train generator to fool discriminator
            g_loss = self.combined.train_on_batch(real_trajs, real_labels)
            
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
            # Assuming user IDs are just the batch indices for this example
            user_ids = np.arange(batch_size)
            rewards = np.zeros((batch_size, 1))
            
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
            
            # Compute advantages and returns for PPO
            values = self.critic.predict_value([
                gen_trajs_for_ppo[0],   # lat_lon
                gen_trajs_for_ppo[3],   # category 
                tf.concat([gen_trajs_for_ppo[1], gen_trajs_for_ppo[2]], axis=2),  # time (day, hour)
                real_trajs[4]           # mask
            ])
            
            # Compute advantages (simplified for single-step trajectories)
            advantages = rewards - values
            returns = rewards
            
            # Compute log probabilities of actions under old policy
            # (Simplified, assuming we're using the actions we just generated)
            # In a real implementation, you'd track the actual log probs
            old_log_probs = np.zeros((batch_size,))  # Placeholder
            
            # Perform PPO update
            policy_loss, value_loss, entropy = self.optimize_policy(
                states=real_trajs[:5],  # Exclude noise
                actions=gen_trajs_for_ppo[:4],  # Exclude mask 
                old_log_probs=old_log_probs,
                advantages=advantages,
                values=values,
                returns=returns
            )
            
            # Update critic network
            critic_loss = self.critic.train_on_batch(
                [
                    gen_trajs_for_ppo[0],   # lat_lon
                    gen_trajs_for_ppo[3],   # category
                    tf.concat([gen_trajs_for_ppo[1], gen_trajs_for_ppo[2]], axis=2),  # time
                    real_trajs[4]           # mask
                ], 
                returns
            )
            
            # ========================
            # 4. Logging and Checkpoints
            # ========================
            
            # Calculate average reward
            avg_reward = np.mean(rewards)
            
            # Log metrics for this epoch
            epoch_time = time.time() - start_time
            print(f"[Epoch {epoch}/{epochs}] "
                  f"D loss: {d_loss[0]:.4f}, acc: {d_loss[1]:.4f} | "
                  f"G loss: {g_loss:.4f} | "
                  f"Policy loss: {policy_loss:.4f} | "
                  f"Value loss: {value_loss:.4f} | "
                  f"Avg reward: {avg_reward:.4f} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Log to WandB
            wandb.log({
                "epoch": epoch,
                "d_loss": d_loss[0],
                "d_accuracy": d_loss[1],
                "g_loss": g_loss,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "avg_reward": avg_reward,
                "critic_loss": critic_loss,
                "epoch_time": epoch_time
            })
            
            # Save for monitoring progress
            self.training_stats['g_losses'].append(g_loss)
            self.training_stats['d_losses'].append(d_loss[0])
            self.training_stats['value_losses'].append(value_loss)
            self.training_stats['policy_losses'].append(policy_loss)
            self.training_stats['rewards'].append(avg_reward)
            self.training_stats['epochs'].append(epoch)
            
            # Save checkpoint if improved
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_epoch = epoch
                no_improvement_count = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print(f"New best model with reward {best_reward:.4f}")
            else:
                no_improvement_count += 1
            
            # Save checkpoint periodically
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                print('Model checkpoint saved')
            
            # Early stopping
            if early_stopping and no_improvement_count >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                print(f"Best model was at epoch {best_epoch} with reward {best_reward:.4f}")
                break
        
        # End WandB session
        wandb.finish()
        
        # Return best model info
        self.best_reward = best_reward
        self.best_epoch = best_epoch
        
        return {
            'best_epoch': best_epoch,
            'best_reward': best_reward,
            'training_stats': self.training_stats
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoints"""
        if is_best:
            self.generator.save_weights(f"checkpoints/generator_best.h5")
            self.discriminator.save_weights(f"checkpoints/discriminator_best.h5")
            self.critic.save_weights(f"checkpoints/critic_best.h5")
        else:
            self.generator.save_weights(f"checkpoints/generator_{epoch}.h5")
            self.discriminator.save_weights(f"checkpoints/discriminator_{epoch}.h5")
            self.critic.save_weights(f"checkpoints/critic_{epoch}.h5")
    
    def load_checkpoint(self, epoch=None, load_best=False):
        """Load model checkpoints"""
        if load_best:
            self.generator.load_weights(f"checkpoints/generator_best.h5")
            self.discriminator.load_weights(f"checkpoints/discriminator_best.h5")
            self.critic.load_weights(f"checkpoints/critic_best.h5")
            print("Loaded best model checkpoints")
        else:
            self.generator.load_weights(f"checkpoints/generator_{epoch}.h5")
            self.discriminator.load_weights(f"checkpoints/discriminator_{epoch}.h5")
            self.critic.load_weights(f"checkpoints/critic_{epoch}.h5")
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