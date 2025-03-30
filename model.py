import tensorflow as tf
import keras
import numpy as np
import random
import scipy.spatial.distance as distance
import os
from datetime import datetime

# Set random seeds for reproducibility
random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)

from keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding
from keras.layers import MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from keras.initializers import he_uniform
from keras.regularizers import l1

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

# Define custom layer for expanding noise
class ExpandNoise(tf.keras.layers.Layer):
    def __init__(self, max_length, **kwargs):
        super(ExpandNoise, self).__init__(**kwargs)
        self.max_length = max_length
        
    def call(self, inputs):
        return tf.tile(tf.expand_dims(inputs, 1), [1, self.max_length, 1])
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.max_length, input_shape[1])
            
# Transformer encoder block implementation
def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-head attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    
    # Add & normalize (first residual connection)
    x = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # Feed Forward network
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    
    # Add & normalize (second residual connection)
    return LayerNormalization(epsilon=1e-6)(x + ff_output)

def positional_encoding(length, depth, use_both=False):
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    
    if use_both:
        pos_encoding = np.concatenate(
            [np.sin(angle_rads), np.cos(angle_rads)],
            axis=-1)
    else:
        # Just use sine for even indices and cosine for odd indices
        pos_encoding = np.zeros((length, depth))
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0:depth:2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 0:depth:2])
    
    return tf.cast(pos_encoding, dtype=tf.float32)

class RL_Transformer_TrajGAN():
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        self.keys = keys
        self.vocab_size = vocab_size
        
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        
        self.x_train = None
        
        # Transformer hyperparameters
        self.head_size = 64
        self.num_heads = 8
        self.ff_dim = 256
        self.transformer_blocks = 4
        self.dropout_rate = 0.1
        
        # Define the optimizers
        self.generator_optimizer = Adam(0.001, 0.5)
        self.discriminator_optimizer = Adam(0.001, 0.5)
        self.critic_optimizer = Adam(0.001)
        self.actor_optimizer = Adam(0.0005)

        # PPO hyperparameters
        self.ppo_epsilon = 0.2  # Clipping parameter
        self.value_coeff = 0.5  # Value function coefficient
        self.entropy_coeff = 0.01  # Entropy coefficient
        
        # Reward hyperparameters
        self.alpha = 0.2  # Privacy weight
        self.beta = 0.6   # Utility weight
        self.gamma = 0.2  # Adversarial (realism) weight
        
        # Build actor-critic model for RL
        self.actor_model, self.critic_model = self.build_actor_critic()
        
        # Build the trajectory discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', 
                                  optimizer=self.discriminator_optimizer, 
                                  metrics=['accuracy'])
        
        # Build the TUL model for privacy evaluation
        self.tul_model = self.build_tul_model()
        self.tul_model.compile(loss='sparse_categorical_crossentropy',
                              optimizer=Adam(0.001),
                              metrics=['accuracy'])
        
        # Evaluation buffer (for computing rewards)
        self.replay_buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': []
        }
    
    def build_actor_critic(self):
        """Build actor (policy) and critic (value) networks with proper policy distribution outputs."""
        # Common inputs for both actor and critic
        inputs = []
        embeddings = []
        
        # Create input layers
        mask = Input(shape=(self.max_length, 1), name='input_mask')
        
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                inputs.append(mask)
                continue
            elif key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                e = TimeDistributed(Dense(64, activation='relu', kernel_initializer=he_uniform(seed=1)))(i)
            else:
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key)
                e = TimeDistributed(Dense(self.vocab_size[key], activation='relu', kernel_initializer=he_uniform(seed=1)))(i)
                
            inputs.append(i)
            embeddings.append(e)
            
        # Add noise input for the actor only
        noise = Input(shape=(self.latent_dim,), name='input_noise')
        inputs.append(noise)
        
        # Concatenate embeddings
        concat_embeddings = Concatenate(axis=2)(embeddings)
        
        # Project to transformer dimension
        x = TimeDistributed(Dense(self.head_size * self.num_heads, 
                                 use_bias=True, 
                                 activation='relu', 
                                 kernel_initializer=he_uniform(seed=1)))(concat_embeddings)
        
        # Add positional encoding
        pos_encoding = Lambda(
            lambda x: x + positional_encoding(self.max_length, self.head_size * self.num_heads),
            output_shape=(self.max_length, self.head_size * self.num_heads)
        )(x)
        
        # Add noise to every position using custom layer
        expanded_noise = ExpandNoise(self.max_length)(noise)
        
        # Concatenate noise to features for actor
        x_with_noise = Concatenate(axis=2)([pos_encoding, expanded_noise])
        x_with_noise = Dense(self.head_size * self.num_heads, activation='relu')(x_with_noise)
        
        # Apply transformer blocks
        transformer_output = x_with_noise
        for _ in range(self.transformer_blocks):
            transformer_output = transformer_encoder_block(
                transformer_output, 
                head_size=self.head_size,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout_rate
            )
        
        # Actor outputs (policy distributions)
        # For lat_lon: use Gaussian distribution (mean and log std)
        latlon_mean = TimeDistributed(Dense(2, activation='tanh'), name='latlon_mean')(transformer_output)
        latlon_mean_scaled = Lambda(
            lambda x: x * self.scale_factor,
            output_shape=(self.max_length, 2)
        )(latlon_mean)
        
        # Log std dev for each lat/lon coordinate (initialized to small negative value)
        latlon_logstd = TimeDistributed(Dense(2, activation='linear', 
                                           kernel_initializer=tf.keras.initializers.Constant(-1.0)), 
                                       name='latlon_logstd')(transformer_output)
        
        # For categorical outputs: use softmax distributions
        day_logits = TimeDistributed(Dense(self.vocab_size['day']), name='day_logits')(transformer_output)
        day_probs = TimeDistributed(Dense(self.vocab_size['day'], activation='softmax'), 
                                   name='day_probs')(day_logits)
        
        hour_logits = TimeDistributed(Dense(self.vocab_size['hour']), name='hour_logits')(transformer_output)
        hour_probs = TimeDistributed(Dense(self.vocab_size['hour'], activation='softmax'), 
                                    name='hour_probs')(hour_logits)
        
        category_logits = TimeDistributed(Dense(self.vocab_size['category']), name='category_logits')(transformer_output)
        category_probs = TimeDistributed(Dense(self.vocab_size['category'], activation='softmax'), 
                                        name='category_probs')(category_logits)
        
        # Add mask output (pass through)
        mask_output = Lambda(lambda x: x, name='mask_output')(mask)
        
        # Actor model with multiple outputs
        actor_model = Model(
            inputs=inputs, 
            outputs=[latlon_mean_scaled, latlon_logstd, day_probs, hour_probs, category_probs, mask_output],
            name='actor'
        )
        
        # Critic model without noise input and with only value output
        # Create a separate set of inputs for critic
        critic_inputs = inputs[:-1]  # All inputs except noise
        critic_embeddings = embeddings
        
        # Process inputs with transformer
        critic_concat = Concatenate(axis=2)(critic_embeddings)
        critic_projected = TimeDistributed(Dense(self.head_size * self.num_heads,
                                               use_bias=True,
                                               activation='relu'))(critic_concat)
        
        critic_pos_encoding = Lambda(
            lambda x: x + positional_encoding(self.max_length, self.head_size * self.num_heads)
        )(critic_projected)
        
        # Apply transformer blocks
        critic_transformer = critic_pos_encoding
        for _ in range(self.transformer_blocks):
            critic_transformer = transformer_encoder_block(
                critic_transformer,
                head_size=self.head_size,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout_rate
            )
        
        # Global pooling for value output
        critic_pooled = GlobalAveragePooling1D()(critic_transformer)
        value_output = Dense(1, activation='linear', name='value')(critic_pooled)
        
        # Critic model
        critic_model = Model(inputs=critic_inputs, outputs=value_output, name='critic')
        
        return actor_model, critic_model
    
    def build_discriminator(self):
        """Build the trajectory discriminator model using transformer architecture."""
        # Input Layer
        inputs = []
        
        # Embedding Layer
        embeddings = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            if key == 'lat_lon':
                i = Input(shape=(self.max_length, self.vocab_size[key]),
                          name='input_disc_' + key)

                # Dense embedding instead of unstacking
                e = TimeDistributed(Dense(64, activation='relu', kernel_initializer=he_uniform(seed=1)))(i)

            else:
                i = Input(shape=(self.max_length,self.vocab_size[key]), name='input_disc_' + key)
                e = TimeDistributed(Dense(self.vocab_size[key], activation='relu', kernel_initializer=he_uniform(seed=1)))(i)
                
            inputs.append(i)
            embeddings.append(e)
            
        # Concatenate embeddings
        concat_embeddings = Concatenate(axis=2)(embeddings)
        
        # Project to transformer dimension
        x = TimeDistributed(Dense(self.head_size * self.num_heads, 
                                 use_bias=True, 
                                 activation='relu', 
                                 kernel_initializer=he_uniform(seed=1)))(concat_embeddings)
        
        # Add positional encoding
        pos_encoding = Lambda(
            lambda x: x + positional_encoding(self.max_length, self.head_size * self.num_heads),
            output_shape=(self.max_length, self.head_size * self.num_heads)
        )(x)
        
        # Apply transformer blocks
        transformer_output = pos_encoding
        for _ in range(self.transformer_blocks):
            transformer_output = transformer_encoder_block(
                transformer_output, 
                head_size=self.head_size,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout_rate
            )
        
        # Global average pooling
        pooled = GlobalAveragePooling1D()(transformer_output)
        
        # Output
        sigmoid = Dense(1, activation='sigmoid', name='discriminator_output')(pooled)

        return Model(inputs=inputs, outputs=sigmoid, name='discriminator')
    
    def build_tul_model(self):
        """Build the Trajectory-User Linking model to evaluate privacy."""
        # Input layers
        input_lat_lon = Input(shape=(self.max_length, 2), name='input_tul_lat_lon')
        input_day = Input(shape=(self.max_length, self.vocab_size['day']), name='input_tul_day')
        input_hour = Input(shape=(self.max_length, self.vocab_size['hour']), name='input_tul_hour')
        input_category = Input(shape=(self.max_length, self.vocab_size['category']), name='input_tul_category')
        
        # Process inputs
        processed_lat_lon = TimeDistributed(Dense(100))(input_lat_lon)
        processed_day = TimeDistributed(Dense(100))(input_day)
        processed_hour = TimeDistributed(Dense(100))(input_hour)
        processed_category = TimeDistributed(Dense(100))(input_category)
        
        # Concatenate processed inputs
        concat = Concatenate(axis=2)([processed_lat_lon, processed_day, processed_hour, processed_category])
        
        # Dropout and LSTM
        dropout1 = Dropout(0.5)(concat)
        lstm = LSTM(units=50, recurrent_regularizer=l1(0.02))(dropout1)
        
        # Final dropout and output layer
        dropout2 = Dropout(0.5)(lstm)
        
        # Assume num_users is 193 based on previous code
        num_users = 193
        output = Dense(units=num_users, kernel_initializer=he_uniform(), activation='softmax', name='tul_output')(dropout2)
        
        # Create model
        model = Model(inputs=[input_lat_lon, input_day, input_hour, input_category], 
                     outputs=output,
                     name='tul_model')
        
        return model
    
    # Fix for the sample_actions method in RL_Transformer_TrajGAN_Fixed.py

    def sample_actions(self, actor_outputs, deterministic=False):
        """Sample actions from policy distributions."""
        # Unpack actor outputs
        latlon_mean, latlon_logstd, day_probs, hour_probs, category_probs, mask = actor_outputs
        batch_size = tf.shape(latlon_mean)[0]
        
        # Apply the mask to prioritize valid timesteps - make sure dimensions match
        valid_mask = tf.cast(mask > 0, tf.float32)
        
        # Sample lat_lon from Gaussian
        if deterministic:
            latlon_actions = latlon_mean
        else:
            latlon_std = tf.exp(latlon_logstd)
            epsilon = tf.random.normal(tf.shape(latlon_mean))
            latlon_actions = latlon_mean + epsilon * latlon_std
        
        # Apply mask to latlon - shape should be [batch_size, max_length, 2]
        latlon_actions = latlon_actions * tf.cast(tf.reshape(valid_mask, [batch_size, self.max_length, 1]), tf.float32)
        
        # Sample categorical actions
        if deterministic:
            day_actions = tf.one_hot(tf.argmax(day_probs, axis=-1), depth=self.vocab_size['day'])
            hour_actions = tf.one_hot(tf.argmax(hour_probs, axis=-1), depth=self.vocab_size['hour'])
            category_actions = tf.one_hot(tf.argmax(category_probs, axis=-1), depth=self.vocab_size['category'])
        else:
            # Reshape for categorical sampling
            day_flat = tf.reshape(day_probs, [-1, self.vocab_size['day']])
            hour_flat = tf.reshape(hour_probs, [-1, self.vocab_size['hour']])
            category_flat = tf.reshape(category_probs, [-1, self.vocab_size['category']])
            
            # Sample from categorical distributions
            day_indices = tf.random.categorical(tf.math.log(day_flat + 1e-10), 1)
            hour_indices = tf.random.categorical(tf.math.log(hour_flat + 1e-10), 1)
            category_indices = tf.random.categorical(tf.math.log(category_flat + 1e-10), 1)
            
            # Reshape back to batch form
            day_indices = tf.reshape(day_indices, [batch_size, self.max_length])
            hour_indices = tf.reshape(hour_indices, [batch_size, self.max_length])
            category_indices = tf.reshape(category_indices, [batch_size, self.max_length])
            
            # Convert to one-hot
            day_actions = tf.one_hot(day_indices, depth=self.vocab_size['day'])
            hour_actions = tf.one_hot(hour_indices, depth=self.vocab_size['hour'])
            category_actions = tf.one_hot(category_indices, depth=self.vocab_size['category'])
        
        # Apply mask correctly - reshape valid_mask to match required dimensions
        # The issue is in this section: the valid_mask shape is [batch_size, max_length, 1]
        # but we need it as [batch_size, max_length, 1] for broadcasting with one-hot encodings
        reshaped_mask = tf.reshape(valid_mask, [batch_size, self.max_length, 1])
        
        # Now apply mask to categorical actions
        day_actions = day_actions * tf.cast(reshaped_mask, tf.float32)
        hour_actions = hour_actions * tf.cast(reshaped_mask, tf.float32)
        category_actions = category_actions * tf.cast(reshaped_mask, tf.float32)
        
        return [latlon_actions, day_actions, hour_actions, category_actions, mask]
    
    # Fix for the compute_log_probs method in RL_Transformer_TrajGAN_Fixed.py

    # Fix for the shape assertion in compute_log_probs method

    def compute_log_probs(self, actions, actor_outputs):
        """Compute log probabilities of actions under current policy."""
        latlon_actions, day_actions, hour_actions, category_actions, mask = actions
        latlon_mean, latlon_logstd, day_probs, hour_probs, category_probs, _ = actor_outputs
        
        # Get batch size and sequence length
        batch_size = tf.shape(latlon_actions)[0]
        sequence_length = tf.shape(latlon_actions)[1]
        
        # Get valid mask
        valid_mask = tf.cast(mask > 0, tf.float32)
        
        # Compute log prob for Gaussian (lat_lon)
        latlon_std = tf.exp(latlon_logstd)
        log_prob_gaussian = -0.5 * (
            tf.math.log(2.0 * np.pi) + 
            2.0 * latlon_logstd + 
            ((latlon_actions - latlon_mean) / latlon_std) ** 2
        )
        
        # Sum across lat/lon dimensions
        log_prob_latlon = tf.reduce_sum(log_prob_gaussian, axis=-1)  # Sum across lat/lon dims
        
        # Compute log probs for categorical distributions
        # For day actions
        log_day_probs = tf.reduce_sum(day_actions * tf.math.log(day_probs + 1e-10), axis=-1)
        
        # For hour actions
        log_hour_probs = tf.reduce_sum(hour_actions * tf.math.log(hour_probs + 1e-10), axis=-1)
        
        # For category actions
        log_category_probs = tf.reduce_sum(category_actions * tf.math.log(category_probs + 1e-10), axis=-1)
        
        # Combine log probs for each timestep
        per_timestep_log_probs = log_prob_latlon + log_day_probs + log_hour_probs + log_category_probs
        
        # Apply mask to only consider valid timesteps - make sure the shapes match
        # Instead of using shape assertion which is causing problems, print shapes for debugging
        # and ensure mask has correct shape
        
        # Ensure valid_mask has same shape as per_timestep_log_probs
        # Both should be [batch_size, sequence_length]
        valid_mask_reshaped = tf.reshape(valid_mask, tf.shape(per_timestep_log_probs))
        
        # Now use the reshaped mask
        masked_log_probs = per_timestep_log_probs * valid_mask_reshaped
        
        # Sum across timesteps and normalize by valid timesteps count
        valid_timesteps = tf.reduce_sum(valid_mask_reshaped, axis=1)
        trajectory_log_probs = tf.reduce_sum(masked_log_probs, axis=1) / (valid_timesteps + 1e-10)
        
        return trajectory_log_probs
    
    def compute_entropy(self, actor_outputs):
        """Compute entropy of the policy for exploration."""
        _, latlon_logstd, day_probs, hour_probs, category_probs, mask = actor_outputs
        
        # Get valid mask
        valid_mask = tf.cast(mask > 0, tf.float32)
        
        # Entropy of Gaussian is 0.5 * log(2*pi*e*sigma^2)
        latlon_std = tf.exp(latlon_logstd)
        gaussian_entropy = 0.5 * tf.math.log(2.0 * np.pi * np.e) + latlon_logstd
        gaussian_entropy = tf.reduce_sum(gaussian_entropy, axis=-1)  # Sum across lat/lon dims
        
        # Entropy of categorical distributions: -sum(p * log(p))
        day_entropy = -tf.reduce_sum(day_probs * tf.math.log(day_probs + 1e-10), axis=-1)
        hour_entropy = -tf.reduce_sum(hour_probs * tf.math.log(hour_probs + 1e-10), axis=-1)
        category_entropy = -tf.reduce_sum(category_probs * tf.math.log(category_probs + 1e-10), axis=-1)
        
        # Combine entropies for each timestep
        per_timestep_entropy = gaussian_entropy + day_entropy + hour_entropy + category_entropy
        
        # Reshape valid_mask to match per_timestep_entropy
        valid_mask_reshaped = tf.reshape(valid_mask, tf.shape(per_timestep_entropy))
        
        # Apply mask
        masked_entropy = per_timestep_entropy * valid_mask_reshaped
        
        # Average across valid timesteps
        batch_size = tf.shape(day_probs)[0]
        sequence_length = tf.shape(day_probs)[1]
        
        # Sum across timesteps and normalize by valid timesteps count
        valid_timesteps = tf.reduce_sum(valid_mask_reshaped, axis=1)
        avg_entropy = tf.reduce_sum(masked_entropy, axis=1) / (valid_timesteps + 1e-10)
        
        return avg_entropy
    
    def compute_gae(self, rewards, values, next_values=None, gamma=0.99, lam=0.95, use_mask=None):
        """Compute Generalized Advantage Estimation."""
        batch_size = len(rewards)
        advantages = np.zeros_like(rewards)
        
        for i in range(batch_size):
            # If mask provided, only consider valid timesteps
            if use_mask is not None:
                valid_indices = np.where(use_mask[i] > 0)[0]
                if len(valid_indices) == 0:
                    continue
                
                # Extract only valid timesteps
                batch_rewards = rewards[i][valid_indices]
                batch_values = values[i][valid_indices]
                
                if next_values is not None:
                    batch_next_values = next_values[i][valid_indices]
                else:
                    # Append a 0 as the next value of the last timestep
                    batch_next_values = np.append(batch_values[1:], 0)
            else:
                batch_rewards = rewards[i]
                batch_values = values[i]
                
                if next_values is not None:
                    batch_next_values = next_values[i]
                else:
                    # Append a 0 as the next value of the last timestep
                    batch_next_values = np.append(batch_values[1:], 0)
            
            # Calculate TD errors: δt = rt + γ*V(s_{t+1}) - V(s_t)
            deltas = batch_rewards + gamma * batch_next_values - batch_values
            
            # Compute GAE-λ advantage
            last_gae = 0
            for t in reversed(range(len(batch_rewards))):
                last_gae = deltas[t] + gamma * lam * last_gae
                
                if use_mask is not None:
                    advantages[i][valid_indices[t]] = last_gae
                else:
                    advantages[i][t] = last_gae
        
        return advantages
    
    def compute_rewards(self, real_trajs, gen_trajs, user_ids):
        """Compute rewards based on privacy, utility, and realism."""
        batch_size = len(real_trajs[0])
        
        # Unpack trajectories
        real_latlon, real_day, real_hour, real_category, real_mask = real_trajs
        gen_latlon, gen_day, gen_hour, gen_category, gen_mask = gen_trajs
        
        # Get valid mask - convert to numpy for processing
        valid_mask = np.array(real_mask > 0, dtype=np.float32)
        
        # Convert tensors to numpy arrays for non-TF operations
        gen_latlon_np = gen_latlon.numpy() if hasattr(gen_latlon, 'numpy') else gen_latlon
        gen_day_np = gen_day.numpy() if hasattr(gen_day, 'numpy') else gen_day
        gen_hour_np = gen_hour.numpy() if hasattr(gen_hour, 'numpy') else gen_hour
        gen_category_np = gen_category.numpy() if hasattr(gen_category, 'numpy') else gen_category
        
        # 1. Evaluate privacy using TUL model
        tul_inputs = [gen_latlon_np, gen_day_np, gen_hour_np, gen_category_np]
        tul_outputs = self.tul_model.predict(tul_inputs)
        
        # Calculate privacy rewards
        privacy_rewards = np.zeros((batch_size,))
        for i in range(batch_size):
            user_id = user_ids[i] % tul_outputs.shape[1]  # Ensure valid user_id
            p_user = tul_outputs[i, user_id]
            # Higher reward for lower identification probability
            privacy_rewards[i] = -np.log(p_user + 1e-10)
        
        # 2. Calculate utility rewards (spatial and semantic similarity)
        utility_rewards = np.zeros((batch_size,))
        for i in range(batch_size):
            mask_i = valid_mask[i].reshape(-1)
            valid_indices = np.where(mask_i > 0)[0]
            
            if len(valid_indices) > 0:
                # Spatial distance between real and generated trajectory points
                real_latlon_i = real_latlon[i][valid_indices]
                gen_latlon_i = gen_latlon_np[i][valid_indices]
                
                spatial_dist = np.mean(np.sqrt(np.sum((real_latlon_i - gen_latlon_i) ** 2, axis=1)))
                
                # Semantic similarity for categorical features
                sem_dist = 0
                
                # Day similarity
                real_day_i = real_day[i][valid_indices]
                gen_day_i = gen_day_np[i][valid_indices]
                day_js_div = self._compute_js_divergence(real_day_i, gen_day_i)
                
                # Hour similarity
                real_hour_i = real_hour[i][valid_indices]
                gen_hour_i = gen_hour_np[i][valid_indices]
                hour_js_div = self._compute_js_divergence(real_hour_i, gen_hour_i)
                
                # Category similarity
                real_category_i = real_category[i][valid_indices]
                gen_category_i = gen_category_np[i][valid_indices]
                category_js_div = self._compute_js_divergence(real_category_i, gen_category_i)
                
                # Combine semantic distances
                sem_dist = (day_js_div + hour_js_div + category_js_div) / 3.0
                
                # Total utility reward (negative distance)
                utility_rewards[i] = -1.0 * (0.7 * spatial_dist + 0.3 * sem_dist)
            else:
                utility_rewards[i] = 0.0  # No valid points
        
        # 3. Calculate adversarial rewards using discriminator
        disc_inputs = [gen_latlon_np, gen_day_np, gen_hour_np, gen_category_np]
        disc_outputs = self.discriminator.predict(disc_inputs)
        
        # Higher reward for fooling the discriminator
        adv_rewards = np.log(disc_outputs + 1e-10).flatten()
        
        # Combine rewards with weights
        final_rewards = (
            self.alpha * privacy_rewards + 
            self.beta * utility_rewards + 
            self.gamma * adv_rewards
        )
        
        # Return all reward components for monitoring
        return final_rewards, privacy_rewards, utility_rewards, adv_rewards
        
    def train_ppo(self, real_trajs, user_ids, ppo_epochs=4, clip_ratio=0.2, target_kl=0.01):
            """Train the model using PPO algorithm."""
            batch_size = len(real_trajs[0])
            
            # Step 1: Generate trajectories using current policy
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Prepare actor inputs
            actor_inputs = real_trajs + [noise]
            
            # Generate policy outputs
            actor_outputs = self.actor_model.predict(actor_inputs)
            
            # Sample actions from the policy
            actions = self.sample_actions(actor_outputs)
            
            # Calculate log probabilities of sampled actions
            old_log_probs = self.compute_log_probs(actions, actor_outputs)
            
            # Step 2: Evaluate trajectories
            rewards, privacy_rewards, utility_rewards, adv_rewards = self.compute_rewards(
                real_trajs, actions, user_ids
            )
            
            # Step 3: Estimate values
            critic_inputs = real_trajs
            values = self.critic_model.predict(critic_inputs).flatten()
            
            # Step 4: Calculate advantages
            advantages = rewards - values
            
            # Normalize advantages
            normalized_advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            # Convert to tensors
            advantages_tensor = tf.constant(normalized_advantages, dtype=tf.float32)
            rewards_tensor = tf.constant(rewards, dtype=tf.float32)
            old_log_probs_tensor = tf.constant(old_log_probs, dtype=tf.float32)
            
            # Step 5: PPO update loop
            actor_losses = []
            critic_losses = []
            kl_divs = []
            
            for _ in range(ppo_epochs):
                # Update critic (value function)
                with tf.GradientTape() as critic_tape:
                    current_values = self.critic_model(critic_inputs)
                    current_values = tf.reshape(current_values, [-1])
                    
                    # Calculate value loss
                    value_loss = tf.reduce_mean(tf.square(rewards_tensor - current_values))
                    
                # Get critic gradients
                critic_grads = critic_tape.gradient(value_loss, self.critic_model.trainable_variables)
                
                # Apply critic gradients
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))
                
                # Update actor (policy)
                with tf.GradientTape() as actor_tape:
                    # Get current policy outputs
                    current_actor_outputs = self.actor_model(actor_inputs)
                    
                    # Calculate log probabilities under current policy
                    current_log_probs = self.compute_log_probs(actions, current_actor_outputs)
                    
                    # Calculate ratio between old and new policy
                    ratio = tf.exp(current_log_probs - old_log_probs_tensor)
                    
                    # Clipped surrogate objective
                    clip_1 = ratio * advantages_tensor
                    clip_2 = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages_tensor
                    
                    # Surrogate loss (negative for gradient ascent)
                    surrogate_loss = -tf.reduce_mean(tf.minimum(clip_1, clip_2))
                    
                    # Add entropy bonus for exploration
                    entropy = tf.reduce_mean(self.compute_entropy(current_actor_outputs))
                    actor_loss = surrogate_loss - self.entropy_coeff * entropy
                    
                # Calculate actor gradients
                actor_grads = actor_tape.gradient(actor_loss, self.actor_model.trainable_variables)
                
                # Apply actor gradients
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))
                
                # Approximate KL divergence for early stopping
                # KL div = mean(log(old_policy) - log(new_policy))
                approx_kl = tf.reduce_mean(old_log_probs_tensor - current_log_probs)
                
                # Store metrics
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(value_loss.numpy())
                kl_divs.append(approx_kl.numpy())
                
                # Early stopping based on KL divergence
                if approx_kl > target_kl:
                    break
            
            # Return metrics for monitoring
            metrics = {
                'actor_loss': np.mean(actor_losses),
                'critic_loss': np.mean(critic_losses),
                'kl_div': np.mean(kl_divs),
                'entropy': entropy.numpy(),
                'privacy_reward': np.mean(privacy_rewards),
                'utility_reward': np.mean(utility_rewards),
                'adv_reward': np.mean(adv_rewards),
                'total_reward': np.mean(rewards)
            }
            
            return metrics
    
    def train_discriminator(self, real_trajs, gen_trajs):
        """Train the discriminator on real and generated trajectories."""
        batch_size = len(real_trajs[0])
        
        # Unpack trajectories
        real_latlon, real_day, real_hour, real_category, _ = real_trajs
        gen_latlon, gen_day, gen_hour, gen_category, _ = gen_trajs
        
        # Create labels
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))
        
        # Train on real trajectories
        d_loss_real = self.discriminator.train_on_batch(
            [real_latlon, real_day, real_hour, real_category],
            real_labels
        )
        
        # Train on generated trajectories
        d_loss_fake = self.discriminator.train_on_batch(
            [gen_latlon, gen_day, gen_hour, gen_category],
            fake_labels
        )
        
        # Average loss
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        return {'discriminator_loss': d_loss[0], 'discriminator_acc': d_loss[1]}
    
    def train_epoch(self, X_train, batch_size, ppo_epochs=4):
        """Train for one complete epoch."""
        # Select random batch of trajectories
        idx = np.random.randint(0, X_train[0].shape[0], batch_size)
        user_ids = np.arange(batch_size) % 193  # Ensure valid user IDs
        
        # Prepare real trajectories
        real_trajs = [X_train[0][idx],  # latlon
                     X_train[1][idx],   # day
                     X_train[2][idx],   # hour
                     X_train[3][idx],   # category
                     X_train[4][idx]]   # mask
        
        # Train using PPO
        ppo_metrics = self.train_ppo(real_trajs, user_ids, ppo_epochs)
        
        # Generate trajectories for discriminator training
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        actor_inputs = real_trajs + [noise]
        actor_outputs = self.actor_model.predict(actor_inputs)
        gen_trajs = self.sample_actions(actor_outputs, deterministic=True)
        
        # Train discriminator
        disc_metrics = self.train_discriminator(real_trajs, gen_trajs)
        
        # Combine metrics
        metrics = {**ppo_metrics, **disc_metrics}
        
        return metrics
    
    def evaluate_model(self, X_val, user_ids):
        """Evaluate model performance on validation set."""
        batch_size = len(X_val[0])
        
        # Prepare validation trajectories
        val_trajs = [X_val[0],    # latlon
                    X_val[1],     # day
                    X_val[2],     # hour
                    X_val[3],     # category
                    X_val[4]]     # mask
        
        # Generate noise
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Generate trajectories
        actor_inputs = val_trajs + [noise]
        actor_outputs = self.actor_model.predict(actor_inputs)
        gen_trajs = self.sample_actions(actor_outputs, deterministic=True)
        
        # Evaluate privacy (using TUL model)
        tul_inputs = [gen_trajs[0], gen_trajs[1], gen_trajs[2], gen_trajs[3]]
        tul_outputs = self.tul_model.predict(tul_inputs)
        pred_users = np.argmax(tul_outputs, axis=1)
        
        # Calculate top-1 accuracy (lower is better for privacy)
        acc_at_1 = np.mean(pred_users == user_ids)
        
        # Calculate utility metrics (spatial distance)
        spatial_distances = []
        for i in range(batch_size):
            mask = val_trajs[4][i].reshape(-1) > 0
            if np.sum(mask) > 0:
                real_latlon = val_trajs[0][i][mask]
                gen_latlon = gen_trajs[0][i][mask]
                
                # Calculate average distance
                dist = np.mean(np.sqrt(np.sum((real_latlon - gen_latlon) ** 2, axis=1)))
                spatial_distances.append(dist)
        
        avg_spatial_dist = np.mean(spatial_distances)
        
        # Calculate adversarial score (discriminator output)
        disc_output = self.discriminator.predict(tul_inputs)
        avg_disc_output = np.mean(disc_output)
        
        # Return metrics
        return {
            'privacy_acc@1': acc_at_1,
            'utility_spatial_dist': avg_spatial_dist,
            'adv_score': avg_disc_output
        }
    
    def save_models(self, path):
        """Save all model weights to disk."""
        os.makedirs(path, exist_ok=True)
        
        self.actor_model.save_weights(f"{path}/actor_model.weights.h5")
        self.critic_model.save_weights(f"{path}/critic_model.weights.h5")
        self.discriminator.save_weights(f"{path}/discriminator_model.weights.h5")
        self.tul_model.save_weights(f"{path}/tul_model.weights.h5")
        
        print(f"Models saved to {path}")
    
    def load_models(self, path):
        """Load all model weights from disk."""
        self.actor_model.load_weights(f"{path}/actor_model.weights.h5")
        self.critic_model.load_weights(f"{path}/critic_model.weights.h5")
        self.discriminator.load_weights(f"{path}/discriminator_model.weights.h5")
        self.tul_model.load_weights(f"{path}/tul_model.weights.h5")
        
        print(f"Models loaded from {path}")
        
    def generate_trajectories(self, conditions, num_samples=1, deterministic=False):
        """Generate trajectories based on provided conditions."""
        # Prepare inputs
        if isinstance(conditions, list):
            # If conditions are provided as a list of trajectories
            inputs = conditions
        else:
            # If conditions are provided as a single trajectory
            inputs = [np.repeat(conditions[0], num_samples, axis=0),
                     np.repeat(conditions[1], num_samples, axis=0),
                     np.repeat(conditions[2], num_samples, axis=0),
                     np.repeat(conditions[3], num_samples, axis=0),
                     np.repeat(conditions[4], num_samples, axis=0)]
        
        # Generate noise
        noise = np.random.normal(0, 1, (inputs[0].shape[0], self.latent_dim))
        
        # Generate trajectories
        actor_inputs = inputs + [noise]
        actor_outputs = self.actor_model.predict(actor_inputs)
        gen_trajs = self.sample_actions(actor_outputs, deterministic=deterministic)
        
        return gen_trajs
    
    def _compute_js_divergence(self, real_dist, gen_dist):
        """Compute Jensen-Shannon divergence between distributions."""
        # Average distributions across timesteps
        real_avg = np.mean(real_dist, axis=0)
        gen_avg = np.mean(gen_dist, axis=0)
        
        # Normalize distributions
        real_norm = real_avg / (np.sum(real_avg) + 1e-10)
        gen_norm = gen_avg / (np.sum(gen_avg) + 1e-10)
        
        # Compute JS divergence
        m = 0.5 * (real_norm + gen_norm)
        js_div = 0.5 * (
            np.sum(real_norm * np.log((real_norm + 1e-10) / (m + 1e-10))) +
            np.sum(gen_norm * np.log((gen_norm + 1e-10) / (m + 1e-10)))
        )
        
        return js_div