import tensorflow as tf
import keras
import numpy as np
import random
from tensorflow.keras import layers
import tensorflow_probability as tfp
import os
import json

random.seed(2020)
np.random.seed(2020)
tf.random.set_seed(2020)
from keras.layers import Input, Add, Average, Dense, LSTM, Lambda, TimeDistributed, Concatenate, Embedding, MultiHeadAttention, LayerNormalization, Dropout
from keras.initializers import he_uniform, glorot_uniform
from keras.regularizers import l1, l2

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

from losses import CustomTrajLoss

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

class KANLayer(layers.Layer):
    """Simplified implementation of a Kolmogorov-Arnold Network (KAN) layer."""
    def __init__(self, 
                 output_dim, 
                 grid_size=6,  # Using smaller grid by default
                 num_components=3,  # Using fewer components by default 
                 spline_order=1,  # Using linear splines (1) for simplicity
                 activation='tanh', 
                 reg_strength=0.0001,
                 **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.num_components = num_components
        self.spline_order = spline_order
        self.activation = activation
        self.reg_strength = reg_strength
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Simplified approach with fewer weights for stability
        # Projection matrices (one per component)
        self.projections = []
        for i in range(self.num_components):
            self.projections.append(
                self.add_weight(
                    shape=(input_dim, 1),
                    initializer=glorot_uniform(),
                    regularizer=l2(self.reg_strength),
                    trainable=True,
                    dtype=tf.float32,
                    name=f'projection_{i}'
                )
            )
        
        # Control points for linear interpolation
        self.control_points = []
        for i in range(self.num_components):
            self.control_points.append(
                self.add_weight(
                    shape=(self.grid_size, self.output_dim),
                    initializer=glorot_uniform(),
                    regularizer=l2(self.reg_strength),
                    trainable=True,
                    dtype=tf.float32,
                    name=f'control_points_{i}'
                )
            )
            
        # Component weights for combining outputs
        self.component_weights = self.add_weight(
            shape=(self.num_components,),
            initializer='ones',
            regularizer=l2(self.reg_strength),
            trainable=True,
            dtype=tf.float32,
            name='component_weights'
        )
        
        # Fixed grid for interpolation
        self.grid = tf.linspace(-1.0, 1.0, self.grid_size)
        
        self.built = True
    
    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return (input_shape[0], input_shape[1], self.output_dim)
    
    def call(self, inputs):
        """Apply the simplified KAN transformation."""
        # Cast inputs to float32 for consistency
        inputs = tf.cast(inputs, tf.float32)
        
        # Extract dimensions
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Reshape inputs to 2D for easier processing
        flat_inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])  # [batch*seq, input_dim]
        
        # Process each component separately and collect results
        component_outputs = []
        
        for i in range(self.num_components):
            # Project input to scalar value in range [-1, 1]
            projection = tf.matmul(flat_inputs, self.projections[i])  # [batch*seq, 1]
            projection = tf.clip_by_value(projection, -0.99, 0.99)  # Avoid exact boundary values
            
            # Convert to index in grid
            grid_idx = tf.cast((projection + 1.0) * 0.5 * (self.grid_size - 1), tf.int32)
            grid_idx = tf.clip_by_value(grid_idx, 0, self.grid_size - 2)  # Ensure valid indexing
            
            # Get adjacent control points for interpolation
            idx = tf.reshape(grid_idx, [-1])  # Flatten for gathering
            next_idx = idx + 1
            
            # Gather control points for each position
            points = self.control_points[i]  # [grid_size, output_dim]
            p0 = tf.gather(points, idx)  # [batch*seq, output_dim]
            p1 = tf.gather(points, next_idx)  # [batch*seq, output_dim]
            
            # Calculate interpolation weights
            projection_flat = tf.reshape(projection, [-1])  # Flatten
            grid_start = tf.gather(self.grid, idx)
            grid_end = tf.gather(self.grid, next_idx)
            t = (projection_flat - grid_start) / tf.maximum(grid_end - grid_start, 1e-6)
            t = tf.expand_dims(t, -1)  # Add dimension for broadcasting
            
            # Linear interpolation
            component_output = p0 + t * (p1 - p0)  # [batch*seq, output_dim]
            component_outputs.append(component_output)
        
        # Stack and weight components
        stacked = tf.stack(component_outputs, axis=1)  # [batch*seq, num_components, output_dim]
        
        # Apply component weights and sum
        weights = tf.reshape(self.component_weights, [1, -1, 1])  # [1, num_components, 1]
        weighted = stacked * weights  # [batch*seq, num_components, output_dim]
        
        # Sum across components
        summed = tf.reduce_sum(weighted, axis=1)  # [batch*seq, output_dim]
        
        # Apply activation
        if self.activation == 'tanh':
            activated = tf.tanh(summed)
        elif self.activation == 'sigmoid':
            activated = tf.sigmoid(summed)
        elif self.activation == 'relu':
            activated = tf.nn.relu(summed)
        elif self.activation == 'softmax':
            activated = tf.nn.softmax(summed, axis=-1)
        else:
            activated = summed
            
        # Reshape back to 3D
        output = tf.reshape(activated, [batch_size, seq_length, self.output_dim])
        
        return output
    
    def get_config(self):
        config = super(KANLayer, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'grid_size': self.grid_size,
            'num_components': self.num_components,
            'spline_order': self.spline_order,
            'activation': self.activation,
            'reg_strength': self.reg_strength
        })
        return config

class KAN_TrajGAN():
    """Trajectory Generator GAN using Kolmogorov-Arnold Networks."""
    
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor):
        self.latent_dim = latent_dim
        self.max_length = max_length
        
        self.keys = keys
        self.vocab_size = vocab_size
        
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        
        self.x_train = None
        
        # Define optimizers with appropriate learning rates and gradient clipping
        self.generator_optimizer = Adam(0.0001, clipnorm=1.0)
        self.discriminator_optimizer = Adam(0.00005, clipnorm=1.0)
        
        # Build networks
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Compile models
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=self.discriminator_optimizer
        )
        
        # Combined model for training
        self.setup_combined_model()
        
        # For compatibility with the test script
        self.discriminator_output_keys = self.keys[:4]  # lat_lon, day, hour, category

    def get_config(self):
        """Return the configuration of the model for serialization."""
        return {
            "latent_dim": self.latent_dim,
            "keys": self.keys,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "lat_centroid": self.lat_centroid,
            "lon_centroid": self.lon_centroid,
            "scale_factor": self.scale_factor
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
            
            print(f"Successfully loaded model from epoch {epoch}")
            return model
            
        except Exception as e:
            print(f"Error loading model from checkpoint: {e}")
            raise
    
    def _create_kan_block(self, inputs, hidden_dim, output_dim, activation):
        """Create a KAN block with enhanced spatial and categorical modeling."""
        # First layer with higher grid resolution for better spatial representation
        x = KANLayer(
            hidden_dim, 
            grid_size=8,  # Increased grid size for better spatial resolution
            num_components=4,  # More components for better categorical representation
            activation='relu'
        )(inputs)
        
        # Layer normalization for stability
        x = LayerNormalization(epsilon=1e-5)(x)
        
        # Second KAN layer
        x = KANLayer(
            output_dim, 
            grid_size=8,
            num_components=4,
            activation=activation
        )(x)
        
        # Residual connection if input and output dimensions match
        if inputs.shape[-1] == output_dim:
            x = Add()([x, inputs])
            
        return x

    def build_generator(self):
        """Build KAN-based generator with enhanced spatial and categorical branches."""
        # Input for latent vector
        z = Input(shape=(self.latent_dim,), name='input_latent', dtype=tf.float32)
        
        # Expand latent vector to sequence
        z_repeated = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, self.max_length, 1]))(z)
        
        # Projection to higher dimension
        x = Dense(128, activation='relu', kernel_initializer='glorot_uniform', dtype=tf.float32)(z_repeated)
        
        # Apply KAN blocks - use deeper architecture for better spatial modeling
        x = self._create_kan_block(x, 128, 128, 'relu')
        x = self._create_kan_block(x, 128, 128, 'relu')
        x = self._create_kan_block(x, 128, 96, 'relu')
        
        # Output layers with specialized branches for each attribute
        outputs = []
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                # For mask, generate a placeholder
                output_mask = Lambda(lambda x: tf.ones_like(x[:,:,:1], dtype=tf.float32))(x)
                outputs.append(output_mask)
            elif key == 'lat_lon':
                # Greatly enhanced spatial coordinates branch with deeper network
                # Create a dedicated spatial feature extraction subnet
                lat_lon_branch = Dense(128, activation='relu', dtype=tf.float32)(x)
                lat_lon_branch = Dense(96, activation='relu', dtype=tf.float32)(lat_lon_branch)
                
                # Add positional encoding to help with sequence ordering
                position_encoding = self._get_positional_encoding(self.max_length, 96)
                lat_lon_branch = lat_lon_branch + position_encoding
                
                # Add a spatial attention layer to focus on important position features
                attention_output = MultiHeadAttention(
                    num_heads=4, key_dim=24, dropout=0.1
                )(lat_lon_branch, lat_lon_branch)
                lat_lon_branch = Add()([lat_lon_branch, attention_output])
                lat_lon_branch = LayerNormalization(epsilon=1e-6)(lat_lon_branch)
                
                # Final spatial processing layers
                lat_lon_branch = Dense(64, activation='relu', dtype=tf.float32)(lat_lon_branch)
                lat_lon_branch = Dense(32, activation='relu', dtype=tf.float32)(lat_lon_branch)
                
                # Special KAN layer for spatial coordinates with higher grid resolution
                lat_lon_output = KANLayer(
                    2, 
                    grid_size=16,  # Higher resolution for spatial data
                    num_components=8,  # More components for better modeling
                    activation='tanh'
                )(lat_lon_branch)
                
                # Scale coordinates to proper range
                scale_factor_param = min(self.scale_factor, 8.0)
                output_scaled = Lambda(lambda x: x * scale_factor_param, dtype=tf.float32)(lat_lon_output)
                outputs.append(output_scaled)
            elif key == 'category':
                # Enhanced category branch with specialized KAN for categorical data
                category_branch = Dense(64, activation='relu', dtype=tf.float32)(x)
                category_branch = Dense(48, activation='relu', dtype=tf.float32)(category_branch)
                
                # Use KAN with increased components for better categorical distribution
                category_output = KANLayer(
                    self.vocab_size[key], 
                    grid_size=10,
                    num_components=5,
                    activation='softmax',
                    reg_strength=0.0001  # Reduced regularization for better fitting
                )(category_branch)
                outputs.append(category_output)
            else:
                # Standard branch for temporal features
                branch = Dense(48, activation='relu', dtype=tf.float32)(x)
                output_branch = Dense(self.vocab_size[key], activation='softmax', dtype=tf.float32)(branch)
                outputs.append(output_branch)
        
        return Model(inputs=z, outputs=outputs)

    def _get_positional_encoding(self, seq_length, d_model):
        """Create positional encoding tensor for sequence modeling."""
        # Initialize positional encoding with zeros
        position_encoding = np.zeros((1, seq_length, d_model), dtype=np.float32)
        
        # Compute position encodings
        positions = np.arange(seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sine to even indices in the array
        position_encoding[0, :, 0::2] = np.sin(positions * div_term)
        
        # Apply cosine to odd indices in the array
        position_encoding[0, :, 1::2] = np.cos(positions * div_term)
        
        return tf.constant(position_encoding, dtype=tf.float32)

    def build_discriminator(self):
        """Build discriminator for evaluating trajectories."""
        inputs = []
        embeddings = []
        
        # Process each feature with separate input branches
        for idx, key in enumerate(self.keys):
            if key == 'mask':
                continue
            else:
                # Input for features
                i = Input(shape=(self.max_length, self.vocab_size[key]), name='input_' + key, dtype=tf.float32)
                
                # Simple Dense layer processing for all features
                e = Dense(32, activation='relu', dtype=tf.float32)(i)
                
                inputs.append(i)
                embeddings.append(e)
        
        # Concatenate all feature embeddings
        concat_input = Concatenate(axis=2)(embeddings)
        
        # Process with Dense layers for stability
        x = Dense(64, activation='relu', dtype=tf.float32)(concat_input)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Final classification
        x = Dense(32, activation='relu', dtype=tf.float32)(x)
        sigmoid = Dense(1, activation='sigmoid', dtype=tf.float32)(x)
        
        return Model(inputs=inputs, outputs=sigmoid)

    def setup_combined_model(self):
        """Setup the combined GAN model for adversarial training."""
        # Latent input
        z = Input(shape=(self.latent_dim,), name='input_latent', dtype=tf.float32)
        
        # Generate trajectories
        gen_trajs = self.generator(z)
        
        # Discriminator predictions
        pred = self.discriminator(gen_trajs[:4])
        
        # Create the combined model
        self.combined = Model(z, pred)
        
        # Create a custom loss for trajectory optimization with original weights
        self.traj_loss = CustomTrajLoss(p_bce=1, p_latlon=10, p_cat=1, p_day=1, p_hour=1)
        
        # Stable loss function
        def stable_bce_loss(y_true, y_pred):
            # Ensure consistent tensor data types
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            # Clip predictions for numerical stability
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            
            # Binary cross entropy
            bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
            bce = tf.clip_by_value(bce, 0.0, 20.0)
            
            return tf.reduce_mean(bce)
        
        # Compile the combined model
        self.combined.compile(loss=stable_bce_loss, optimizer=self.generator_optimizer)
    
    def train_step(self, real_trajs, batch_size=256):
        """Perform one step of training for the KAN-GAN."""
        try:
            # Sample random noise for the generator
            z = tf.random.normal([batch_size, self.latent_dim])
            
            # Generate trajectories
            gen_trajs = self.generator.predict(z, verbose=0)
            
            # Ensure consistent data types - convert everything to float32
            real_trajs = [tf.cast(tensor, tf.float32) for tensor in real_trajs]
            gen_trajs = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
            
            # Safe conversions with clipping
            real_trajs = [tf.clip_by_value(tensor, -1e6, 1e6) for tensor in real_trajs]
            gen_trajs = [tf.clip_by_value(tensor, -1e6, 1e6) for tensor in gen_trajs]
            
            # Update the trajectory loss
            self.traj_loss.set_trajectories(real_trajs, gen_trajs)
            
            # Train the discriminator (with label smoothing for stability)
            # Add stronger label smoothing to combat overfitting
            real_labels = tf.ones((batch_size, 1), dtype=tf.float32) * 0.85  # Less confident real label
            fake_labels = tf.zeros((batch_size, 1), dtype=tf.float32) + 0.15  # More positive fake label
            
            # Add more significant noise to real data for discriminator training to prevent overfitting
            noise_factor = 0.1
            noisy_real = [
                tensor + tf.random.normal(tf.shape(tensor), mean=0.0, stddev=noise_factor, dtype=tf.float32)
                for tensor in real_trajs[:4]
            ]
            
            # Also add mild noise to fake data for discriminator training
            noisy_fake = [
                tensor + tf.random.normal(tf.shape(tensor), mean=0.0, stddev=noise_factor * 0.5, dtype=tf.float32)
                for tensor in gen_trajs[:4]
            ]
            
            # Train discriminator on real data
            d_loss_real = self.discriminator.train_on_batch(
                noisy_real,
                real_labels
            )
            
            # Train discriminator on fake data
            d_loss_fake = self.discriminator.train_on_batch(
                noisy_fake,
                fake_labels
            )
            
            # Calculate the relative strength of the discriminator
            disc_strength = (2.0 - (d_loss_real + d_loss_fake))  # Higher value means stronger discriminator
            
            # Adaptive generator training based on discriminator strength
            gen_train_steps = 1
            if disc_strength > 1.0:  # If discriminator is strong (lower loss)
                gen_train_steps = 2
            if disc_strength > 1.5:  # If discriminator is very strong
                gen_train_steps = 3
                
            # Train the generator multiple times for better balance
            g_losses = []
            # Use softer target for generator to prevent model collapse
            gen_labels = tf.ones((batch_size, 1), dtype=tf.float32) * 0.85
            
            for _ in range(gen_train_steps):
                # Use fresh noise for each generator training step with slight perturbation
                z_gen = tf.random.normal([batch_size, self.latent_dim]) + tf.random.normal([batch_size, self.latent_dim], stddev=0.1)
                g_loss = self.combined.train_on_batch(
                    z_gen,
                    gen_labels
                )
                if not np.isnan(g_loss) and not np.isinf(g_loss):
                    g_losses.append(g_loss)
            
            g_loss = np.mean(g_losses) if g_losses else 1.0
            
            # Calculate utility metrics
            utility_metrics = self.compute_utility_metrics(real_trajs, gen_trajs)
            
            return {
                "d_loss_real": float(d_loss_real),
                "d_loss_fake": float(d_loss_fake),
                "g_loss": float(g_loss),
                "disc_strength": float(disc_strength),
                "utility_metrics": utility_metrics
            }
            
        except Exception as e:
            print(f"Error in train_step: {e}")
            import traceback
            traceback.print_exc()
            return {
                "d_loss_real": 1.0,
                "d_loss_fake": 1.0,
                "g_loss": 1.0,
                "disc_strength": 1.0,
                "utility_metrics": {
                    "spatial_loss": 0.0,
                    "category_loss": 0.0,
                    "day_loss": 0.0,
                    "hour_loss": 0.0,
                    "total_utility": 0.0
                }
            }
    
    def compute_utility_metrics(self, real_trajs, gen_trajs):
        """Compute utility metrics with enhanced spatial and categorical emphasis."""
        try:
            # Ensure all tensors have the same dtype (float32) before operations
            real_trajs = [tf.cast(tensor, tf.float32) for tensor in real_trajs]
            gen_trajs = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
            
            # Enhanced spatial utility (lat-lon) - now with higher weighting
            spatial_mse = tf.reduce_mean(tf.square(gen_trajs[0] - real_trajs[0]))
            
            # Add Haversine distance calculation for geographic accuracy with Earth context
            EARTH_RADIUS_KM = 6371.0  # Earth's radius in kilometers
            
            # Convert to radians for earth distance calculation
            real_coords = real_trajs[0] * tf.constant(np.pi / 180, dtype=tf.float32)
            gen_coords = gen_trajs[0] * tf.constant(np.pi / 180, dtype=tf.float32)
            
            # Extract lat/lon components
            real_lat = real_coords[:,:,0]
            real_lon = real_coords[:,:,1]
            gen_lat = gen_coords[:,:,0]
            gen_lon = gen_coords[:,:,1]
            
            # Improved haversine formula
            lat_diff = gen_lat - real_lat
            lon_diff = gen_lon - real_lon
            
            # Haversine formula components
            a = tf.sin(lat_diff/2)**2 + tf.cos(real_lat) * tf.cos(gen_lat) * tf.sin(lon_diff/2)**2
            c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1-a))
            geo_distance = EARTH_RADIUS_KM * c
            
            # Take the mean across all points
            spatial_haversine = tf.reduce_mean(geo_distance)
            
            # Combine MSE and geographic distance for spatial utility with higher weight on haversine
            spatial_loss = spatial_mse + tf.constant(0.25, dtype=tf.float32) * spatial_haversine
            
            # Add a diversity measure for spatial coordinates
            # Calculate pairwise distances between generated trajectories
            batch_size = tf.shape(gen_trajs[0])[0]
            
            # Reshape for broadcasting
            points1 = tf.reshape(gen_coords, [batch_size, -1, 2])  # [batch, seq, 2]
            points2 = tf.reshape(gen_coords, [batch_size, 1, -1, 2])  # [batch, 1, seq, 2]
            
            # Compute pairwise distances within batch along spatial dimension
            # This creates a tensor of shape [batch, batch, seq]
            pairwise_diffs = points1[:, tf.newaxis, :, :] - points2
            pairwise_distances = tf.reduce_mean(tf.sqrt(tf.reduce_sum(pairwise_diffs**2, axis=-1)), axis=-1)
            
            # Use the mean of the distances as a diversity score
            # Exclude self-comparisons (diagonal elements)
            mask = 1.0 - tf.eye(batch_size)
            masked_distances = pairwise_distances * tf.cast(mask, tf.float32)
            spatial_diversity = tf.reduce_sum(masked_distances) / (tf.cast(batch_size, tf.float32) * (tf.cast(batch_size, tf.float32) - 1.0))
            
            # Temporal utility (day and hour)
            temp_day_loss = tf.reduce_mean(tf.reduce_sum(
                real_trajs[2] * tf.math.log(tf.clip_by_value(gen_trajs[2], 1e-7, 1.0)), 
                axis=-1))
            
            temp_hour_loss = -tf.reduce_mean(tf.reduce_sum(
                real_trajs[3] * tf.math.log(tf.clip_by_value(gen_trajs[3], 1e-7, 1.0)), 
                axis=-1))
            
            # Enhanced category utility with similarity measure
            # Use both cross-entropy and JS divergence
            real_cat = tf.nn.softmax(real_trajs[1])
            gen_cat = tf.clip_by_value(gen_trajs[1], 1e-7, 1.0)
            
            # Cross-entropy component
            cat_ce = -tf.reduce_mean(tf.reduce_sum(
                real_cat * tf.math.log(gen_cat), 
                axis=-1))
            
            # Calculate JS divergence approximation
            m = 0.5 * (real_cat + gen_cat)
            js_div = 0.5 * (
                tf.reduce_mean(tf.reduce_sum(real_cat * tf.math.log(tf.clip_by_value(real_cat / (m + 1e-7), 1e-7, 1e8)), axis=-1)) +
                tf.reduce_mean(tf.reduce_sum(gen_cat * tf.math.log(tf.clip_by_value(gen_cat / (m + 1e-7), 1e-7, 1e8)), axis=-1))
            )
            
            # Calculate category diversity similar to spatial diversity
            category_diversity = self._compute_category_diversity(gen_cat)
            
            # Convert all tensors to numpy before combining
            spatial_loss_np = float(spatial_loss.numpy())
            spatial_diversity_np = float(spatial_diversity.numpy())
            cat_ce_np = float(cat_ce.numpy())
            js_div_np = float(js_div.numpy())
            category_diversity_np = float(category_diversity.numpy())
            temp_day_loss_np = float(temp_day_loss.numpy())
            temp_hour_loss_np = float(temp_hour_loss.numpy())
            
            # Combined category loss with diversity bonus
            cat_loss = cat_ce_np + 0.5 * js_div_np - 0.5 * category_diversity_np
            
            # Calculate individual components for debugging
            spatial_metric = -spatial_loss_np + spatial_diversity_np
            day_metric = -temp_day_loss_np
            hour_metric = -temp_hour_loss_np
            category_metric = -cat_loss
            
            # Combine utility components with adjusted weights to prioritize spatial and category
            # Include diversity bonuses to combat mode collapse
            # Increase the weight of spatial component for better spatial utility
            utility_metric = -(5.0 * spatial_loss_np + 2.5 * cat_loss + 1.0 * temp_day_loss_np + 0.5 * temp_hour_loss_np) + (2.0 * spatial_diversity_np + 1.5 * category_diversity_np)
            
            return {
                "spatial_loss": spatial_loss_np,
                "spatial_diversity": spatial_diversity_np,
                "category_loss": cat_loss,
                "category_diversity": category_diversity_np,
                "day_loss": temp_day_loss_np,
                "hour_loss": temp_hour_loss_np,
                "total_utility": utility_metric
            }
            
        except Exception as e:
            print(f"Error in utility metrics calculation: {e}")
            return {
                "spatial_loss": 0.0,
                "spatial_diversity": 0.0,
                "category_loss": 0.0,
                "category_diversity": 0.0,
                "day_loss": 0.0,
                "hour_loss": 0.0,
                "total_utility": 0.0
            }
            
    def _compute_category_diversity(self, category_probs):
        """Compute diversity in generated categorical distributions."""
        batch_size = tf.shape(category_probs)[0]
        
        # Flatten the category probabilities for each trajectory
        # This combines all time steps
        flat_probs = tf.reshape(category_probs, [batch_size, -1])
        
        # Compute cosine distance between pairs of trajectories
        normalized = tf.nn.l2_normalize(flat_probs, axis=1)
        similarity = tf.matmul(normalized, normalized, transpose_b=True)
        
        # Convert similarity to distance (1 - similarity)
        distance = 1.0 - similarity
        
        # Mask out self-comparisons
        mask = 1.0 - tf.eye(batch_size)
        masked_distance = distance * mask
        
        # Average distance is our diversity measure
        diversity = tf.reduce_sum(masked_distance) / (tf.cast(batch_size, tf.float32) * (tf.cast(batch_size, tf.float32) - 1.0))
        
        return diversity
    
    def train(self, epochs=200, batch_size=256, sample_interval=10):
        """Train the KAN-GAN model."""
        # Load training data
        x_train = np.load('data/final_train.npy', allow_pickle=True)
        self.x_train = x_train
        
        # Padding with consistent dtype
        X_train = [pad_sequences(f, self.max_length, padding='pre', dtype='float32') for f in x_train[:5]]
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Save initial configuration
        with open(f'results/model_config_0.json', 'w') as f:
            json.dump(self.get_config(), f)
        
        print(f"Starting training for {epochs} epochs...")
        best_utility = float('-inf')
        best_epoch = 0
        
        for epoch in range(epochs):
            # Shuffle and sample batches
            indices = np.random.permutation(X_train[0].shape[0])
            num_batches = len(indices) // batch_size
            
            epoch_metrics = []
            for batch in range(num_batches):
                batch_indices = indices[batch * batch_size:(batch + 1) * batch_size]
                batch_X = [X[batch_indices] for X in X_train]
                
                # Convert batch to float32 for consistent typing
                batch_X = [tf.cast(X, tf.float32) for X in batch_X]
                
                # Train step
                metrics = self.train_step(batch_X, batch_size)
                epoch_metrics.append(metrics)
            
            # Average metrics across batches
            avg_metrics = {
                'd_loss_real': np.mean([m['d_loss_real'] for m in epoch_metrics]),
                'd_loss_fake': np.mean([m['d_loss_fake'] for m in epoch_metrics]),
                'g_loss': np.mean([m['g_loss'] for m in epoch_metrics]),
                'utility': np.mean([m['utility_metrics']['total_utility'] for m in epoch_metrics])
            }
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"D_real: {avg_metrics['d_loss_real']:.4f}, D_fake: {avg_metrics['d_loss_fake']:.4f}, G: {avg_metrics['g_loss']:.4f}")
                print(f"Utility: {avg_metrics['utility']:.4f}")
            
            # Save checkpoints
            if epoch % sample_interval == 0 or epoch == epochs - 1:
                self.save_checkpoint(epoch + 1)
            
            # Track best model by utility
            if avg_metrics['utility'] > best_utility:
                best_utility = avg_metrics['utility']
                best_epoch = epoch
                self.save_checkpoint('best')
                print(f"New best model at epoch {epoch} with utility score: {best_utility:.4f}")
        
        print(f"\nTraining completed. Best model found at epoch {best_epoch} with utility score: {best_utility:.4f}")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoints."""
        # Make sure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Save configuration
        config_path = f'results/model_config_{epoch}.json'
        with open(config_path, 'w') as f:
            json.dump(self.get_config(), f)
        print(f"Saved model configuration to {config_path}")
        
        # Save generator weights
        generator_path = f'results/generator_{epoch}.weights.h5'
        self.generator.save_weights(generator_path)
        print(f"Saved generator weights to {generator_path}")
        
        # Save discriminator weights
        discriminator_path = f'results/discriminator_{epoch}.weights.h5'
        self.discriminator.save_weights(discriminator_path)
        print(f"Saved discriminator weights to {discriminator_path}")
        
    def nf_distribution(self):
        """Provide compatibility with normalizing flow approach."""
        class DummyDistribution:
            def __init__(self, latent_dim):
                self.latent_dim = latent_dim
                
            def sample(self, n):
                """Sample from standard normal distribution."""
                return tf.random.normal([n, self.latent_dim])
                
            def bijector(self):
                class DummyBijector:
                    def inverse(self, x):
                        """Identity function."""
                        return x
                return DummyBijector()
                
            @property
            def distribution(self):
                """Return self to mimic the API of the NF distribution."""
                return self
                
            @property
            def bijector(self):
                """Return a dummy bijector to mimic the API of the NF distribution."""
                return self.bijector()
            
        return DummyDistribution(self.latent_dim)