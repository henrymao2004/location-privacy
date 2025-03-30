import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Input, Flatten, GlobalAveragePooling1D, Concatenate, LSTM, Dropout,
    TimeDistributed, Lambda
)
from tensorflow.keras.models import Model
from models.transformer_components import TransformerEncoderLayer, PositionalEncoding

class RepeatLayer(tf.keras.layers.Layer):
    def __init__(self, repeats, axis, **kwargs):
        super(RepeatLayer, self).__init__(**kwargs)
        self.repeats = repeats
        self.axis = axis
        
    def call(self, inputs):
        return tf.repeat(inputs, repeats=self.repeats, axis=self.axis)
    
    def get_config(self):
        config = super().get_config()
        config.update({"repeats": self.repeats, "axis": self.axis})
        return config

class CriticNetwork:
    def __init__(self, d_model=128, num_heads=4, dff=512, max_length=144, category_size=100):
        """
        Initialize the critic network for value function estimation.
        
        Args:
            d_model: Dimension of the model's hidden layers
            num_heads: Number of attention heads in transformer layers
            dff: Dimension of feed-forward network in transformer layers
            max_length: Maximum sequence length for trajectories
            category_size: Size of category vocabulary (default 100)
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.max_length = max_length
        self.category_size = category_size
        self.model = self.build_critic()
    
    def build_critic(self):
        """
        Build the critic network using transformer architecture.
        Includes dual output heads for overall value and utility-specific value.
        
        Returns:
            Keras Model: The compiled critic model
        """
        # Input for trajectory representation
        lat_lon_input = Input(shape=(self.max_length, 2), name='critic_lat_lon_input')
        # Use fixed dimension for category input
        category_input = Input(shape=(self.max_length, self.category_size), name='critic_category_input')
        time_input = Input(shape=(self.max_length, 2), name='critic_time_input')  # day, hour
        mask_input = Input(shape=(self.max_length, 1), name='critic_mask_input')
        
        # Project each input to common dimension using TimeDistributed to process each time step separately
        lat_lon_proj = TimeDistributed(Dense(self.d_model // 2, activation='relu'))(lat_lon_input)
        lat_lon_proj = TimeDistributed(Dense(self.d_model, activation='relu'))(lat_lon_proj)
        
        category_proj = TimeDistributed(Dense(self.d_model // 2, activation='relu'))(category_input)
        category_proj = TimeDistributed(Dense(self.d_model, activation='relu'))(category_proj)
        
        time_proj = TimeDistributed(Dense(self.d_model // 2, activation='relu'))(time_input)
        time_proj = TimeDistributed(Dense(self.d_model, activation='relu'))(time_proj)
        
        # Combine features
        combined_features = Concatenate(axis=2)([lat_lon_proj, category_proj, time_proj])
        combined_features = TimeDistributed(Dense(self.d_model, activation='relu'))(combined_features)
        
        # Add positional encoding
        pos_encoding = PositionalEncoding(self.max_length, self.d_model)
        encoded_features = pos_encoding(combined_features)
        
        # Transformer encoder layers with residual connections
        x = encoded_features
        for i in range(3):  # Increased from 2 to 3 layers
            # Add a residual connection if not the first layer
            layer_output = TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=0.1
            )(x, training=True)
            
            # Residual connection with layer normalization
            if i > 0:
                x = Lambda(lambda inputs: inputs[0] + inputs[1])([x, layer_output])
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
            else:
                x = layer_output
        
        # Apply mask to zero out padded positions
        # Use our custom RepeatLayer instead of direct tf.repeat
        mask_broadcast = RepeatLayer(repeats=self.d_model, axis=2)(mask_input)
        x = Lambda(lambda inputs: inputs[0] * inputs[1])([x, mask_broadcast])
        
        # Global pooling to get sequence-level representation
        x = GlobalAveragePooling1D()(x)
        
        # Common feature extraction
        common_features = Dense(128, activation='relu')(x)
        common_features = Dropout(0.1)(common_features)
        common_features = Dense(64, activation='relu')(common_features)
        common_features = Dropout(0.1)(common_features)
        
        # Split into two heads - one for overall value and one for utility value
        # Overall value head (used for regular PPO)
        value_head = Dense(32, activation='relu')(common_features)
        value_output = Dense(1, name='value')(value_head)
        
        # Utility-specific value head (helps model understand utility better)
        utility_head = Dense(32, activation='relu')(common_features)
        utility_value_output = Dense(1, name='utility_value')(utility_head)
        
        # Combine the outputs into a single model with two outputs
        model = Model(
            inputs=[lat_lon_input, category_input, time_input, mask_input],
            outputs=[value_output, utility_value_output]
        )
        
        # Custom loss weights (prioritize overall value but give utility value some weight)
        loss_weights = {'value': 1.0, 'utility_value': 0.5}
        
        # Compile model with MSE loss for both outputs
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4), 
            loss={'value': 'mse', 'utility_value': 'mse'},
            loss_weights=loss_weights
        )
        
        return model
    
    def predict_value(self, states):
        """
        Predict the value of the input state.
        
        Args:
            states: A list of state components [lat_lon, category, time, mask]
            
        Returns:
            values: Predicted state values (primary value only)
        """
        # Convert all inputs to numpy arrays if they're tensors
        states_np = []
        for state in states:
            if isinstance(state, tf.Tensor):
                state = state.numpy()
            states_np.append(state)
        
        # Get predictions from both outputs, but return only the main value
        predictions = self.model.predict(states_np, verbose=0)
        return predictions[0]  # Return main value output
    
    def predict_utility_value(self, states):
        """
        Predict the utility-specific value of the input state.
        
        Args:
            states: A list of state components [lat_lon, category, time, mask]
            
        Returns:
            utility_values: Predicted utility values
        """
        # Convert all inputs to numpy arrays if they're tensors
        states_np = []
        for state in states:
            if isinstance(state, tf.Tensor):
                state = state.numpy()
            states_np.append(state)
        
        # Get predictions from both outputs, but return only the utility value
        predictions = self.model.predict(states_np, verbose=0)
        return predictions[1]  # Return utility value output
    
    def train_on_batch(self, states, returns, utility_returns=None):
        """
        Train the critic network on a batch of data.
        
        Args:
            states: A list of state components [lat_lon, category, time, mask]
            returns: Target values for the states (overall value)
            utility_returns: Target utility values (optional, if None, uses returns)
            
        Returns:
            loss: Training loss
        """
        # Convert all inputs to numpy arrays if they're tensors
        states_np = []
        for state in states:
            if isinstance(state, tf.Tensor):
                state = state.numpy()
            states_np.append(state)
            
        # Convert returns to numpy if it's a tensor
        if isinstance(returns, tf.Tensor):
            returns = returns.numpy()
        
        # If utility_returns not provided, use returns
        if utility_returns is None:
            utility_returns = returns
        elif isinstance(utility_returns, tf.Tensor):
            utility_returns = utility_returns.numpy()
        
        # Train on batch with both outputs
        return self.model.train_on_batch(states_np, {'value': returns, 'utility_value': utility_returns})
    
    def save_weights(self, filepath):
        """Save the critic model weights to a file"""
        # Ensure filepath ends with .weights.h5
        if not filepath.endswith('.weights.h5'):
            if filepath.endswith('.h5'):
                filepath = filepath.replace('.h5', '.weights.h5')
            else:
                filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load the critic model weights from a file"""
        # Ensure filepath ends with .weights.h5
        if not filepath.endswith('.weights.h5'):
            if filepath.endswith('.h5'):
                filepath = filepath.replace('.h5', '.weights.h5')
            else:
                filepath = filepath + '.weights.h5'
        self.model.load_weights(filepath) 