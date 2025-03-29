import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Input, Flatten, GlobalAveragePooling1D, Concatenate, LSTM, Dropout
)
from tensorflow.keras.models import Model
from models.transformer_components import TransformerEncoderLayer, PositionalEncoding

class CriticNetwork:
    def __init__(self, d_model=128, num_heads=4, dff=512, max_length=144):
        """
        Initialize the critic network for value function estimation.
        
        Args:
            d_model: Dimension of the model's hidden layers
            num_heads: Number of attention heads in transformer layers
            dff: Dimension of feed-forward network in transformer layers
            max_length: Maximum sequence length for trajectories
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.max_length = max_length
        self.model = self.build_critic()
    
    def build_critic(self):
        """
        Build the critic network using transformer architecture.
        
        Returns:
            Keras Model: The compiled critic model
        """
        # Input for trajectory representation
        lat_lon_input = Input(shape=(self.max_length, 2), name='critic_lat_lon_input')
        category_input = Input(shape=(self.max_length, None), name='critic_category_input')
        time_input = Input(shape=(self.max_length, 2), name='critic_time_input')  # day, hour
        mask_input = Input(shape=(self.max_length, 1), name='critic_mask_input')
        
        # Project each input to common dimension
        lat_lon_proj = Dense(self.d_model, activation='relu')(lat_lon_input)
        category_proj = Dense(self.d_model, activation='relu')(category_input)
        time_proj = Dense(self.d_model, activation='relu')(time_input)
        
        # Combine features
        combined_features = Concatenate(axis=2)([lat_lon_proj, category_proj, time_proj])
        combined_features = Dense(self.d_model, activation='relu')(combined_features)
        
        # Add positional encoding
        pos_encoding = PositionalEncoding(self.max_length, self.d_model)
        encoded_features = pos_encoding(combined_features)
        
        # Transformer encoder layers
        x = encoded_features
        for _ in range(2):  # Stack 2 encoder layers
            x = TransformerEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=0.1
            )(x, training=True)
        
        # Apply mask to zero out padded positions
        mask_broadcast = tf.repeat(mask_input, repeats=self.d_model, axis=2)
        x = x * mask_broadcast
        
        # Global pooling to get sequence-level representation
        x = GlobalAveragePooling1D()(x)
        
        # Value prediction head
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.1)(x)
        value_output = Dense(1, name='value')(x)
        
        # Build and compile model
        model = Model(
            inputs=[lat_lon_input, category_input, time_input, mask_input],
            outputs=value_output
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mse')
        
        return model
    
    def predict_value(self, states):
        """
        Predict the value of the input state.
        
        Args:
            states: A list of state components [lat_lon, category, time, mask]
            
        Returns:
            values: Predicted state values
        """
        return self.model.predict(states)
    
    def train_on_batch(self, states, returns):
        """
        Train the critic network on a batch of data.
        
        Args:
            states: A list of state components [lat_lon, category, time, mask]
            returns: Target values for the states
            
        Returns:
            loss: Training loss
        """
        return self.model.train_on_batch(states, returns)
    
    def save_weights(self, filepath):
        """Save the critic model weights to a file"""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load the critic model weights from a file"""
        self.model.load_weights(filepath) 