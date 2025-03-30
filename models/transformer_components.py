import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, LayerNormalization, MultiHeadAttention, Dropout,
    Input, Concatenate, Embedding, GlobalAveragePooling1D, Lambda
)
import numpy as np

class PositionalEncoding(Layer):
    def __init__(self, max_length, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        self.pe = self.create_positional_encoding()
        
    def create_positional_encoding(self):
        position = np.arange(self.max_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe = np.zeros((self.max_length, self.d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return tf.constant(pe, dtype=tf.float32)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        length = tf.shape(inputs)[1]
        
        # Get positional encoding for actual length
        pos_enc = self.pe[:length, :]
        
        # Broadcast to match input batch size
        pos_enc = tf.broadcast_to(pos_enc, [batch_size, length, self.d_model])
        
        return inputs + pos_enc

class TransformerEncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderLayer, self).__init__(**kwargs)
        
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn_layer1 = Dense(dff, activation='relu')
        self.ffn_layer2 = Dense(d_model)
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training, mask=None):
        attn_output = self.mha(inputs, inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn_layer1(out1)
        ffn_output = self.ffn_layer2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerDecoderLayer, self).__init__(**kwargs)
        
        self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        
        self.ffn_layer1 = Dense(dff, activation='relu')
        self.ffn_layer2 = Dense(d_model)
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)
    
    def call(self, inputs, enc_output, training, look_ahead_mask=None, padding_mask=None):
        # Self attention with look ahead mask
        attn1 = self.mha1(inputs, inputs, inputs, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)
        
        # Attention over encoder output
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        
        # Feed forward
        ffn_output = self.ffn_layer1(out2)
        ffn_output = self.ffn_layer2(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        
        return self.layernorm3(out2 + ffn_output)

class PaddingMaskLayer(Layer):
    """Layer to create padding mask compatible with Keras Functional API"""
    def __init__(self, **kwargs):
        super(PaddingMaskLayer, self).__init__(**kwargs)
        
    def call(self, seq):
        # Create mask from sequence where 1 indicates padded positions
        eq_zero = tf.math.equal(seq, 0)
        mask = tf.cast(eq_zero, tf.float32)
        # Add dimensions for attention broadcasting
        return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    
    def get_config(self):
        return super().get_config()

class LookAheadMaskLayer(Layer):
    """Layer to create look-ahead mask compatible with Keras Functional API"""
    def __init__(self, **kwargs):
        super(LookAheadMaskLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # inputs is used only to get the sequence length
        size = tf.shape(inputs)[1]
        # Create mask to prevent attention to future tokens
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (size, size)
    
    def get_config(self):
        return super().get_config()

class CombinedMaskLayer(Layer):
    """Layer to combine padding mask and look-ahead mask"""
    def __init__(self, **kwargs):
        super(CombinedMaskLayer, self).__init__(**kwargs)
        self.padding_mask_layer = PaddingMaskLayer()
        self.look_ahead_mask_layer = LookAheadMaskLayer()
    
    def call(self, inputs):
        seq = inputs
        size = tf.shape(seq)[1]
        
        pad_mask = self.padding_mask_layer(seq)
        look_ahead_mask = self.look_ahead_mask_layer(seq)
        
        combined = tf.maximum(pad_mask, look_ahead_mask)
        return combined
    
    def get_config(self):
        return super().get_config()

# For backwards compatibility, provide functional versions of the mask functions
def create_padding_mask(seq):
    """Creates padding mask in functional style (not for use in Keras models)"""
    mask_layer = PaddingMaskLayer()
    return mask_layer(seq)

def create_look_ahead_mask(size):
    """Creates look-ahead mask in functional style (not for use in Keras models)"""
    # For compatibility with existing code, but use the layer version for model building
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def combined_mask(seq, size):
    """Combines padding and look-ahead masks in functional style (not for use in Keras models)"""
    mask_layer = CombinedMaskLayer()
    return mask_layer(seq) 