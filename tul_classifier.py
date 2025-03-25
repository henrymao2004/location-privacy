import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense, LSTM, Concatenate, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TULClassifier:
    def __init__(self, max_length, vocab_size, num_users):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.num_users = num_users
        self.model = self.build_model()
        
    def build_model(self):
        """Build MARC-based TUL classifier"""
        # Trajectory input
        traj_input = Input(shape=(self.max_length, sum(self.vocab_size.values())))
        
        # Transformer layers for trajectory encoding
        transformer_block1 = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.1)
        transformer_block2 = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.1)
        
        x = transformer_block1(traj_input)
        x = transformer_block2(x)
        
        # Global average pooling to reduce sequence dimension
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers for trajectory features
        dense1 = Dense(128, activation='relu')(x)
        dropout1 = Dropout(0.3)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        
        # User embedding
        user_input = Input(shape=(1,))
        user_embedding = Dense(32, activation='relu')(user_input)
        
        # Combine trajectory and user features
        combined = Concatenate()([dense2, user_embedding])
        
        # Output layer
        output = Dense(1, activation='sigmoid')(combined)
        
        model = Model(inputs=[traj_input, user_input], outputs=output)
        model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
        
        return model
    
    def train(self, trajectories, users, labels, epochs=50, batch_size=32, validation_split=0.2):
        """Train the TUL classifier"""
        history = self.model.fit(
            [trajectories, users],
            labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        return history
    
    def predict(self, trajectory, user):
        """Predict linkage probability between trajectory and user"""
        return self.model.predict([trajectory, user]) 