import tensorflow as tf
import json
import os

class MARC:
    def __init__(self):
        self.model = None
        
    def build_model(self):
        """Build the MARC model architecture"""
        # Input layers
        input_day = tf.keras.layers.Input(shape=(144,), name='input_day')
        input_hour = tf.keras.layers.Input(shape=(144,), name='input_hour')
        input_category = tf.keras.layers.Input(shape=(144,), name='input_category')
        input_lat_lon = tf.keras.layers.Input(shape=(144, 40), name='input_lat_lon')
        
        # Embedding layers
        emb_day = tf.keras.layers.Embedding(input_dim=7, output_dim=100, input_length=144, 
                                          embeddings_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05),
                                          name='emb_day')(input_day)
        emb_hour = tf.keras.layers.Embedding(input_dim=24, output_dim=100, input_length=144,
                                           embeddings_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05),
                                           name='emb_hour')(input_hour)
        emb_category = tf.keras.layers.Embedding(input_dim=10, output_dim=100, input_length=144,
                                               embeddings_initializer=tf.keras.initializers.RandomUniform(-0.05, 0.05),
                                               name='emb_category')(input_category)
        emb_lat_lon = tf.keras.layers.Dense(100, activation='linear', name='emb_lat_lon',
                                          kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in'))(input_lat_lon)
        
        # Concatenate embeddings
        concat = tf.keras.layers.Concatenate(axis=2)([emb_day, emb_hour, emb_category, emb_lat_lon])
        
        # Dropout and LSTM
        dropout1 = tf.keras.layers.Dropout(0.5)(concat)
        lstm = tf.keras.layers.LSTM(50, recurrent_regularizer=tf.keras.regularizers.L1L2(l1=0.02, l2=0.0),
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_avg'),
                                  recurrent_initializer=tf.keras.initializers.Orthogonal())(dropout1)
        
        # Final layers
        dropout2 = tf.keras.layers.Dropout(0.5)(lstm)
        output = tf.keras.layers.Dense(193, activation='softmax', name='dense_1',
                                     kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in'))(dropout2)
        
        # Create model
        self.model = tf.keras.Model(inputs=[input_day, input_hour, input_category, input_lat_lon], outputs=output)
        
    def load_weights(self, weights_path):
        """Load pre-trained weights for the MARC model"""
        if self.model is None:
            self.build_model()
        self.model.load_weights(weights_path)
        
    def __call__(self, x):
        """Forward pass through the model"""
        if self.model is None:
            raise ValueError("Model weights have not been loaded. Call load_weights first.")
        return self.model(x) 