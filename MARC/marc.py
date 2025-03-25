import tensorflow as tf

class MARC:
    def __init__(self):
        self.model = None
        
    def load_weights(self, weights_path):
        """Load pre-trained weights for the MARC model"""
        self.model = tf.keras.models.load_model(weights_path)
        
    def __call__(self, x):
        """Forward pass through the model"""
        if self.model is None:
            raise ValueError("Model weights have not been loaded. Call load_weights first.")
        return self.model(x) 