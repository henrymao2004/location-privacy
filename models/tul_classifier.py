import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Bidirectional, 
    Dropout, GlobalAveragePooling1D, Concatenate
)

class TULClassifier:
    """
    Trajectory-User Linking classifier for evaluating privacy.
    This class provides an interface to a model that attempts to link
    trajectories to users, which is used for privacy evaluation.
    """
    
    def __init__(self, num_users, max_length=144, feature_dim=64):
        """
        Initialize the TUL classifier.
        
        Args:
            num_users: Number of users to classify
            max_length: Maximum trajectory length
            feature_dim: Dimension of feature embeddings
        """
        self.num_users = num_users
        self.max_length = max_length
        self.feature_dim = feature_dim
        self.model = self.build_model()
    
    def build_model(self):
        """
        Build the TUL classification model.
        
        Returns:
            model: Keras model for trajectory-user linking
        """
        # Input layers
        latlon_input = Input(shape=(self.max_length, 2), name='latlon_input')
        category_input = Input(shape=(self.max_length, None), name='category_input')
        time_input = Input(shape=(self.max_length, 2), name='time_input')  # day, hour
        mask_input = Input(shape=(self.max_length, 1), name='mask_input')
        
        # Feature extraction
        latlon_features = Dense(self.feature_dim, activation='relu')(latlon_input)
        category_features = Dense(self.feature_dim, activation='relu')(category_input)
        time_features = Dense(self.feature_dim, activation='relu')(time_input)
        
        # Combine features
        combined = Concatenate(axis=2)([latlon_features, category_features, time_features])
        
        # Apply mask
        mask_expanded = tf.repeat(mask_input, repeats=self.feature_dim*3, axis=2)
        masked_features = combined * mask_expanded
        
        # Bidirectional LSTM for sequence processing
        lstm = Bidirectional(LSTM(128, return_sequences=True))(masked_features)
        
        # Apply mask again after LSTM
        mask_expanded_lstm = tf.repeat(mask_input, repeats=256, axis=2)  # 2*128 due to bidirectional
        masked_lstm = lstm * mask_expanded_lstm
        
        # Global pooling
        pooled = GlobalAveragePooling1D()(masked_lstm)
        
        # Fully connected layers
        x = Dense(256, activation='relu')(pooled)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer for user classification
        output = Dense(self.num_users, activation='softmax', name='user_output')(x)
        
        # Build and compile model
        model = Model(
            inputs=[latlon_input, category_input, time_input, mask_input],
            outputs=output
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, user_ids, epochs=20, batch_size=32, validation_split=0.2):
        """
        Train the TUL classifier.
        
        Args:
            X_train: List of trajectory components [latlon, category, time, mask]
            user_ids: Array of user IDs corresponding to each trajectory
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            history: Training history
        """
        return self.model.fit(
            X_train,
            user_ids,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
    
    def predict(self, X):
        """
        Predict user probabilities for trajectories.
        
        Args:
            X: List of trajectory components [latlon, category, time, mask]
            
        Returns:
            probs: Probabilities for each user class
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, user_ids):
        """
        Evaluate the TUL classifier.
        
        Args:
            X_test: List of trajectory components [latlon, category, time, mask]
            user_ids: Array of user IDs corresponding to each trajectory
            
        Returns:
            metrics: Evaluation metrics [loss, accuracy]
        """
        return self.model.evaluate(X_test, user_ids)
    
    def save(self, filepath):
        """Save the TUL model"""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load the TUL model"""
        self.model = load_model(filepath)
    
    def compute_acc_at_k(self, X_test, user_ids, k=1):
        """
        Compute accuracy@k metric for TUL evaluation.
        
        Args:
            X_test: List of trajectory components [latlon, category, time, mask]
            user_ids: Array of true user IDs
            k: k value for accuracy@k
            
        Returns:
            acc_at_k: Accuracy@k metric value
        """
        # Get predictions
        predictions = self.predict(X_test)
        
        # Get top k predicted classes for each sample
        top_k_classes = tf.math.top_k(predictions, k=k).indices.numpy()
        
        # Check if true class is in top k
        correct = 0
        for i, user_id in enumerate(user_ids):
            if user_id in top_k_classes[i]:
                correct += 1
        
        # Calculate accuracy@k
        acc_at_k = correct / len(user_ids)
        
        return acc_at_k 