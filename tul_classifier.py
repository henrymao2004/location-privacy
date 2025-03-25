import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense, LSTM, Concatenate, Dropout
from keras.models import Model
from keras.optimizers import Adam

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
        
        # LSTM layers for trajectory encoding
        lstm1 = LSTM(128, return_sequences=True)(traj_input)
        lstm2 = LSTM(64)(lstm1)
        
        # Dense layers for trajectory features
        dense1 = Dense(128, activation='relu')(lstm2)
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