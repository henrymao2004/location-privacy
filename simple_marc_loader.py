#!/usr/bin/env python
# Simple script to load MARC model using improved TUL classifier

from models.tul_classifier import TULClassifier
import numpy as np

if __name__ == "__main__":
    print("Loading MARC model...")
    
    # Use the simplified factory method to load the model directly
    tul = TULClassifier.load_from_marc("MARC")
    
    print(f"Model loaded with {tul.num_users} users and sequence length {tul.max_length}")
    
    # Create dummy input data for a simple test
    batch_size = 1
    seq_len = tul.max_length
    
    # Create sample inputs for the MARC model
    # Day of week (0-6) 
    day = np.zeros((batch_size, seq_len))
    # Hour of day (0-23)
    hour = np.zeros((batch_size, seq_len))
    # Category ID (0-9)
    category = np.zeros((batch_size, seq_len))
    # Lat-lon features (expanded to 40 dimensions as required by MARC)
    latlon = np.zeros((batch_size, seq_len, 40))
    
    # Fill first 5 positions with sample data
    day[:, :5] = [1, 2, 3, 4, 5]  # Mon-Fri
    hour[:, :5] = [9, 12, 15, 18, 21]  # Different times of day
    category[:, :5] = [2, 3, 5, 7, 9]  # Different POI categories
    
    # Sample coordinates
    latlon[:, 0, :2] = [40.7128, -74.0060]  # New York
    latlon[:, 1, :2] = [34.0522, -118.2437]  # Los Angeles
    latlon[:, 2, :2] = [41.8781, -87.6298]  # Chicago
    latlon[:, 3, :2] = [29.7604, -95.3698]  # Houston
    latlon[:, 4, :2] = [39.9526, -75.1652]  # Philadelphia
    
    # Run prediction
    print("Running inference on sample trajectory...")
    predictions = tul.predict([day, hour, category, latlon])
    
    # Check results
    top_users = np.argsort(predictions[0])[-5:][::-1]  # Top 5 users
    top_probs = np.sort(predictions[0])[-5:][::-1]  # Corresponding probabilities
    
    print("\nTop 5 predicted users:")
    for i, (user_id, prob) in enumerate(zip(top_users, top_probs)):
        print(f"  {i+1}. User {user_id}: {prob*100:.2f}%")
    
    print("\nModel successfully loaded and tested") 