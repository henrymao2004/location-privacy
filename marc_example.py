#!/usr/bin/env python
# Example of loading and using the MARC model with the improved TUL classifier

from models.tul_classifier import TULClassifier
import numpy as np

def main():
    # Method 1: Load via load() method
    print("\n=== Method 1: Using load() method ===")
    tul1 = TULClassifier(num_users=193, category_size=10)
    tul1.load("MARC/MARC_Weight.h5")  # Should automatically find MARC.json
    
    # Method 2: Load via load_marc_model() method
    print("\n=== Method 2: Using load_marc_model() method ===")
    tul2 = TULClassifier(num_users=193, category_size=10)
    tul2.load_marc_model("MARC/MARC_Weight.h5", "MARC/MARC.json")
    
    # Method 3: Load via factory method (simplest)
    print("\n=== Method 3: Using load_from_marc() factory method ===")
    tul3 = TULClassifier.load_from_marc("MARC")
    
    # Verify models were loaded correctly
    check_model(tul1, "Model 1")
    check_model(tul2, "Model 2") 
    check_model(tul3, "Model 3")

def check_model(model, name):
    """Check if the model was loaded correctly by inspecting its structure"""
    if model and model.model:
        print(f"\n{name} details:")
        print(f"  Number of users: {model.num_users}")
        print(f"  Max sequence length: {model.max_length}")
        print(f"  Category size: {model.category_size}")
        print(f"  Model input names: {model.model.input_names}")
        print(f"  Model has {len(model.model.layers)} layers")
        
        # Try a simple prediction
        try:
            # Create dummy input data for MARC model
            batch_size = 1
            seq_len = model.max_length
            
            # Day of week (0-6)
            day = np.zeros((batch_size, seq_len))
            # Hour of day (0-23)
            hour = np.zeros((batch_size, seq_len))
            # Category (0-9)
            category = np.zeros((batch_size, seq_len))
            # Lat-lon features (expanded to 40 dimensions)
            latlon = np.zeros((batch_size, seq_len, 40))
            
            # Add some random values
            day[:, :5] = np.random.randint(0, 7, (batch_size, 5))
            hour[:, :5] = np.random.randint(0, 24, (batch_size, 5))
            category[:, :5] = np.random.randint(0, 10, (batch_size, 5))
            latlon[:, :5, :2] = np.random.random((batch_size, 5, 2))
            
            # Try to predict
            prediction = model.predict([day, hour, category, latlon])
            print(f"  Prediction shape: {prediction.shape}")
            print(f"  Top predicted user: {np.argmax(prediction[0])}")
            print(f"  Prediction successful: True")
        except Exception as e:
            print(f"  Prediction failed: {e}")
    else:
        print(f"\n{name} not loaded correctly")

if __name__ == "__main__":
    main() 