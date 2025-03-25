import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

def load_raw_data(data_path):
    """Load raw trajectory data"""
    return np.load(data_path, allow_pickle=True)

def normalize_coordinates(trajectories, lat_centroid, lon_centroid, scale_factor):
    """Normalize latitude and longitude coordinates"""
    normalized = []
    for traj in trajectories:
        normalized_traj = traj.copy()
        normalized_traj['lat'] = (traj['lat'] - lat_centroid) / scale_factor
        normalized_traj['lon'] = (traj['lon'] - lon_centroid) / scale_factor
        normalized.append(normalized_traj)
    return normalized

def encode_categorical_features(trajectories, max_length):
    """One-hot encode categorical features"""
    encoded = []
    for traj in trajectories:
        encoded_traj = {}
        
        # Encode coordinates
        encoded_traj['lat_lon'] = np.column_stack([traj['lat'], traj['lon']])
        
        # Encode day (0-6)
        encoded_traj['day'] = tf.keras.utils.to_categorical(traj['day'], num_classes=7)
        
        # Encode hour (0-23)
        encoded_traj['hour'] = tf.keras.utils.to_categorical(traj['hour'], num_classes=24)
        
        # Encode category
        encoded_traj['category'] = tf.keras.utils.to_categorical(traj['category'], num_classes=10)
        
        # Create mask
        encoded_traj['mask'] = np.ones((len(traj['lat']), 1))
        
        # Pad sequences if necessary
        for key in encoded_traj:
            if len(encoded_traj[key]) < max_length:
                padding = np.zeros((max_length - len(encoded_traj[key]), encoded_traj[key].shape[-1]))
                encoded_traj[key] = np.vstack([encoded_traj[key], padding])
            elif len(encoded_traj[key]) > max_length:
                encoded_traj[key] = encoded_traj[key][:max_length]
        
        encoded.append(encoded_traj)
    return encoded

def split_data(data, train_ratio, val_ratio, test_ratio, random_state=42):
    """Split data into train, validation, and test sets"""
    # First split: train and temp
    train_data, temp_data = train_test_split(
        data, 
        test_size=1-train_ratio,
        random_state=random_state
    )
    
    # Second split: validation and test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=1-val_size,
        random_state=random_state
    )
    
    return train_data, val_data, test_data

def preprocess_data(data_path, max_length, lat_centroid, lon_centroid, scale_factor,
                   train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Complete data preprocessing pipeline"""
    # Load raw data
    raw_data = load_raw_data(data_path)
    
    # Normalize coordinates
    normalized_data = normalize_coordinates(raw_data, lat_centroid, lon_centroid, scale_factor)
    
    # Encode categorical features
    encoded_data = encode_categorical_features(normalized_data, max_length)
    
    # Split data
    train_data, val_data, test_data = split_data(
        encoded_data,
        train_ratio,
        val_ratio,
        test_ratio
    )
    
    return train_data, val_data, test_data 