import numpy as np
import torch
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
        encoded_traj['day'] = np.eye(7)[traj['day']]
        
        # Encode hour (0-23)
        encoded_traj['hour'] = np.eye(24)[traj['hour']]
        
        # Encode category
        encoded_traj['category'] = np.eye(10)[traj['category']]
        
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

def preprocess_data(data_path, max_length, lat_centroid, lon_centroid, scale_factor):
    """Preprocess trajectory data"""
    # Load raw data
    raw_data = load_raw_data(data_path)
    
    # Encode trajectories
    encoded_data = encode_categorical_features(raw_data, max_length)
    
    # Split data
    train_data, temp_data = train_test_split(encoded_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    return train_data, val_data, test_data 