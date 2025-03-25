import numpy as np
import torch
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data.csv2npy import data_conversion

def load_raw_data(data_path):
    """Load raw trajectory data from CSV"""
    return pd.read_csv(data_path)

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
    train_df = pd.read_csv(os.path.join(os.path.dirname(data_path), 'train_latlon.csv'))
    test_df = pd.read_csv(os.path.join(os.path.dirname(data_path), 'test_latlon.csv'))
    
    # Convert to one-hot encoded numpy arrays
    train_data = data_conversion(train_df, 'tid', max_length)
    test_data = data_conversion(test_df, 'tid', max_length)
    
    # Split train into train and validation
    train_indices = np.arange(len(train_data[0]))
    train_idx, val_idx = train_test_split(train_indices, test_size=0.15, random_state=42)
    
    # Create final datasets
    train_final = [{
        'lat_lon': train_data[0][i],
        'day': train_data[1][i],
        'hour': train_data[2][i],
        'category': train_data[3][i],
        'mask': train_data[4][i]
    } for i in train_idx]
    
    val_final = [{
        'lat_lon': train_data[0][i],
        'day': train_data[1][i],
        'hour': train_data[2][i],
        'category': train_data[3][i],
        'mask': train_data[4][i]
    } for i in val_idx]
    
    test_final = [{
        'lat_lon': test_data[0][i],
        'day': test_data[1][i],
        'hour': test_data[2][i],
        'category': test_data[3][i],
        'mask': test_data[4][i]
    } for i in range(len(test_data[0]))]
    
    # Normalize coordinates
    for dataset in [train_final, val_final, test_final]:
        for traj in dataset:
            traj['lat_lon'][:, 0] = (traj['lat_lon'][:, 0] - lat_centroid) / scale_factor
            traj['lat_lon'][:, 1] = (traj['lat_lon'][:, 1] - lon_centroid) / scale_factor
    
    return train_final, val_final, test_final 