import sys
import pandas as pd
import numpy as np
import torch
from model import TrajGAN
from tul_classifier import TULClassifier

# Set random seeds for reproducibility
np.random.seed(2020)
torch.manual_seed(2020)

if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    n_batch_size = int(sys.argv[2])
    n_sample_interval = int(sys.argv[3])
    
    latent_dim = 100
    max_length = 48
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {
        'lat_lon': 2,  # latitude and longitude
        'day': 7,      # days of week
        'hour': 24,    # hours in day
        'category': 9, # POI categories
        'mask': 1      # mask for variable length
    }
    
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
    scale_factor=max(max(abs(tr['lat'].max() - lat_centroid),
                         abs(te['lat'].max() - lat_centroid),
                         abs(tr['lat'].min() - lat_centroid),
                         abs(te['lat'].min() - lat_centroid),
                        ),
                     max(abs(tr['lon'].max() - lon_centroid),
                         abs(te['lon'].max() - lon_centroid),
                         abs(tr['lon'].min() - lon_centroid),
                         abs(te['lon'].min() - lon_centroid),
                        ))
    
    # Initialize TUL classifier
    print("Initializing TUL classifier...")
    num_users = len(np.unique(np.load('data/final_train.npy', allow_pickle=True)[-1]))  # Get number of unique users
    tul_classifier = TULClassifier(max_length, vocab_size, num_users)
    
    # Train TUL classifier if needed
    # This step can be skipped if you have a pre-trained classifier
    print("Training TUL classifier...")
    trajectories = np.load('data/final_train.npy', allow_pickle=True)[0]  # Assuming first element contains trajectory data
    users = np.load('data/final_train.npy', allow_pickle=True)[-1]        # Assuming last element contains user IDs
    labels = np.ones(len(users))  # Create positive samples
    tul_classifier.train(trajectories, users, labels, epochs=10)
    
    # Initialize TrajGAN with RL components
    print("Initializing TrajGAN with RL components...")
    gan = TrajGAN(
        latent_dim=latent_dim,
        keys=keys,
        vocab_size=vocab_size,
        max_length=max_length,
        lat_centroid=lat_centroid,
        lon_centroid=lon_centroid,
        scale_factor=scale_factor,
        tul_classifier=tul_classifier
    )
    
    # Training parameters
    rl_update_interval = 5  # How often to perform RL updates
    
    # Train the model
    print("Starting training...")
    gan.train(
        epochs=n_epochs,
        batch_size=n_batch_size,
        sample_interval=n_sample_interval,
        rl_update_interval=rl_update_interval
    )
    
    print("Training completed!")