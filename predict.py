import sys
import pandas as pd
import numpy as np
import torch
from model import TrajGAN
from transformer import TransformerBlock

if __name__ == '__main__':
    n_epochs = int(sys.argv[1])
    
    latent_dim = 100
    max_length = 144
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon":2,"day":7,"hour":24,"category":10,"mask":1}
    
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
    
    # Initialize model
    gan = TrajGAN(
        latent_dim=latent_dim,
        keys=keys,
        vocab_size=vocab_size,
        max_length=max_length,
        lat_centroid=lat_centroid,
        lon_centroid=lon_centroid,
        scale_factor=scale_factor
    )
    
    # Load the trained generator
    gan.generator.load_state_dict(torch.load(f'params/G_model_{n_epochs}.pt'))
    gan.generator.eval()
    
    # Test data
    x_test = np.load('data/final_test.npy', allow_pickle=True)
    
    # Prepare input data
    X_test = []
    for i in range(5):  # First 5 elements are trajectory data
        X_test.append(torch.tensor(x_test[i], dtype=torch.float32))
    
    # Add random noise
    noise = torch.randn(len(x_test[0]), latent_dim)
    X_test.append(noise)
    
    # Generate trajectories
    print("Generating synthetic trajectories...")
    with torch.no_grad():
        gen_trajs = gan.generator(X_test)
    
    # Convert generated trajectories to DataFrame format
    print("Converting to DataFrame format...")
    syn_trajs = pd.DataFrame({
        'lat': gen_trajs[0][:, :, 0].flatten().numpy(),
        'lon': gen_trajs[0][:, :, 1].flatten().numpy(),
        'day': torch.argmax(gen_trajs[1], dim=-1).flatten().numpy(),
        'hour': torch.argmax(gen_trajs[2], dim=-1).flatten().numpy(),
        'category': torch.argmax(gen_trajs[3], dim=-1).flatten().numpy()
    })
    
    # Save synthetic trajectories
    print("Saving synthetic trajectories...")
    syn_trajs.to_csv('results/syn_traj_test.csv', index=False)
    print("Synthetic trajectories saved to results/syn_traj_test.csv")
    
    
    
    
    
    
    
    
    
    
    