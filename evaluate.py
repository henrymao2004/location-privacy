import pandas as pd
import numpy as np
import torch
from model import TrajGAN
from tul_classifier import TULClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import seaborn as sns

def load_data(train_file, test_file):
    """Load and preprocess training and test data"""
    tr = pd.read_csv(train_file)
    te = pd.read_csv(test_file)
    
    # Calculate centroids and scale factor
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
    scale_factor = max(
        max(abs(tr['lat'].max() - lat_centroid),
            abs(te['lat'].max() - lat_centroid),
            abs(tr['lat'].min() - lat_centroid),
            abs(te['lat'].min() - lat_centroid)),
        max(abs(tr['lon'].max() - lon_centroid),
            abs(te['lon'].max() - lon_centroid),
            abs(tr['lon'].min() - lon_centroid),
            abs(te['lon'].min() - lon_centroid))
    )
    
    return tr, te, lat_centroid, lon_centroid, scale_factor

def evaluate_spatial_metrics(real_trajs, syn_trajs):
    """Evaluate spatial metrics between real and synthetic trajectories"""
    # Calculate MSE and MAE for latitude and longitude
    lat_mse = mean_squared_error(real_trajs['lat'], syn_trajs['lat'])
    lon_mse = mean_squared_error(real_trajs['lon'], syn_trajs['lon'])
    lat_mae = mean_absolute_error(real_trajs['lat'], syn_trajs['lat'])
    lon_mae = mean_absolute_error(real_trajs['lon'], syn_trajs['lon'])
    
    return {
        'lat_mse': lat_mse,
        'lon_mse': lon_mse,
        'lat_mae': lat_mae,
        'lon_mae': lon_mae
    }

def evaluate_temporal_metrics(real_trajs, syn_trajs):
    """Evaluate temporal metrics between real and synthetic trajectories"""
    # Calculate distribution differences for day and hour
    day_kl = np.mean(np.abs(real_trajs['day'].value_counts(normalize=True) - 
                           syn_trajs['day'].value_counts(normalize=True)))
    hour_kl = np.mean(np.abs(real_trajs['hour'].value_counts(normalize=True) - 
                            syn_trajs['hour'].value_counts(normalize=True)))
    
    return {
        'day_dist_diff': day_kl,
        'hour_dist_diff': hour_kl
    }

def evaluate_category_metrics(real_trajs, syn_trajs):
    """Evaluate category distribution metrics"""
    # Calculate distribution differences for POI categories
    cat_kl = np.mean(np.abs(real_trajs['category'].value_counts(normalize=True) - 
                           syn_trajs['category'].value_counts(normalize=True)))
    
    return {
        'category_dist_diff': cat_kl
    }

def plot_distributions(real_trajs, syn_trajs, save_dir='results/plots'):
    """Plot distributions of various attributes"""
    # Create plots directory if it doesn't exist
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot temporal distributions
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(data=real_trajs, x='day', alpha=0.5, label='Real')
    sns.histplot(data=syn_trajs, x='day', alpha=0.5, label='Synthetic')
    plt.title('Day Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.histplot(data=real_trajs, x='hour', alpha=0.5, label='Real')
    sns.histplot(data=syn_trajs, x='hour', alpha=0.5, label='Synthetic')
    plt.title('Hour Distribution')
    plt.legend()
    plt.savefig(f'{save_dir}/temporal_distributions.png')
    plt.close()
    
    # Plot category distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data=real_trajs, x='category', alpha=0.5, label='Real')
    sns.histplot(data=syn_trajs, x='category', alpha=0.5, label='Synthetic')
    plt.title('POI Category Distribution')
    plt.legend()
    plt.savefig(f'{save_dir}/category_distribution.png')
    plt.close()
    
    # Plot spatial distribution
    plt.figure(figsize=(10, 8))
    plt.scatter(real_trajs['lon'], real_trajs['lat'], alpha=0.5, label='Real', s=1)
    plt.scatter(syn_trajs['lon'], syn_trajs['lat'], alpha=0.5, label='Synthetic', s=1)
    plt.title('Spatial Distribution')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.savefig(f'{save_dir}/spatial_distribution.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    tr, te, lat_centroid, lon_centroid, scale_factor = load_data(
        'data/train_latlon.csv',
        'data/test_latlon.csv'
    )
    
    # Load synthetic trajectories
    print("Loading synthetic trajectories...")
    syn_trajs = pd.read_csv('results/syn_traj_test.csv')
    
    # Evaluate metrics
    print("Evaluating metrics...")
    spatial_metrics = evaluate_spatial_metrics(te, syn_trajs)
    temporal_metrics = evaluate_temporal_metrics(te, syn_trajs)
    category_metrics = evaluate_category_metrics(te, syn_trajs)
    
    # Print results
    print("\nEvaluation Results:")
    print("Spatial Metrics:")
    for metric, value in spatial_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nTemporal Metrics:")
    for metric, value in temporal_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    print("\nCategory Metrics:")
    for metric, value in category_metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_distributions(te, syn_trajs)
    print("Plots saved to results/plots/")

if __name__ == '__main__':
    main() 