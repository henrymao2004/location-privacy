#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import argparse
from keras.preprocessing.sequence import pad_sequences

# Import the model
from model import RL_Enhanced_Transformer_TrajGAN

def load_model(epoch=20):
    """Load the trained model from the specified epoch.
    
    Args:
        epoch: Epoch number of the saved model (default: 110)
    
    Returns:
        model: Loaded GAN model
    """
    # Model parameters
    latent_dim = 100
    max_length = 144
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon": 2, "day": 7, "hour": 24, "category": 10, "mask": 1}
    
    # Load data statistics
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
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
    
    # Initialize the model
    model = RL_Enhanced_Transformer_TrajGAN(
        latent_dim, keys, vocab_size, max_length, 
        lat_centroid, lon_centroid, scale_factor
    )
    
    # Load the saved model weights
    try:
        model.generator.load_weights(f'results/generator_{epoch}.weights.h5')
        model.discriminator.load_weights(f'results/discriminator_{epoch}.weights.h5')
        model.critic.load_weights(f'results/critic_{epoch}.weights.h5')
        print(f"Successfully loaded model weights from epoch {epoch}")
    except Exception as e:
        print(f"Error loading weights from results directory: {e}")
        try:
            model.generator.load_weights(f'training_params/G_model_{epoch}.h5')
            model.discriminator.load_weights(f'training_params/D_model_{epoch}.h5')
            model.critic.load_weights(f'training_params/C_model_{epoch}.h5')
            print(f"Successfully loaded model weights from training_params directory")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise

    return model

def prepare_data(batch_size=256, use_test_data=True, max_length=144, latent_dim=100):
    """Prepare data for trajectory generation.
    
    Args:
        batch_size: Number of trajectories to generate
        use_test_data: Whether to use test data as conditioning input
        max_length: Maximum sequence length
        latent_dim: Dimension of the latent noise vector
    
    Returns:
        data: Prepared data for trajectory generation
        traj_lengths: Actual lengths of trajectories
    """
    # Load data statistics
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
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
    
    if use_test_data:
        # Load test data directly from CSV
        print("Using test_latlon.csv for trajectory generation...")
        test_df = pd.read_csv('data/test_latlon.csv')
        
        # Print available columns for debugging
        print(f"Available columns in test_latlon.csv: {test_df.columns.tolist()}")
        
        # Get unique trajectory IDs
        unique_tids = test_df['tid'].unique()
        
        # Select a subset if needed
        if batch_size > len(unique_tids):
            print(f"Warning: Requested batch size {batch_size} exceeds test data size {len(unique_tids)}.")
            print("Using all available test data.")
            batch_size = len(unique_tids)
        
        # Select random trajectories
        selected_tids = np.random.choice(unique_tids, batch_size, replace=False)
        filtered_df = test_df[test_df['tid'].isin(selected_tids)]
        
        # Calculate trajectory lengths
        traj_lengths = filtered_df.groupby('tid').size().values.reshape(-1, 1)
        
        # Prepare data structure
        lat_lon = np.zeros((batch_size, max_length, 2))
        category = np.zeros((batch_size, max_length, 10))  # 10 categories
        day = np.zeros((batch_size, max_length, 7))        # 7 days
        hour = np.zeros((batch_size, max_length, 24))      # 24 hours
        mask = np.zeros((batch_size, max_length, 1))
        
        # Fill in data for each trajectory
        for i, tid in enumerate(selected_tids):
            # Get trajectory data and create point index if needed
            traj_data = filtered_df[filtered_df['tid'] == tid]
            
            # Add point_idx if not present
            if 'point_idx' not in traj_data.columns:
                traj_data = traj_data.reset_index(drop=True)
                traj_data['point_idx'] = range(len(traj_data))
                
            # Sort by point_idx
            traj_data = traj_data.sort_values('point_idx')
            length = len(traj_data)
            
            # Set mask (1 for valid points, 0 for padding)
            mask[i, -length:, 0] = 1
            
            # Fill latitude and longitude (scaled)
            lat_lon[i, -length:, 0] = (traj_data['lat'].values - lat_centroid) / scale_factor
            lat_lon[i, -length:, 1] = (traj_data['lon'].values - lon_centroid) / scale_factor
            
            # One-hot encode categorical variables
            for j, (_, point) in enumerate(traj_data.iterrows()):
                # Check if category exists, default to 0 if missing
                cat_val = int(point.get('category', 0)) if 'category' in point else 0
                cat_val = min(cat_val, 9)  # Ensure value is within bounds (0-9)
                category[i, -length+j, cat_val] = 1
                
                # Check if day exists, default to 0 if missing
                day_val = int(point.get('day', 0)) if 'day' in point else 0
                day_val = min(day_val, 6)  # Ensure value is within bounds (0-6)
                day[i, -length+j, day_val] = 1
                
                # Check if hour exists, default to 0 if missing
                hour_val = int(point.get('hour', 0)) if 'hour' in point else 0
                hour_val = min(hour_val, 23)  # Ensure value is within bounds (0-23)
                hour[i, -length+j, hour_val] = 1
        
        # Create generator inputs
        # Model expects inputs in the same order as the 'keys' list: ['lat_lon', 'day', 'hour', 'category', 'mask']
        X_data = [lat_lon, day, hour, category, mask]
        
        # Add random noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        X_data.append(noise)
        
        return X_data, traj_lengths
    else:
        # Generate completely new trajectories without conditioning
        vocab_size = {"lat_lon": 2, "day": 7, "hour": 24, "category": 10, "mask": 1}
        
        # Create a batch of random inputs
        lat_lon = np.zeros((batch_size, max_length, 2))
        category = np.zeros((batch_size, max_length, vocab_size["category"]))
        day = np.zeros((batch_size, max_length, vocab_size["day"]))
        hour = np.zeros((batch_size, max_length, vocab_size["hour"]))
        mask = np.zeros((batch_size, max_length, 1))
        
        # Set random values for one-hot encoded features
        traj_lengths = np.zeros((batch_size, 1))
        
        for i in range(batch_size):
            seq_len = np.random.randint(10, max_length)
            traj_lengths[i, 0] = seq_len
            
            # Create mask (1 for valid points, 0 for padding)
            mask[i, -seq_len:, 0] = 1
            
            # Random starting point
            lat_offset = np.random.uniform(-scale_factor/2, scale_factor/2)
            lon_offset = np.random.uniform(-scale_factor/2, scale_factor/2)
            
            # Set initial point
            lat_lon[i, -seq_len, 0] = lat_offset
            lat_lon[i, -seq_len, 1] = lon_offset
            
            # Set random category, day, hour for first point
            cat_idx = np.random.randint(0, vocab_size["category"])
            day_idx = np.random.randint(0, vocab_size["day"])
            hour_idx = np.random.randint(0, vocab_size["hour"])
            
            category[i, -seq_len, cat_idx] = 1
            day[i, -seq_len, day_idx] = 1
            hour[i, -seq_len, hour_idx] = 1
        
        # Create inputs for the generator
        X_gen = [lat_lon, day, hour, category, mask]
        
        # Add random noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        X_gen.append(noise)
        
        return X_gen, traj_lengths

def generate_trajectories(model, data, traj_lengths):
    """Generate synthetic trajectories using the trained model.
    
    Args:
        model: Trained GAN model
        data: Prepared data for trajectory generation
        traj_lengths: Actual lengths of trajectories
    
    Returns:
        df_synthetic: DataFrame containing synthetic trajectories
    """
    # Generate trajectories
    print("Generating synthetic trajectories...")
    
    # Print input shapes for debugging
    print("Input shapes:")
    for i, d in enumerate(data):
        print(f"Input {i}: {d.shape}")
    
    try:
        prediction = model.generator.predict(data)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Trying with a different data format...")
        # Sometimes the model expects input as separate arguments rather than a list
        try:
            prediction = model.generator.predict(data, verbose=1)
        except Exception as e:
            print(f"Second attempt failed: {e}")
            raise
    
    # Post-process the generated data
    batch_size = prediction[0].shape[0]
    max_length = prediction[0].shape[1]
    
    # Initialize lists to hold the processed trajectory data
    processed_trajs = []
    
    # Process each trajectory
    for i in range(batch_size):
        # Get actual trajectory length
        length = int(traj_lengths[i][0])
        
        # Extract and process each feature
        lat_lon = prediction[0][i, max_length-length:, :]
        category = np.argmax(prediction[1][i, max_length-length:, :], axis=1)
        day = np.argmax(prediction[2][i, max_length-length:, :], axis=1)
        hour = np.argmax(prediction[3][i, max_length-length:, :], axis=1)
        
        # Create a trajectory dataframe
        traj_df = pd.DataFrame({
            'tid': i,
            'point_idx': range(length),
            'lat': lat_lon[:, 0] + model.lat_centroid,
            'lon': lat_lon[:, 1] + model.lon_centroid,
            'day': day,
            'hour': hour,
            'category': category
        })
        
        processed_trajs.append(traj_df)
    
    # Combine all trajectories
    df_synthetic = pd.concat(processed_trajs, ignore_index=True)
    
    return df_synthetic

def main():
    """Main function to run the inference."""
    parser = argparse.ArgumentParser(description='Generate synthetic trajectories using the model from epoch 110')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Number of trajectories to generate')
    parser.add_argument('--output_file', type=str, default='results/synthetic_trajectories_epoch20.csv',
                       help='Path to save the generated trajectories')
    parser.add_argument('--use_test_data', action='store_true', default=True,
                       help='Use test data as conditioning input')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode with more verbose output')
    parser.add_argument('--epoch', type=int, default=110,
                       help='Epoch number to load model from')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Configure verbose output for debugging
    if args.debug:
        tf.debugging.set_log_device_placement(True)
        print("Debug mode enabled")
    
    # Load model
    print(f"Loading model from epoch {args.epoch}...")
    try:
        model = load_model(epoch=args.epoch)
        
        # Print model input/output shapes for debugging
        if args.debug:
            print("\nModel summary:")
            print("Generator inputs:")
            for i, inp in enumerate(model.generator.inputs):
                print(f"Input {i}: {inp.name} - Shape: {inp.shape}")
            print("\nGenerator outputs:")
            for i, out in enumerate(model.generator.outputs):
                print(f"Output {i}: {out.name} - Shape: {out.shape}")
            print("\n")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Checking data directory structure...")
        os.system("find . -name '*.h5' -o -name '*.weights.h5' | sort")
        raise
    
    # Prepare data
    print(f"Preparing data for batch size {args.batch_size}...")
    try:
        data, traj_lengths = prepare_data(
            batch_size=args.batch_size,
            use_test_data=args.use_test_data
        )
    except Exception as e:
        print(f"Error preparing data: {e}")
        print("Checking data files...")
        os.system("find data -type f | sort")
        os.system("ls -la data/final_test.npy")
        raise
    
    # Debug data shape
    if args.debug:
        print("Data shapes:")
        for i, d in enumerate(data):
            print(f"Data component {i}: {d.shape}")
        print(f"Trajectory lengths shape: {traj_lengths.shape}")
        
    # Generate trajectories
    print("Generating synthetic trajectories...")
    synthetic_trajectories = generate_trajectories(model, data, traj_lengths)
    
    # Save to file
    print(f"Saving {len(synthetic_trajectories)} trajectory points to {args.output_file}...")
    synthetic_trajectories.to_csv(args.output_file, index=False)
    
    print("Done!")
    print(f"Generated {synthetic_trajectories['tid'].nunique()} trajectories with a total of {len(synthetic_trajectories)} points.")

if __name__ == '__main__':
    main() 