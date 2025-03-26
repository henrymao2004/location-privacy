import sys
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import argparse
from keras.preprocessing.sequence import pad_sequences

# Import the model
from model import RL_Enhanced_Transformer_TrajGAN

def load_model_and_data(model_epoch, batch_size=256, use_test_data=True):
    """Load the trained model and prepare data for inference.
    
    Args:
        model_epoch: Epoch number for the saved model weights
        batch_size: Number of trajectories to generate
        use_test_data: Whether to use test data as conditioning input
    
    Returns:
        model: Loaded GAN model
        data: Prepared data for trajectory generation
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
        model.generator.load_weights(f'results/generator_{model_epoch}.weights.h5')
        print(f"Successfully loaded generator weights from epoch {model_epoch}")
    except:
        print(f"Could not load weights from results/generator_{model_epoch}.weights.h5")
        print("Trying alternative weight paths...")
        try:
            model.generator.load_weights(f'training_params/G_model_{model_epoch}.h5')
            print(f"Successfully loaded generator weights from training_params/G_model_{model_epoch}.h5")
        except Exception as e:
            print(f"Error loading generator weights: {e}")
            raise
    
    # Prepare data for inference
    if use_test_data:
        # Load test data for conditioning the generator
        x_test = np.load('data/final_test.npy', allow_pickle=True)
        
        # Select a subset if needed
        if batch_size > len(x_test[0]):
            print(f"Warning: Requested batch size {batch_size} exceeds test data size {len(x_test[0])}.")
            print("Using all available test data.")
            batch_size = len(x_test[0])
        
        # Select a random subset of the test data
        indices = np.random.choice(len(x_test[0]), batch_size, replace=False)
        
        # Prepare the data
        x_test_subset = [x[indices] for x in x_test]
        X_test = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in x_test_subset]
        
        # Add random noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        X_test.append(noise)
        
        # Get trajectory lengths for post-processing
        traj_lengths = x_test[6][indices]
        
        return model, X_test, traj_lengths
    else:
        # Generate completely new trajectories without conditioning
        # In this case, we need to create random inputs for all features
        
        # Create a batch of random inputs
        lat_lon = np.zeros((batch_size, max_length, 2))
        category = np.zeros((batch_size, max_length, vocab_size["category"]))
        day = np.zeros((batch_size, max_length, vocab_size["day"]))
        hour = np.zeros((batch_size, max_length, vocab_size["hour"]))
        
        # Set random values for one-hot encoded features
        for i in range(batch_size):
            seq_len = np.random.randint(10, max_length)
            
            # Create mask (1 for valid points, 0 for padding)
            mask = np.zeros((batch_size, max_length, 1))
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
        X_gen = [lat_lon, category, day, hour, mask]
        
        # Add random noise
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        X_gen.append(noise)
        
        # Create dummy trajectory lengths
        traj_lengths = np.ones((batch_size, 1)) * 20  # Default 20 points per trajectory
        
        return model, X_gen, traj_lengths

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
    prediction = model.generator.predict(data)
    
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

def evaluate_trajectories(real_df, synthetic_df):
    """Evaluate the quality of generated trajectories.
    
    Args:
        real_df: DataFrame containing real trajectories
        synthetic_df: DataFrame containing synthetic trajectories
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Basic statistics
    real_traj_count = real_df['tid'].nunique()
    syn_traj_count = synthetic_df['tid'].nunique()
    
    avg_real_points = real_df.groupby('tid').size().mean()
    avg_syn_points = synthetic_df.groupby('tid').size().mean()
    
    # Category distribution
    real_cat_dist = real_df['category'].value_counts(normalize=True)
    syn_cat_dist = synthetic_df['category'].value_counts(normalize=True)
    
    # Day distribution
    real_day_dist = real_df['day'].value_counts(normalize=True)
    syn_day_dist = synthetic_df['day'].value_counts(normalize=True)
    
    # Hour distribution
    real_hour_dist = real_df['hour'].value_counts(normalize=True)
    syn_hour_dist = synthetic_df['hour'].value_counts(normalize=True)
    
    # Compute Jensen-Shannon divergence for distributions
    def js_divergence(p, q):
        # Ensure all categories are present in both distributions
        all_cats = set(p.index).union(set(q.index))
        p_full = pd.Series([p.get(c, 0) for c in all_cats], index=all_cats)
        q_full = pd.Series([q.get(c, 0) for c in all_cats], index=all_cats)
        
        # Avoid zero probabilities
        p_full = p_full + 1e-10
        q_full = q_full + 1e-10
        
        # Normalize
        p_full = p_full / p_full.sum()
        q_full = q_full / q_full.sum()
        
        # M is the average distribution
        m = 0.5 * (p_full + q_full)
        
        # KL divergence
        kl_pm = np.sum(p_full * np.log(p_full / m))
        kl_qm = np.sum(q_full * np.log(q_full / m))
        
        # JS divergence
        js = 0.5 * (kl_pm + kl_qm)
        return js
    
    js_category = js_divergence(real_cat_dist, syn_cat_dist)
    js_day = js_divergence(real_day_dist, syn_day_dist)
    js_hour = js_divergence(real_hour_dist, syn_hour_dist)
    
    # Collect metrics
    metrics = {
        'real_trajectories': real_traj_count,
        'synthetic_trajectories': syn_traj_count,
        'avg_real_points': avg_real_points,
        'avg_synthetic_points': avg_syn_points,
        'js_category': js_category,
        'js_day': js_day,
        'js_hour': js_hour
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic trajectories using trained GAN model')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number of the trained model')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of trajectories to generate')
    parser.add_argument('--output_file', type=str, default='inference/synthetic_trajectories.csv',
                       help='Path to save synthetic trajectories')
    parser.add_argument('--use_test', action='store_true', default=True,
                       help='Use test data as conditioning input for generation')
    parser.add_argument('--evaluate', action='store_true', default=False,
                       help='Evaluate the generated trajectories against real data')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Load model and data
    model, data, traj_lengths = load_model_and_data(
        args.epoch, 
        batch_size=args.batch_size,
        use_test_data=args.use_test
    )
    
    # Generate synthetic trajectories
    synthetic_df = generate_trajectories(model, data, traj_lengths)
    
    # Save generated trajectories
    synthetic_df.to_csv(args.output_file, index=False)
    print(f"Generated {len(synthetic_df['tid'].unique())} synthetic trajectories")
    print(f"Saved to {args.output_file}")
    
    # Evaluate if requested
    if args.evaluate:
        try:
            # Load real test data
            real_df = pd.read_csv('data/test_latlon.csv')
            
            # Evaluate
            metrics = evaluate_trajectories(real_df, synthetic_df)
            
            # Print evaluation results
            print("\nEvaluation Metrics:")
            print("-" * 50)
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_file = args.output_file.replace('.csv', '_metrics.csv')
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Evaluation metrics saved to {metrics_file}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
    
if __name__ == '__main__':
    main() 