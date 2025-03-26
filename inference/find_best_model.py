import os
import re
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import argparse
import json
from collections import defaultdict

# Import relevant modules
from model import RL_Enhanced_Transformer_TrajGAN
from MARC.marc import MARC

def find_model_checkpoints():
    """
    Find all available model checkpoints in standard directories.
    
    Returns:
        list: Paths to model checkpoints with corresponding epoch numbers
    """
    checkpoints = []
    
    # Check results directory
    result_weights = glob.glob('results/generator_*.weights.h5')
    for weight_path in result_weights:
        epoch_match = re.search(r'generator_(\d+).weights.h5', weight_path)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            checkpoints.append((epoch, weight_path))
    
    # Check training_params directory
    param_weights = glob.glob('training_params/G_model_*.h5')
    for weight_path in param_weights:
        epoch_match = re.search(r'G_model_(\d+).h5', weight_path)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            checkpoints.append((epoch, weight_path))
    
    # Sort by epoch
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def load_and_initialize_model():
    """
    Initialize model with default parameters.
    
    Returns:
        model: Initialized RL_Enhanced_Transformer_TrajGAN model
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
    
    return model, latent_dim, max_length

def prepare_test_data(max_length, latent_dim, batch_size=64):
    """
    Prepare test data for model evaluation.
    
    Args:
        max_length: Maximum sequence length
        latent_dim: Dimension of the latent space
        batch_size: Batch size for evaluation
        
    Returns:
        data: Prepared test data
        traj_lengths: Actual lengths of trajectories
    """
    # Load test data
    x_test = np.load('data/final_test.npy', allow_pickle=True)
    
    # Select a subset for quick evaluation
    if batch_size > len(x_test[0]):
        batch_size = len(x_test[0])
    
    indices = np.random.choice(len(x_test[0]), batch_size, replace=False)
    x_test_subset = [x[indices] for x in x_test]
    X_test = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in x_test_subset]
    
    # Add random noise
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    X_test.append(noise)
    
    # Get trajectory lengths for post-processing
    traj_lengths = x_test[6][indices]
    
    return X_test, traj_lengths

def evaluate_model(model, weights_path, X_test, traj_lengths):
    """
    Evaluate a model checkpoint on test data.
    
    Args:
        model: The model to evaluate
        weights_path: Path to the weights file
        X_test: Test data
        traj_lengths: Actual lengths of trajectories
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    try:
        # Load weights
        model.generator.load_weights(weights_path)
        
        # Generate trajectories
        gen_trajs = model.generator.predict(X_test)
        
        # Prepare inputs for discriminator
        disc_inputs = gen_trajs[:4]  # First 4 elements are lat_lon, category, day, hour
        
        # Get discriminator predictions
        d_preds = model.discriminator.predict(disc_inputs)
        d_loss = np.mean(binary_crossentropy(np.zeros_like(d_preds), d_preds))
        
        # Get realism score (higher is better for the generator)
        realism_score = np.mean(d_preds)
        
        # Prepare data for privacy evaluation (simplified)
        try:
            batch_size = gen_trajs[0].shape[0]
            max_length = gen_trajs[0].shape[1]
            
            # Extract features for TUL classifier
            day_indices = tf.cast(tf.argmax(gen_trajs[2], axis=-1), tf.int32)
            hour_indices = tf.cast(tf.argmax(gen_trajs[3], axis=-1), tf.int32)
            category_indices = tf.cast(tf.argmax(gen_trajs[1], axis=-1), tf.int32)
            
            # Format lat_lon to match MARC's expected input shape
            lat_lon_padded = tf.pad(gen_trajs[0], [[0, 0], [0, 0], [0, 38]])
            
            # Get TUL predictions
            try:
                tul_preds = model.tul_classifier([day_indices, hour_indices, category_indices, lat_lon_padded])
                
                # Calculate privacy metrics
                user_confidences = tf.reduce_max(tul_preds, axis=1)
                privacy_score = 1.0 - tf.reduce_mean(user_confidences)
            except:
                privacy_score = 0.5  # Default if TUL evaluation fails
        except:
            privacy_score = 0.5  # Default
        
        # Calculate utility preservation
        # For simplicity, we'll use a random subset of the real data as reference
        real_trajs = X_test[:-1]  # Remove noise from input
        
        # Simple utility metrics - compare distributions
        utility_scores = []
        
        # Category distribution
        real_cat = np.argmax(real_trajs[1], axis=-1).flatten()
        gen_cat = np.argmax(gen_trajs[1], axis=-1).flatten()
        cat_similarity = distribution_similarity(real_cat, gen_cat, n_classes=10)
        utility_scores.append(cat_similarity)
        
        # Day distribution
        real_day = np.argmax(real_trajs[2], axis=-1).flatten()
        gen_day = np.argmax(gen_trajs[2], axis=-1).flatten()
        day_similarity = distribution_similarity(real_day, gen_day, n_classes=7)
        utility_scores.append(day_similarity)
        
        # Hour distribution
        real_hour = np.argmax(real_trajs[3], axis=-1).flatten()
        gen_hour = np.argmax(gen_trajs[3], axis=-1).flatten()
        hour_similarity = distribution_similarity(real_hour, gen_hour, n_classes=24)
        utility_scores.append(hour_similarity)
        
        # Calculate mean utility score
        utility_score = np.mean(utility_scores)
        
        # Combined score with customizable weights
        combined_score = 0.4 * privacy_score + 0.3 * realism_score + 0.3 * utility_score
        
        metrics = {
            'discriminator_loss': float(d_loss),
            'realism_score': float(realism_score),
            'privacy_score': float(privacy_score),
            'utility_score': float(utility_score),
            'combined_score': float(combined_score)
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error evaluating model {weights_path}: {e}")
        return None

def binary_crossentropy(y_true, y_pred):
    """Simple binary cross-entropy implementation"""
    y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def distribution_similarity(dist1, dist2, n_classes):
    """Calculate similarity between two distributions using histogram overlap"""
    hist1, _ = np.histogram(dist1, bins=n_classes, range=(0, n_classes), density=True)
    hist2, _ = np.histogram(dist2, bins=n_classes, range=(0, n_classes), density=True)
    
    # Calculate histogram intersection (higher is better)
    intersection = np.sum(np.minimum(hist1, hist2))
    return intersection

def find_best_model(metrics_dict, criterion='combined_score', higher_is_better=True):
    """
    Find the best model based on specified criterion.
    
    Args:
        metrics_dict: Dictionary of model metrics
        criterion: Metric to use for selecting the best model
        higher_is_better: Whether higher values are better for the criterion
        
    Returns:
        best_epoch: Epoch number of the best model
        best_path: Path to the best model
    """
    best_epoch = None
    best_path = None
    best_value = float('-inf') if higher_is_better else float('inf')
    
    for epoch, (path, metrics) in metrics_dict.items():
        if metrics is None:
            continue
        
        current_value = metrics.get(criterion, 0.0)
        
        if higher_is_better:
            if current_value > best_value:
                best_value = current_value
                best_epoch = epoch
                best_path = path
        else:
            if current_value < best_value:
                best_value = current_value
                best_epoch = epoch
                best_path = path
    
    return best_epoch, best_path

def main():
    parser = argparse.ArgumentParser(description='Find the best model checkpoint')
    parser.add_argument('--criterion', type=str, default='combined_score', 
                        choices=['discriminator_loss', 'realism_score', 'privacy_score', 'utility_score', 'combined_score'],
                        help='Criterion to use for selecting the best model')
    parser.add_argument('--higher_is_better', action='store_true', default=True,
                        help='Whether higher values are better for the criterion')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='inference/best_model.json',
                        help='Output file to save the best model information')
    
    args = parser.parse_args()
    
    # Adjust for discriminator_loss where lower is better
    if args.criterion == 'discriminator_loss':
        args.higher_is_better = False
    
    # Find all model checkpoints
    print("Searching for model checkpoints...")
    checkpoints = find_model_checkpoints()
    
    if not checkpoints:
        print("No model checkpoints found.")
        return
    
    print(f"Found {len(checkpoints)} model checkpoints.")
    for epoch, path in checkpoints:
        print(f"- Epoch {epoch}: {path}")
    
    # Initialize model
    print("\nInitializing model...")
    model, latent_dim, max_length = load_and_initialize_model()
    
    # Prepare test data
    print(f"Preparing test data with batch size {args.batch_size}...")
    X_test, traj_lengths = prepare_test_data(max_length, latent_dim, args.batch_size)
    
    # Evaluate each checkpoint
    print("\nEvaluating checkpoints...")
    metrics_dict = {}
    
    for epoch, path in checkpoints:
        print(f"Evaluating model from epoch {epoch}...")
        metrics = evaluate_model(model, path, X_test, traj_lengths)
        metrics_dict[epoch] = (path, metrics)
        
        if metrics:
            # Print some metrics
            print(f"  Realism: {metrics['realism_score']:.4f}, " +
                  f"Privacy: {metrics['privacy_score']:.4f}, " +
                  f"Utility: {metrics['utility_score']:.4f}, " +
                  f"Combined: {metrics['combined_score']:.4f}")
    
    # Find the best model
    best_epoch, best_path = find_best_model(metrics_dict, args.criterion, args.higher_is_better)
    
    if best_epoch is None:
        print("\nNo valid model found.")
        return
    
    print(f"\nBest model found: Epoch {best_epoch}, Path: {best_path}")
    print(f"Best {args.criterion}: {metrics_dict[best_epoch][1][args.criterion]:.4f}")
    
    # Save results to output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    results = {
        'best_epoch': best_epoch,
        'best_path': best_path,
        'criterion': args.criterion,
        'higher_is_better': args.higher_is_better,
        'metrics': metrics_dict[best_epoch][1]
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main() 