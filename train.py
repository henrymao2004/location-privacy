import numpy as np
import os
import pandas as pd
import tensorflow as tf
from model import RL_Enhanced_Transformer_TrajGAN
from MARC.marc import MARC
import gc
import time

def compute_data_stats():
    """Compute statistics from training data."""
    # Load training data
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
    # Compute centroids
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
    # Compute scale factor
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
    
    # Load training data for category size
    x_train = np.load('data/final_train.npy', allow_pickle=True)
    category_size = x_train[3].shape[-1]  # Get category vocabulary size
    
    # Get max sequence length
    max_length = 144  # Default value, can be adjusted based on your data
    
    # Create stats dictionary
    stats = {
        'lat_centroid': lat_centroid,
        'lon_centroid': lon_centroid,
        'scale_factor': scale_factor,
        'category_size': category_size,
        'max_length': max_length
    }
    
    # Save stats
    np.save('data/data_stats.npy', stats)
    return stats

def configure_gpu_memory():
    """Configure GPU memory growth to prevent OOM errors."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to prevent allocating all memory up front
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            
            # Set mixed precision policy for faster training
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision policy set to mixed_float16")
            
            # Reduce precision for operations that support it
            tf.config.optimizer.set_experimental_options({
                'auto_mixed_precision': True,
                'layout_optimizer': True,
                'constant_folding': True,
                'shape_optimization': True
            })
            print("TensorFlow optimizer experimental options set")
            
            # Set reasonable memory limits to avoid OOM
            # This reserves 2GB for the system and uses the rest for TensorFlow
            for gpu in gpus:
                memory_limit = 10 * 1024  # 10GB or adjust according to your GPU
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                print(f"Memory limit set to {memory_limit/1024:.1f}GB for GPU")
                
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPUs found, using CPU only.")

def main():
    # Configure GPU for optimal performance
    configure_gpu_memory()
    
    # Use deterministic operations for reproducibility
    if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
        tf.config.experimental.enable_op_determinism()
        print("TensorFlow deterministic operations enabled")
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Compute or load data statistics
    if not os.path.exists('data/data_stats.npy'):
        print("Computing data statistics...")
        data_stats = compute_data_stats()
    else:
        print("Loading data statistics...")
        data_stats = np.load('data/data_stats.npy', allow_pickle=True).item()
    
    # Initialize model parameters
    latent_dim = 100
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {
        'lat_lon': 2,
        'day': 7,
        'hour': 24,
        'category': data_stats['category_size'],
        'mask': 1
    }
    max_length = data_stats['max_length']
    lat_centroid = data_stats['lat_centroid']
    lon_centroid = data_stats['lon_centroid']
    scale_factor = data_stats['scale_factor']
    
    # Clean up memory before loading TUL classifier
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    
    # Initialize TUL classifier (MARC)
    print("Loading TUL classifier...")
    tul_classifier = MARC()
    try:
        tul_classifier.load_weights('MARC/weights/MARC_Weight.h5')
        print("TUL classifier weights loaded successfully")
    except Exception as e:
        print(f"Warning: Error loading TUL classifier weights: {e}")
        print("Training will proceed with a default classifier")
    
    # Clean up memory before model initialization
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    
    # Initialize and train the model
    print("Initializing RL_Enhanced_Transformer_TrajGAN model...")
    model = RL_Enhanced_Transformer_TrajGAN(
        latent_dim=latent_dim,
        keys=keys,
        vocab_size=vocab_size,
        max_length=max_length,
        lat_centroid=lat_centroid,
        lon_centroid=lon_centroid,
        scale_factor=scale_factor
    )
    
    # Set TUL classifier for reward computation
    model.tul_classifier = tul_classifier
    
    # Training parameters - adjusted for better performance
    epochs = 2000
    # Use smaller batch size to avoid memory issues
    batch_size = 32  # Further reduced from 64
    sample_interval = 10
    
    # Early stopping parameters
    early_stopping = True
    patience = 15  # Reduced patience for faster response
    min_delta = 0.001
    
    print(f"Starting training with optimized parameters:")
    print(f"- Epochs: {epochs}")
    print(f"- Batch size: {batch_size} (reduced to prevent OOM errors)")
    print(f"- Early stopping patience: {patience}")
    
    # Clean up memory before training
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
    
    # Start timing the training
    start_time = time.time()
    
    try:
        # Train the model with early stopping and wandb integration
        model.train(
            epochs=epochs, 
            batch_size=batch_size, 
            sample_interval=sample_interval,
            early_stopping=early_stopping,
            patience=patience,
            min_delta=min_delta
        )
        
        # Report training time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print(f"Training completed in {hours}h {minutes}m {seconds}s")
        
    except Exception as e:
        print(f"Error during training: {e}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()
        
        # Report partial training time
        partial_time = time.time() - start_time
        hours = int(partial_time // 3600)
        minutes = int((partial_time % 3600) // 60)
        seconds = int(partial_time % 60)
        
        print(f"Training interrupted after {hours}h {minutes}m {seconds}s")

if __name__ == '__main__':
    main()