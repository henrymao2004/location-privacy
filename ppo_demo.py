import argparse
import numpy as np
import tensorflow as tf
import os
import time
from models.ppo_trajgan import RL_Enhanced_Transformer_TrajGAN
from models.tul_classifier import TULClassifier

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PPO-enhanced TrajGAN training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension size')
    parser.add_argument('--save_interval', type=int, default=5, help='Checkpoint saving interval')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    
    return parser.parse_args()

def load_data():
    """Load and prepare data."""
    print("Loading data...")
    
    # Load training data
    x_train = np.load('data/final_train.npy', allow_pickle=True)
    
    # Load or compute data statistics
    if os.path.exists('data/data_stats.npy'):
        data_stats = np.load('data/data_stats.npy', allow_pickle=True).item()
    else:
        raise ValueError("Data statistics not found. Please run train.py first.")
    
    return x_train, data_stats

def initialize_model(data_stats, latent_dim, learning_rate=0.0001):
    """Initialize the PPO-enhanced model."""
    print("Initializing model...")
    
    # Define model parameters
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
    
    # Initialize model
    model = RL_Enhanced_Transformer_TrajGAN(
        latent_dim=latent_dim,
        keys=keys,
        vocab_size=vocab_size,
        max_length=max_length,
        lat_centroid=lat_centroid,
        lon_centroid=lon_centroid,
        scale_factor=scale_factor
    )
    
    # Update learning rate if specified
    if learning_rate != 0.0001:
        model.gen_lr = learning_rate
        model.disc_lr = learning_rate
        model.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    return model

def initialize_tul_classifier(data_stats):
    """Initialize the TUL classifier for privacy evaluation."""
    print("Initializing TUL classifier...")
    
    # Assume we have 100 users for this demo
    # In a real scenario, you would determine this from your data
    num_users = 100
    
    # Initialize TUL classifier
    tul_classifier = TULClassifier(
        num_users=num_users,
        max_length=data_stats['max_length'],
        feature_dim=64
    )
    
    # Try to load pre-trained TUL model if available
    if os.path.exists('models/tul_model.h5'):
        try:
            tul_classifier.load('models/tul_model.h5')
            print("Loaded pre-trained TUL model")
        except Exception as e:
            print(f"Error loading TUL model: {e}")
            print("Will use untrained TUL classifier")
    else:
        print("No pre-trained TUL model found")
    
    return tul_classifier

def train_model(model, x_train, tul_classifier, args):
    """Train the PPO-enhanced model."""
    print("Starting model training...")
    
    # Set TUL classifier for privacy reward computation
    model.set_tul_classifier(tul_classifier)
    
    # Start timing the training
    start_time = time.time()
    
    # Train the model
    training_results = model.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_interval=args.save_interval,
        early_stopping=True,
        patience=args.patience,
        min_delta=0.001
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Report best model information
    if isinstance(training_results, dict):
        best_epoch = training_results.get('best_epoch', 0)
        best_reward = training_results.get('best_reward', 0.0)
        print(f"Best model found at epoch {best_epoch} with reward {best_reward:.4f}")
        
        # Save final statistics to a file
        with open('training_results.txt', 'w') as f:
            f.write(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
            f.write(f"Best model found at epoch {best_epoch} with reward {best_reward:.4f}\n")
            f.write("\nTraining statistics:\n")
            
            # Record loss history
            f.write("\nEpoch,D_Loss,G_Loss,Policy_Loss,Value_Loss,Reward\n")
            for i, epoch in enumerate(training_results['training_stats']['epochs']):
                f.write(f"{epoch},{training_results['training_stats']['d_losses'][i]:.4f},"
                       f"{training_results['training_stats']['g_losses'][i]:.4f},"
                       f"{training_results['training_stats']['policy_losses'][i]:.4f},"
                       f"{training_results['training_stats']['value_losses'][i]:.4f},"
                       f"{training_results['training_stats']['rewards'][i]:.4f}\n")
    
    return training_results

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Configure GPU memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
            
            # Set reasonable memory limits to avoid OOM
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=10 * 1024)]  # 10GB limit
                )
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPUs found, using CPU only.")
    
    # Load data
    x_train, data_stats = load_data()
    
    # Initialize model
    model = initialize_model(data_stats, args.latent_dim, args.learning_rate)
    
    # Initialize TUL classifier
    tul_classifier = initialize_tul_classifier(data_stats)
    
    # Train the model
    train_model(model, x_train, tul_classifier, args)
    
    print("Training complete. Model weights saved in 'checkpoints/' directory.")
    print("Best model saved as 'checkpoints/generator_best.h5', 'checkpoints/discriminator_best.h5', and 'checkpoints/critic_best.h5'")

if __name__ == '__main__':
    main() 