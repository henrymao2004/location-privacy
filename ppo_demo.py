import sys
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from models.ppo_trajgan import RL_Enhanced_Transformer_TrajGAN
from models.tul_classifier import TULClassifier
from keras.preprocessing.sequence import pad_sequences

# Set random seed for reproducibility
np.random.seed(2023)
tf.random.set_seed(2023)

# Enable memory growth to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth enabled for {device}")

if __name__ == '__main__':
    # Get epochs and patience from command line arguments or use defaults
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    patience = int(sys.argv[2]) if len(sys.argv) > 2 else 50  # Increased patience (default 50)
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    
    print(f"Training for {n_epochs} epochs with patience {patience}")
    print(f"Using batch size {batch_size}")
    print("Enhanced training with utility-focused rewards enabled")
    
    # Model parameters
    latent_dim = 100
    max_length = 144
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon": 2, "day": 7, "hour": 24, "category": 10, "mask": 1}
    
    # Load training data statistics
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
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
    
    print(f"Data statistics: lat_centroid={lat_centroid}, lon_centroid={lon_centroid}, scale_factor={scale_factor}")
    
    # Initialize the PPO-Enhanced Transformer TrajGAN model
    model = RL_Enhanced_Transformer_TrajGAN(
        latent_dim=latent_dim, 
        keys=keys, 
        vocab_size=vocab_size, 
        max_length=max_length, 
        lat_centroid=lat_centroid, 
        lon_centroid=lon_centroid, 
        scale_factor=scale_factor
    )
    
    # Configure reward weights to emphasize utility
    model.reward_function.alpha = 0.3  # Privacy weight (decreased from default)
    model.reward_function.beta = 0.5   # Utility weight (increased from default)
    model.reward_function.gamma = 0.2  # GAN weight (unchanged)
    
    print(f"Reward weights configured - Privacy: {model.reward_function.alpha}, " +
          f"Utility: {model.reward_function.beta}, GAN: {model.reward_function.gamma}")
    
    # Load training data
    print("Loading training data...")
    x_train = np.load('data/final_train.npy', allow_pickle=True)
    
    # Print data shapes for debugging
    for i, component in enumerate(x_train):
        component_name = ['lat_lon', 'day', 'hour', 'category', 'mask'][i] if i < 5 else f"component_{i}"
        print(f"{component_name}: shape={component.shape if hasattr(component, 'shape') else 'list with len ' + str(len(component))}")
        
    # Initialize TUL classifier for privacy evaluation
    print("Initializing TUL classifier...")
    num_users = 193  # Default value, adapt if needed
    tul_classifier = TULClassifier(
        num_users=num_users,
        max_length=max_length,
        feature_dim=64,
        category_size=vocab_size['category']
    )
    
    # Try to load pre-trained TUL model if available
    tul_model_paths = [
        'models/tul_model.h5',
        'MARC/MARC_Weight.h5',
        'checkpoints/tul_model.h5'
    ]
    
    tul_loaded = False
    for path in tul_model_paths:
        if os.path.exists(path):
            print(f"Loading TUL model from {path}")
            if tul_classifier.load(path):
                tul_loaded = True
                break
    
    if not tul_loaded:
        print("WARNING: Could not load TUL model. Privacy rewards may not be accurate.")
    
    # Set TUL classifier for privacy reward computation
    model.set_tul_classifier(tul_classifier)
    
    # Create directories for checkpoints and results if they don't exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Try to load existing model weights if continuing training
    try_load_checkpoint = False  # Set to True to try loading existing checkpoint
    
    if try_load_checkpoint:
        checkpoint_path = 'checkpoints/generator_best.weights.h5'
        if os.path.exists(checkpoint_path):
            print(f"Loading existing best model from {checkpoint_path}")
            model.load_checkpoint(load_best=True)
            print("Checkpoint loaded successfully")
    
    print(f"Starting training with batch size {batch_size}...")
    
    # Train the model with increased patience and utility focus
    results = model.train(
        epochs=n_epochs,
        batch_size=batch_size,
        sample_interval=5,  # Save checkpoint every 5 epochs
        early_stopping=True,
        patience=patience,  # Using the increased patience
        min_delta=0.001,
        use_wandb=True  # Set to True if you want to use Weights & Biases
    )
    
    # Print training results
    if isinstance(results, dict):
        best_epoch = results.get('best_epoch', 0)
        best_reward = results.get('best_reward', 0.0)
        best_utility = results.get('best_utility', 0.0)
        
        print(f"Training completed. Best model found at epoch {best_epoch}.")
        print(f"Best combined score: {best_reward:.4f}")
        print(f"Best utility score: {best_utility:.4f}")
        
        # Save the training results to a file
        with open('results/training_results.txt', 'w') as f:
            f.write(f"Best model found at epoch {best_epoch}\n")
            f.write(f"Best combined score: {best_reward:.4f}\n")
            f.write(f"Best utility score: {best_utility:.4f}\n\n")
            
            if 'training_stats' in results:
                f.write("\nTraining statistics:\n")
                if 'epochs' in results['training_stats']:
                    f.write("\nEpoch,D_Loss,G_Loss,Policy_Loss,Value_Loss,Reward,Utility\n")
                    for i, epoch in enumerate(results['training_stats']['epochs']):
                        try:
                            f.write(f"{epoch},{results['training_stats']['d_losses'][i]:.4f},"
                                  f"{results['training_stats']['g_losses'][i]:.4f},"
                                  f"{results['training_stats']['policy_losses'][i]:.4f},"
                                  f"{results['training_stats']['value_losses'][i]:.4f},"
                                  f"{results['training_stats']['rewards'][i]:.4f},"
                                  f"{results['training_stats']['utility_rewards'][i]:.4f}\n")
                        except IndexError:
                            pass  # Skip if data is incomplete
    else:
        print("Training completed but no detailed results were returned.")
    
    print("Training complete. The best model is saved in the checkpoints directory.")
    print("Now you can use inference.py to generate synthetic trajectories using the trained model.") 