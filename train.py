#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train the improved RL-Enhanced Transformer-TrajGAN model for privacy-preserving trajectory generation
"""

import numpy as np
import os
import tensorflow as tf
import argparse
from datetime import datetime
import sys
import time
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(2020)
tf.random.set_seed(2020)

from model import RL_Transformer_TrajGAN
from keras.preprocessing.sequence import pad_sequences

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train RL-Enhanced Transformer-TrajGAN model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimension of latent noise vector')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='Number of PPO epochs per training epoch')
    parser.add_argument('--checkpoint_interval', type=int, default=10, help='Interval for saving model checkpoints')
    parser.add_argument('--alpha', type=float, default=0.2, help='Weight for privacy reward')
    parser.add_argument('--beta', type=float, default=0.6, help='Weight for utility reward')
    parser.add_argument('--gamma', type=float, default=0.2, help='Weight for adversarial reward')
    parser.add_argument('--ppo_epsilon', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Entropy coefficient for exploration')
    parser.add_argument('--early_stop', type=int, default=20, help='Early stopping patience (epochs)')
    parser.add_argument('--privacy_weight', type=float, default=0.6, help='Privacy weight for combined metric')
    parser.add_argument('--utility_weight', type=float, default=0.4, help='Utility weight for combined metric')
    parser.add_argument('--log_dir', type=str, default=None, help='Directory for saving logs')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    
    # Add transformer hyperparameters
    parser.add_argument('--head_size', type=int, default=64, help='Size of attention heads')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ff_dim', type=int, default=256, help='Feed forward dimension')
    parser.add_argument('--transformer_blocks', type=int, default=4, help='Number of transformer blocks')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Initialize wandb if requested
    if args.use_wandb:
        import wandb
        wandb_name = f"rl-transformer-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project="rl-transformer-trajgan",
            name=wandb_name,
            config=vars(args)
        )
    
    # Create log directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/run_{timestamp}"
    else:
        log_dir = args.log_dir
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{log_dir}/plots", exist_ok=True)
    
    # Create log file
    log_file = open(f"{log_dir}/training_log.txt", "w")
    
    # Define training data parameters
    max_length = 48  # Maximum trajectory length
    
    # Vocabulary sizes for each feature
    vocab_size = {
        'lat_lon': 2,     # lat, lon coordinates
        'day': 7,         # days of week
        'hour': 24,       # hours of day
        'category': 10,   # POI categories
        'mask': 1         # mask for valid points
    }
    
    # Feature keys
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    
    # Geographical parameters (for normalization)
    lat_centroid = 40.7128  # NYC latitude
    lon_centroid = -74.0060  # NYC longitude
    scale_factor = 0.01      # Scaling factor for coordinates
    
    # Log configuration
    print("="*50)
    print("RL-Enhanced Transformer-TrajGAN Training")
    print("="*50)
    print(f"Training configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - PPO epochs per batch: {args.ppo_epochs}")
    print(f"  - Latent dimension: {args.latent_dim}")
    print(f"  - Reward weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"  - Transformer config: heads={args.num_heads}, head_size={args.head_size}, ff_dim={args.ff_dim}")
    print(f"  - Log directory: {log_dir}")
    print("="*50)
    
    # Write configuration to log file
    log_file.write("="*50 + "\n")
    log_file.write("RL-Enhanced Transformer-TrajGAN Training\n")
    log_file.write("="*50 + "\n")
    log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Training configuration:\n")
    log_file.write(f"  - Epochs: {args.epochs}\n")
    log_file.write(f"  - Batch size: {args.batch_size}\n")
    log_file.write(f"  - PPO epochs per batch: {args.ppo_epochs}\n")
    log_file.write(f"  - Latent dimension: {args.latent_dim}\n")
    log_file.write(f"  - Reward weights: α={args.alpha}, β={args.beta}, γ={args.gamma}\n")
    log_file.write(f"  - Transformer config: heads={args.num_heads}, head_size={args.head_size}, ff_dim={args.ff_dim}\n")
    log_file.write("="*50 + "\n\n")
    
    try:
        # Initialize model
        model = RL_Transformer_TrajGAN(
            latent_dim=args.latent_dim,
            keys=keys,
            vocab_size=vocab_size,
            max_length=max_length,
            lat_centroid=lat_centroid,
            lon_centroid=lon_centroid,
            scale_factor=scale_factor
        )
        
        # Set hyperparameters
        model.head_size = args.head_size
        model.num_heads = args.num_heads
        model.ff_dim = args.ff_dim
        model.transformer_blocks = args.transformer_blocks
        model.dropout_rate = args.dropout_rate
        
        # Set reward weights
        model.alpha = args.alpha
        model.beta = args.beta
        model.gamma = args.gamma
        
        # Set PPO parameters
        model.ppo_epsilon = args.ppo_epsilon
        model.entropy_coeff = args.entropy_coeff
        
        # Load training data
        print("Loading training data...")
        x_train = np.load('data/final_train.npy', allow_pickle=True)
        X_train = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in x_train]
        
        # Load validation data
        val_size = min(1000, X_train[0].shape[0])
        val_indices = np.random.choice(X_train[0].shape[0], val_size, replace=False)
        X_val = [f[val_indices] for f in X_train]
        
        # Validation user IDs
        val_user_ids = np.arange(val_size) % 193
        
        # Initialize metric tracking
        metrics_history = {
            'actor_loss': [],
            'critic_loss': [],
            'discriminator_loss': [],
            'discriminator_acc': [],
            'privacy_reward': [],
            'utility_reward': [],
            'adv_reward': [],
            'total_reward': [],
            'val_privacy': [],
            'val_utility': [],
            'val_combined': []
        }
        
        # Initialize early stopping variables
        best_score = float('inf')
        best_epoch = 0
        patience_counter = 0
        
        # Main training loop
        print("\nStarting training...")
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            train_metrics = model.train_epoch(X_train, args.batch_size, args.ppo_epochs)
            
            # Evaluate on validation set
            val_metrics = model.evaluate_model(X_val, val_user_ids)
            
            # Calculate combined validation score
            privacy_score = val_metrics['privacy_acc@1']  # Lower is better for privacy
            utility_score = val_metrics['utility_spatial_dist']  # Lower is better for utility
            
            # Normalize scores to 0-1 range (assuming 0.5 is max acceptable privacy leak, 0.1 is max acceptable distance)
            norm_privacy_score = min(1.0, privacy_score / 0.5)
            norm_utility_score = min(1.0, utility_score / 0.1)
            
            # Weighted combination (lower is better)
            combined_score = args.privacy_weight * norm_privacy_score + args.utility_weight * norm_utility_score
            
            # Update metrics history
            metrics_history['actor_loss'].append(train_metrics['actor_loss'])
            metrics_history['critic_loss'].append(train_metrics['critic_loss'])
            metrics_history['discriminator_loss'].append(train_metrics['discriminator_loss'])
            metrics_history['discriminator_acc'].append(train_metrics['discriminator_acc'])
            metrics_history['privacy_reward'].append(train_metrics['privacy_reward'])
            metrics_history['utility_reward'].append(train_metrics['utility_reward'])
            metrics_history['adv_reward'].append(train_metrics['adv_reward'])
            metrics_history['total_reward'].append(train_metrics['total_reward'])
            metrics_history['val_privacy'].append(privacy_score)
            metrics_history['val_utility'].append(utility_score)
            metrics_history['val_combined'].append(combined_score)
            
            # Log training progress
            epoch_time = time.time() - start_time
            log_message = (
                f"[Epoch {epoch}/{args.epochs}] "
                f"Time: {epoch_time:.2f}s | "
                f"A_Loss: {train_metrics['actor_loss']:.4f} | "
                f"C_Loss: {train_metrics['critic_loss']:.4f} | "
                f"D_Loss: {train_metrics['discriminator_loss']:.4f} | "
                f"D_Acc: {train_metrics['discriminator_acc']:.4f} | "
                f"Reward: {train_metrics['total_reward']:.4f} | "
                f"Val Privacy: {privacy_score:.4f} | "
                f"Val Utility: {utility_score:.4f} | "
                f"Val Score: {combined_score:.4f}"
            )
            
            print(log_message)
            log_file.write(log_message + "\n")
            
            # Log to wandb if enabled
            if args.use_wandb:
                wandb_metrics = {
                    'epoch': epoch,
                    'actor_loss': train_metrics['actor_loss'],
                    'critic_loss': train_metrics['critic_loss'],
                    'discriminator_loss': train_metrics['discriminator_loss'],
                    'discriminator_acc': train_metrics['discriminator_acc'],
                    'privacy_reward': train_metrics['privacy_reward'],
                    'utility_reward': train_metrics['utility_reward'],
                    'adv_reward': train_metrics['adv_reward'],
                    'total_reward': train_metrics['total_reward'],
                    'val_privacy_acc1': privacy_score,
                    'val_utility_dist': utility_score,
                    'val_combined_score': combined_score,
                    'epoch_time': epoch_time
                }
                wandb.log(wandb_metrics)
            
            # Check if this is the best model
            if combined_score < best_score:
                best_score = combined_score
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                model.save_models(f"{log_dir}/best_model")
                
                # Log best model update
                print(f"New best model! Score: {combined_score:.4f}")
                log_file.write(f"New best model saved at epoch {epoch} with score {combined_score:.4f}\n")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs. Best: {best_score:.4f} at epoch {best_epoch}")
                log_file.write(f"No improvement for {patience_counter} epochs. Best: {best_score:.4f} at epoch {best_epoch}\n")
            
            # Early stopping check
            if patience_counter >= args.early_stop:
                print(f"Early stopping triggered after {epoch} epochs")
                log_file.write(f"Early stopping triggered after {epoch} epochs\n")
                break
            
            # Save checkpoint if needed
            if epoch % args.checkpoint_interval == 0:
                model.save_models(f"{log_dir}/checkpoints/epoch_{epoch}")
                print(f"Checkpoint saved at epoch {epoch}")
                log_file.write(f"Checkpoint saved at epoch {epoch}\n")
                
                # Plot training metrics
                plot_training_metrics(metrics_history, log_dir, epoch)
        
        # Training complete
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n" + "="*50)
        print(f"Training completed at {end_time}")
        print(f"Best model at epoch {best_epoch} with score {best_score:.4f}")
        print("="*50)
        
        log_file.write("\n" + "="*50 + "\n")
        log_file.write(f"Training completed at {end_time}\n")
        log_file.write(f"Best model at epoch {best_epoch} with score {best_score:.4f}\n")
        log_file.write("="*50 + "\n")
        
        # Final plots
        plot_training_metrics(metrics_history, log_dir, args.epochs, final=True)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        log_file.write(f"Error during training: {str(e)}\n")
        raise
    
    finally:
        log_file.close()
        if args.use_wandb:
            wandb.finish()

def plot_training_metrics(metrics_history, log_dir, epoch, final=False):
    """Plot training metrics and save to disk."""
    if len(metrics_history['actor_loss']) == 0:
        return  # Skip plotting if no data yet
    # Create a 2x3 grid of plots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    # Convert epoch numbers to list
    epochs = list(range(1, len(metrics_history['actor_loss']) + 1))
    
    # Plot actor and critic losses
    axs[0, 0].plot(epochs, metrics_history['actor_loss'], 'b-', label='Actor Loss')
    axs[0, 0].plot(epochs, metrics_history['critic_loss'], 'r-', label='Critic Loss')
    axs[0, 0].set_title('Actor & Critic Losses')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot discriminator metrics
    axs[0, 1].plot(epochs, metrics_history['discriminator_loss'], 'g-', label='Discriminator Loss')
    axs[0, 1].plot(epochs, metrics_history['discriminator_acc'], 'm-', label='Discriminator Accuracy')
    axs[0, 1].set_title('Discriminator Metrics')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot reward components
    axs[0, 2].plot(epochs, metrics_history['privacy_reward'], 'c-', label='Privacy Reward')
    axs[0, 2].plot(epochs, metrics_history['utility_reward'], 'y-', label='Utility Reward')
    axs[0, 2].plot(epochs, metrics_history['adv_reward'], 'k-', label='Adversarial Reward')
    axs[0, 2].plot(epochs, metrics_history['total_reward'], 'g-', label='Total Reward')
    axs[0, 2].set_title('Reward Components')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Reward')
    axs[0, 2].legend()
    axs[0, 2].grid(True)
    
    # Plot validation privacy metric
    axs[1, 0].plot(epochs, metrics_history['val_privacy'], 'r-')
    axs[1, 0].set_title('Validation Privacy (ACC@1)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy (lower is better)')
    axs[1, 0].grid(True)
    
    # Plot validation utility metric
    axs[1, 1].plot(epochs, metrics_history['val_utility'], 'b-')
    axs[1, 1].set_title('Validation Utility (Spatial Distance)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Distance (lower is better)')
    axs[1, 1].grid(True)
    
    # Plot validation combined score
    axs[1, 2].plot(epochs, metrics_history['val_combined'], 'g-')
    axs[1, 2].set_title('Validation Combined Score')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Score (lower is better)')
    axs[1, 2].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    if final:
        plt.savefig(f"{log_dir}/plots/final_metrics.png")
    else:
        plt.savefig(f"{log_dir}/plots/metrics_epoch_{epoch}.png")
    
    plt.close()

if __name__ == "__main__":
    main()