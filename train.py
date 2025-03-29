import numpy as np
import os
import pandas as pd
import wandb
from model import RL_Enhanced_Transformer_TrajGAN
from MARC.marc import MARC
from collections import deque
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the WandbModelCheckpoint class OUTSIDE of any function
class WandbModelCheckpoint:
    def __init__(self, model, checkpoint_dir, best_model_name, patience=20):
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.best_model_name = best_model_name
        self.best_reward = float('-inf')
        self.best_epoch = 0
        self.patience = patience
        self.no_improvement_count = 0
        self.should_stop = False
        # Store recent rewards to detect overfitting trends
        self.recent_rewards = deque(maxlen=5)
    
    def on_epoch_end(self, epoch, logs=None):
        if logs and 'reward' in logs:
            current_reward = logs['reward']
            self.recent_rewards.append(current_reward)
            
            # Check if current reward is better than best reward
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self.best_epoch = epoch
                self.no_improvement_count = 0
                
                # Save best model
                self.model.save_best_checkpoint(self.checkpoint_dir, self.best_model_name)
                
                # Log to wandb
                wandb.log({
                    "best_reward": self.best_reward, 
                    "best_reward_epoch": self.best_epoch
                })
                print(f"\nNew best model saved at epoch {epoch} with reward {current_reward:.4f}")
            else:
                self.no_improvement_count += 1
                
                # Log early stopping metrics
                wandb.log({
                    "epochs_without_improvement": self.no_improvement_count
                })
                
                # Check if we should stop training
                if self.no_improvement_count >= self.patience:
                    self.should_stop = True
                    print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                    print(f"Best model was at epoch {self.best_epoch} with reward {self.best_reward:.4f}")
                    return True  # Signal to stop training
            
            return False  # Continue training

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

# Modify the original train method to include early stopping
def train_with_early_stopping(self, epochs=2000, batch_size=256, sample_interval=10, 
                             save_best=True, checkpoint_dir='results'):
    """Train the model with early stopping."""
    # Make sure the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize the early stopping callback
    early_stopping = WandbModelCheckpoint(
        model=self,
        checkpoint_dir=checkpoint_dir,
        best_model_name="best_early_stopping_model",
        patience=20  # Configure as needed
    )
    
    # Add patience to wandb config
    if self.wandb:
        self.wandb.config.update({
            "early_stopping_patience": early_stopping.patience
        })
    
    # Training data
    x_train = np.load('data/final_train.npy', allow_pickle=True)
    self.x_train = x_train
    
    # Padding
    X_train = [pad_sequences(f, self.max_length, padding='pre', dtype='float64') 
              for f in x_train]
    self.X_train = X_train
    
    # Check if we need to rebuild the model with correct input shapes
    needs_rebuild = False
    actual_shapes = {}
    
    for i, key in enumerate(self.keys):
        if key != 'mask':
            actual_shape = X_train[i].shape
            print(f"Data shape for {key}: {actual_shape}")
            if key == 'category' and actual_shape[2] != self.vocab_size[key]:
                print(f"Mismatch for {key}: expected {self.vocab_size[key]}, got {actual_shape[2]}")
                self.vocab_size[key] = actual_shape[2]
                needs_rebuild = True
            actual_shapes[key] = actual_shape[2]
    
    # Rebuild the model if needed
    if needs_rebuild:
        print("Rebuilding model with correct input shapes...")
        # Save optimizer state
        optimizer_weights = None
        if hasattr(self, 'actor_optimizer') and hasattr(self.actor_optimizer, 'get_weights'):
            optimizer_weights = self.actor_optimizer.get_weights()
        
        # Rebuild models
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.discriminator = self.build_discriminator()
        
        # Compile models
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer)
        self.critic.compile(loss='mse', optimizer=self.critic_optimizer)
        
        self.setup_combined_model()
        # Restore optimizer state if available
        if optimizer_weights is not None:
            self.actor_optimizer.set_weights(optimizer_weights)
        
        print("Model rebuilt successfully!")
    
    # Training loop
    print(f"Starting training for {epochs} epochs with early stopping (patience={early_stopping.patience})...")
    for epoch in range(epochs):
        # Sample batch
        idx = np.random.randint(0, len(X_train[0]), batch_size)
        batch = [X[idx] for X in X_train]
        
        # Training step
        metrics = self.train_step(batch, batch_size)
        
        # Log metrics to WandB
        if self.wandb:
            wandb_metrics = {
                "epoch": epoch,
                "d_loss_real": metrics['d_loss_real'],
                "d_loss_fake": metrics['d_loss_fake'],
                "g_loss": metrics['g_loss'],
                "c_loss": metrics['c_loss']
            }
            
            # Calculate combined D loss for tracking
            d_loss_combined = (metrics['d_loss_real'] + metrics['d_loss_fake']) / 2
            wandb_metrics["d_loss_combined"] = d_loss_combined
            
            # Add any available reward metrics
            if 'reward' in metrics:
                wandb_metrics["reward"] = metrics['reward']
                
                # Check for early stopping
                should_stop = early_stopping.on_epoch_end(epoch, metrics)
                if should_stop:
                    print(f"Training stopped at epoch {epoch} due to early stopping.")
                    break
                
                # Track and save the best model based on reward
                if save_best and metrics['reward'] > self.best_reward:
                    self.best_reward = metrics['reward']
                    wandb_metrics["best_reward"] = self.best_reward
                    self.save_best_checkpoint(checkpoint_dir, f"best_reward_model")
                    print(f"New best reward: {self.best_reward:.4f} at epoch {epoch}")
            
            # Track and save based on generator loss
            if save_best and metrics['g_loss'] < self.best_g_loss:
                self.best_g_loss = metrics['g_loss']
                wandb_metrics["best_g_loss"] = self.best_g_loss
                self.save_best_checkpoint(checkpoint_dir, f"best_g_loss_model")
                print(f"New best generator loss: {self.best_g_loss:.4f} at epoch {epoch}")
            
            # Track and save based on discriminator loss
            if save_best and d_loss_combined < self.best_d_loss:
                self.best_d_loss = d_loss_combined
                wandb_metrics["best_d_loss"] = self.best_d_loss
                self.save_best_checkpoint(checkpoint_dir, f"best_d_loss_model")
                print(f"New best discriminator loss: {self.best_d_loss:.4f} at epoch {epoch}")
            
            # Log to wandb
            self.wandb.log(wandb_metrics)
        
            # Generate trajectory visualizations for wandb every 50 epochs
            if epoch % 50 == 0:
                self.sample_trajectories_for_wandb(epoch)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"D_real: {metrics['d_loss_real']:.4f}, D_fake: {metrics['d_loss_fake']:.4f}, G: {metrics['g_loss']:.4f}, C: {metrics['c_loss']:.4f}")
            if 'reward' in metrics:
                print(f"Reward: {metrics['reward']:.4f}")
        
        # Save checkpoints
        if epoch % sample_interval == 0:
            self.save_checkpoint(epoch)
    
    # Log final training summary
    wandb.log({
        "training_completed": True,
        "total_epochs_trained": epoch + 1,
        "early_stopping_best_epoch": early_stopping.best_epoch,
        "early_stopping_best_reward": early_stopping.best_reward
    })
    
    print(f"\nTraining completed. Best model was at epoch {early_stopping.best_epoch} "
          f"with reward {early_stopping.best_reward:.4f}")
    
    return early_stopping.best_epoch, early_stopping.best_reward

def main():
    # Initialize wandb
    wandb.init(
        project="location",
        entity="xutao-henry-mao-vanderbilt-university",
        config={
            "architecture": "RL_Enhanced_Transformer_TrajGAN",
            "epochs": 2000,
            "batch_size": 256,
            "latent_dim": 100,
            "max_length": 144,
            "rl_params": {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_epsilon": 0.2,
                "ppo_epochs": 4
            },
            "reward_weights": {
                "adversarial": 0.5,
                "utility": 0.5,
                "privacy": 0.5
            },
            "early_stopping": {
                "patience": 20,
                "min_delta": 0.001
            }
        }
    )
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Compute or load data statistics
    if not os.path.exists('data/data_stats.npy'):
        print("Computing data statistics...")
        data_stats = compute_data_stats()
    else:
        print("Loading data statistics...")
        data_stats = np.load('data/data_stats.npy', allow_pickle=True).item()
    
    # Add data statistics to wandb config
    wandb.config.update({
        "lat_centroid": data_stats['lat_centroid'],
        "lon_centroid": data_stats['lon_centroid'],
        "scale_factor": data_stats['scale_factor'],
        "category_size": data_stats['category_size']
    })
    
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
    
    # Initialize TUL classifier (MARC)
    tul_classifier = MARC()
    tul_classifier.load_weights('/root/autodl-tmp/location-privacy/MARC/MARC_Weight.h5')
    
    # Initialize and train the model
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
    
    # Set wandb instance for the model
    model.set_wandb(wandb)
    
    # Training parameters
    epochs = 2000
    batch_size = 256
    sample_interval = 10
    
    # Log model architecture as text
    generator_summary = []
    model.generator.summary(print_fn=lambda x: generator_summary.append(x))
    discriminator_summary = []
    model.discriminator.summary(print_fn=lambda x: discriminator_summary.append(x))
    critic_summary = []
    model.critic.summary(print_fn=lambda x: critic_summary.append(x))
    
    wandb.log({
        "generator_architecture": wandb.Html("<pre>" + "\n".join(generator_summary) + "</pre>"),
        "discriminator_architecture": wandb.Html("<pre>" + "\n".join(discriminator_summary) + "</pre>"),
        "critic_architecture": wandb.Html("<pre>" + "\n".join(critic_summary) + "</pre>"),
    })
    
    # Replace the original train method with our early stopping version
    original_train = model.train
    model.train = lambda *args, **kwargs: train_with_early_stopping(model, *args, **kwargs)
    
    # Train the model with early stopping
    best_epoch, best_reward = model.train(
        epochs=epochs, 
        batch_size=batch_size, 
        sample_interval=sample_interval, 
        save_best=True, 
        checkpoint_dir=checkpoint_dir
    )
    
    # Log final results
    wandb.log({
        "final_best_epoch": best_epoch,
        "final_best_reward": best_reward
    })
    
    # Save the best model path to wandb
    wandb.save(f"{checkpoint_dir}/best_early_stopping_model*")
    
    # Close wandb run when done
    wandb.finish()

if __name__ == '__main__':
    main()