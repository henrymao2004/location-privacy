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
        self.best_pre_norm_reward = float('-inf')
        self.best_epoch = 0
        self.patience = patience
        self.wait_count = 0  # Renamed from no_improvement_count for consistency
        self.should_stop = False
        # Store recent rewards to detect overfitting trends
        self.recent_rewards = deque(maxlen=5)
    
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            # Prefer pre-normalized reward when available
            if 'pre_norm_reward' in logs and logs['pre_norm_reward'] != 0.0:
                current_reward = logs['pre_norm_reward']
                reward_type = 'pre-normalized'
                self.recent_rewards.append(current_reward)
                
                # Check if current reward is better than best reward
                if current_reward > self.best_pre_norm_reward:
                    self.best_pre_norm_reward = current_reward
                    self.best_reward = logs.get('reward', 0.0)  # Also track normalized reward
                    self.best_epoch = epoch
                    self.wait_count = 0  # Reset counter when improvement is found
                    
                    # Save best model
                    self.model.save_best_checkpoint(self.checkpoint_dir, self.best_model_name)
                    
                    # Log to wandb
                    wandb.log({
                        "best_pre_norm_reward": self.best_pre_norm_reward, 
                        "best_reward": self.best_reward,
                        "best_reward_epoch": self.best_epoch
                    })
                    print(f"\nNew best model saved at epoch {epoch} with {reward_type} reward {current_reward:.4f}")
                else:
                    self.wait_count += 1  # Increment counter when no improvement
                    
                    # Log early stopping metrics
                    wandb.log({
                        "epochs_without_improvement": self.wait_count
                    })
                    
                    # Check if we should stop training
                    if self.wait_count >= self.patience:
                        self.should_stop = True
                        print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                        print(f"Best model was at epoch {self.best_epoch} with {reward_type} reward {self.best_pre_norm_reward:.4f}")
                        return True  # Signal to stop training
            
            # Fall back to normalized reward if pre-normalized not available
            elif 'reward' in logs:
                current_reward = logs['reward']
                reward_type = 'normalized'
                self.recent_rewards.append(current_reward)
                
                # Check if current reward is better than best reward
                if current_reward > self.best_reward:
                    self.best_reward = current_reward
                    self.best_epoch = epoch
                    self.wait_count = 0  # Reset counter when improvement is found
                    
                    # Save best model
                    self.model.save_best_checkpoint(self.checkpoint_dir, self.best_model_name)
                    
                    # Log to wandb
                    wandb.log({
                        "best_reward": self.best_reward, 
                        "best_reward_epoch": self.best_epoch
                    })
                    print(f"\nNew best model saved at epoch {epoch} with {reward_type} reward {current_reward:.4f}")
                else:
                    self.wait_count += 1  # Increment counter when no improvement
                    
                    # Log early stopping metrics
                    wandb.log({
                        "epochs_without_improvement": self.wait_count
                    })
                    
                    # Check if we should stop training
                    if self.wait_count >= self.patience:
                        self.should_stop = True
                        print(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                        print(f"Best model was at epoch {self.best_epoch} with {reward_type} reward {self.best_reward:.4f}")
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
    
    try:
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
        
        # Stats for tracking
        best_reward = float('-inf')
        best_epoch = 0
        mean_reward_history = []
        
        for epoch in range(epochs):
            # Get random batch for training
            try:
                batch_indices = np.random.randint(0, len(X_train[0]), batch_size)
                batch = [X[batch_indices] for X in X_train]
                
                # Training step with error handling
                metrics = self.train_step(batch, batch_size)
                
                if epoch % sample_interval == 0:
                    print(f"Epoch {epoch}/{epochs}")
                    print(f"D_loss_real: {metrics['d_loss_real']:.4f}, D_loss_fake: {metrics['d_loss_fake']:.4f}")
                    print(f"G_loss: {metrics['g_loss']:.4f}, C_loss: {metrics['c_loss']:.4f}")
                    print(f"Mean reward: {metrics['reward']:.4f}")
                    if 'pre_norm_reward' in metrics:
                        print(f"Pre-norm mean reward: {metrics['pre_norm_reward']:.4f}")
                    
                    if self.wandb:
                        # Log metrics to wandb
                        wandb_metrics = {
                            "epoch": epoch,
                            "d_loss_real": metrics['d_loss_real'],
                            "d_loss_fake": metrics['d_loss_fake'],
                            "g_loss": metrics['g_loss'],
                            "c_loss": metrics['c_loss'],
                            "reward": metrics['reward']
                        }
                        
                        self.wandb.log(wandb_metrics)
                    
                    # Generate and visualize sample trajectories
                    if hasattr(self, 'sample_trajectories_for_wandb'):
                        self.sample_trajectories_for_wandb(epoch)
                
                # Track early stopping
                improved = early_stopping.on_epoch_end(epoch, metrics)
                
                # Save the model periodically
                if epoch % 50 == 0 and epoch > 0:
                    self.save_checkpoint(epoch)
                
                # Save best model based on mean reward
                mean_reward = metrics['reward']
                mean_reward_history.append(mean_reward)
                
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_epoch = epoch
                    if save_best:
                        # Save best model
                        self.save_checkpoint(f"best_{epoch}")
                
                # Check for early stopping
                if early_stopping.should_stop:
                    print(f"\nEarly stopping triggered after {early_stopping.wait_count} epochs without improvement")
                    print(f"Best model was at epoch {early_stopping.best_epoch} with reward {early_stopping.best_reward:.4f}")
                    break
                
                # Additional stopping criteria: check for NaN losses
                if np.isnan(metrics['g_loss']) or np.isnan(metrics['d_loss_real']) or np.isnan(metrics['d_loss_fake']):
                    print("NaN loss detected, stopping training")
                    break
                
                # Check for excessive loss values that indicate instability
                if metrics['g_loss'] > 1000 or metrics['d_loss_real'] > 1000 or metrics['d_loss_fake'] > 1000:
                    print("High loss values detected, might indicate training instability")
                    # Reduce learning rate if high losses are detected consistently
                    if epoch > 10 and np.mean(mean_reward_history[-10:]) < np.mean(mean_reward_history[-20:-10]):
                        print("Reducing learning rate to stabilize training")
                        self.actor_optimizer.learning_rate = self.actor_optimizer.learning_rate * 0.5
                        self.critic_optimizer.learning_rate = self.critic_optimizer.learning_rate * 0.5
                        self.discriminator_optimizer.learning_rate = self.discriminator_optimizer.learning_rate * 0.5
                        print(f"New learning rate: {self.actor_optimizer.learning_rate.numpy()}")
            
            except Exception as e:
                print(f"Error during epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing to next epoch...")
                continue
        
        # Save final model
        self.save_checkpoint("final")
        
        # Return best epoch and reward
        if self.wandb:
            self.wandb.log({
                "training_complete": True,
                "best_epoch": best_epoch,
                "best_reward": best_reward
            })
        
        print(f"\nTraining completed. Best model was at epoch {early_stopping.best_epoch} "
              f"with reward {early_stopping.best_reward:.4f}")
        
        return early_stopping.best_epoch, early_stopping.best_reward
    
    except Exception as e:
        print(f"Critical error during training: {e}")
        import traceback
        traceback.print_exc()
        # Try to save a checkpoint even if training fails
        try:
            self.save_checkpoint("emergency_save")
            print("Emergency checkpoint saved")
        except:
            print("Failed to save emergency checkpoint")
        
        # Return default values
        return 0, 0.0

def main():
    # Initialize wandb
    wandb.init(
        project="location",
        entity="xutao-henry-mao-vanderbilt-university",
        config={
            "architecture": "RL_Enhanced_Transformer_TrajGAN",
            "epochs": 500,
            "batch_size": 64,
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
    
    # Set wandb instance for the model
    model.set_wandb(wandb)
    
    # Training parameters
    epochs = 500
    batch_size = 64
    sample_interval = 5
    
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