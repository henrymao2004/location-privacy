import tensorflow as tf
import numpy as np
import os
from datetime import datetime

class PPOTrainer:
    def __init__(self, model, tul_classifier, train_dataset, val_dataset, config):
        self.model = model
        self.tul_classifier = tul_classifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Initialize metrics tracking
        self.history = {
            'generator_loss': [],
            'critic_loss': [],
            'discriminator_loss': [],
            'rewards_mean': [],
            'advantages_mean': [],
            'val_rewards': []
        }
        
        # Early stopping
        self.best_val_reward = float('-inf')
        self.patience_counter = 0
        
    def train(self, epochs, save_dir):
        """Train the model using PPO"""
        for epoch in range(epochs):
            # Training step
            train_metrics = self._train_epoch()
            
            # Validation step
            val_metrics = self._validate()
            
            # Update history
            self._update_history(train_metrics, val_metrics)
            
            # Early stopping check
            if self._check_early_stopping(val_metrics['rewards_mean']):
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Save checkpoint
            if val_metrics['rewards_mean'] > self.best_val_reward:
                self._save_checkpoint(epoch, save_dir)
            
            # Print progress
            self._print_progress(epoch, train_metrics, val_metrics)
        
        return self.history
    
    def _train_epoch(self):
        """Train for one epoch"""
        self.train_dataset.shuffle()
        epoch_metrics = {
            'generator_loss': [],
            'critic_loss': [],
            'discriminator_loss': [],
            'rewards_mean': [],
            'advantages_mean': []
        }
        
        for batch in self.train_dataset:
            # Training step
            metrics = self.model.train_step(batch, self.config['batch_size'])
            
            # Update metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key].append(metrics[key])
        
        # Average metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def _validate(self):
        """Validate the model"""
        val_metrics = {
            'rewards_mean': [],
            'generator_loss': [],
            'critic_loss': [],
            'discriminator_loss': []
        }
        
        for batch in self.val_dataset:
            # Generate trajectories
            noise = tf.random.normal([self.config['batch_size'], self.model.latent_dim])
            gen_trajs = self.model.generator([*batch[:4], batch[4], noise])
            
            # Compute rewards
            rewards, _ = self.model._compute_rewards(batch, gen_trajs, self.tul_classifier)
            
            # Update metrics
            val_metrics['rewards_mean'].append(tf.reduce_mean(rewards))
            
            # Compute losses
            metrics = self.model.train_step(batch, self.config['batch_size'])
            for key in val_metrics:
                if key in metrics:
                    val_metrics[key].append(metrics[key])
        
        return {k: np.mean(v) for k, v in val_metrics.items()}
    
    def _update_history(self, train_metrics, val_metrics):
        """Update training history"""
        for key in self.history:
            if key in train_metrics:
                self.history[key].append(train_metrics[key])
            elif key in val_metrics:
                self.history[key].append(val_metrics[key])
    
    def _check_early_stopping(self, val_reward):
        """Check if early stopping should be triggered"""
        if val_reward > self.best_val_reward:
            self.best_val_reward = val_reward
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config['early_stopping']['patience']:
                return True
        return False
    
    def _save_checkpoint(self, epoch, save_dir):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}_{timestamp}")
        self.model.save_checkpoint(epoch)
    
    def _print_progress(self, epoch, train_metrics, val_metrics):
        """Print training progress"""
        print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
        print("Training Metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")
        print("\nValidation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}") 