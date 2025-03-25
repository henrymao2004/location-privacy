import torch
import numpy as np
import os
from datetime import datetime
from torch.utils.data import DataLoader

class PPOTrainer:
    def __init__(self, model, tul_classifier, train_dataset, val_dataset, config, device):
        self.model = model
        self.tul_classifier = tul_classifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.device = device
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
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
        self.model.train()
        epoch_metrics = {
            'generator_loss': [],
            'critic_loss': [],
            'discriminator_loss': [],
            'rewards_mean': [],
            'advantages_mean': []
        }
        
        for batch in self.train_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
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
        self.model.eval()
        val_metrics = {
            'rewards_mean': [],
            'generator_loss': [],
            'critic_loss': [],
            'discriminator_loss': []
        }
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Generate trajectories
                noise = torch.randn(self.config['batch_size'], self.model.latent_dim, device=self.device)
                gen_trajs = self.model.generator([*batch[:4], batch[4], noise])
                
                # Compute rewards
                rewards, _ = self.model._compute_rewards(batch, gen_trajs, self.tul_classifier)
                
                # Update metrics
                val_metrics['rewards_mean'].append(torch.mean(rewards).item())
                
                # Compute losses
                metrics = self.model.train_step(batch, self.config['batch_size'])
                for key in val_metrics:
                    if key in metrics:
                        val_metrics[key].append(metrics[key])
        
        return {k: np.mean(v) for k, v in val_metrics.items()}
    
    def _update_history(self, train_metrics, val_metrics):
        """Update training history"""
        for key in train_metrics:
            self.history[key].append(train_metrics[key])
        self.history['val_rewards'].append(val_metrics['rewards_mean'])
    
    def _check_early_stopping(self, val_reward):
        """Check if early stopping should be triggered"""
        if val_reward > self.best_val_reward + self.config['early_stopping']['min_delta']:
            self.best_val_reward = val_reward
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config['early_stopping']['patience']:
                return True
        return False
    
    def _save_checkpoint(self, epoch, save_dir):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'best_val_reward': self.best_val_reward,
            'history': self.history
        }
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    def _print_progress(self, epoch, train_metrics, val_metrics):
        """Print training progress"""
        print(f"\nEpoch {epoch + 1}")
        print("Training metrics:")
        for key, value in train_metrics.items():
            print(f"{key}: {value:.4f}")
        print("\nValidation metrics:")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}") 