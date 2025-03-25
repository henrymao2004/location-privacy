import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import TransformerBlock

class TULClassifier(nn.Module):
    def __init__(self, max_length, vocab_size, num_users):
        super(TULClassifier, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.num_users = num_users
        
        # Transformer layers for trajectory encoding
        self.transformer1 = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.1)
        self.transformer2 = TransformerBlock(embed_dim=128, num_heads=4, ff_dim=256, rate=0.1)
        
        # Global average pooling
        self.avg_pool = lambda x: torch.mean(x, dim=1)
        
        # Dense layers for trajectory features
        self.dense1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(128, 64)
        
        # User embedding
        self.user_embedding = nn.Linear(1, 32)
        
        # Output layer
        self.output = nn.Linear(96, 1)  # 64 + 32 = 96
        self.sigmoid = nn.Sigmoid()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
    def forward(self, trajectory, user):
        # Transformer layers
        x = self.transformer1(trajectory)
        x = self.transformer2(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        
        # Dense layers for trajectory features
        x = F.relu(self.dense1(x))
        x = self.dropout1(x)
        x = F.relu(self.dense2(x))
        
        # User embedding
        user_embed = F.relu(self.user_embedding(user))
        
        # Combine trajectory and user features
        combined = torch.cat([x, user_embed], dim=1)
        
        # Output layer
        output = self.sigmoid(self.output(combined))
        return output
    
    def train(self, trajectories, users, labels, epochs=50, batch_size=32, validation_split=0.2):
        """Train the TUL classifier"""
        # Convert inputs to PyTorch tensors
        trajectories = torch.tensor(trajectories, dtype=torch.float32)
        users = torch.tensor(users, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        # Split into train and validation sets
        n_samples = len(trajectories)
        n_train = int(n_samples * (1 - validation_split))
        indices = torch.randperm(n_samples)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        train_trajectories = trajectories[train_indices]
        train_users = users[train_indices]
        train_labels = labels[train_indices]
        
        val_trajectories = trajectories[val_indices]
        val_users = users[val_indices]
        val_labels = labels[val_indices]
        
        # Training loop
        for epoch in range(epochs):
            self.train()  # Set to training mode
            total_loss = 0
            n_batches = 0
            
            # Training
            for i in range(0, len(train_trajectories), batch_size):
                batch_trajectories = train_trajectories[i:i+batch_size]
                batch_users = train_users[i:i+batch_size]
                batch_labels = train_labels[i:i+batch_size]
                
                self.optimizer.zero_grad()
                outputs = self(batch_trajectories, batch_users)
                loss = F.binary_cross_entropy(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            # Validation
            self.eval()  # Set to evaluation mode
            with torch.no_grad():
                val_outputs = self(val_trajectories, val_users)
                val_loss = F.binary_cross_entropy(val_outputs, val_labels)
                val_accuracy = torch.mean((val_outputs > 0.5).float() == val_labels)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Training Loss: {total_loss/n_batches:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_accuracy:.4f}')
    
    def predict(self, trajectory, user):
        """Predict linkage probability between trajectory and user"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
            user = torch.tensor(user, dtype=torch.float32)
            output = self(trajectory, user)
            return output.numpy() 