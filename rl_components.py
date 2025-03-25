import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformer import TransformerBlock

class CriticNetwork(nn.Module):
    def __init__(self, max_length, vocab_size):
        super(CriticNetwork, self).__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # Transformer
        self.transformer = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)
        
        # Global average pooling
        self.avg_pool = lambda x: torch.mean(x, dim=1)
        
        # Value prediction
        self.value_head = nn.Linear(100, 1)
        
    def forward(self, x):
        # Transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        
        # Value prediction
        value = self.value_head(x)
        return value

class RewardFunction:
    def __init__(self, discriminator, tul_classifier, w1=0.3, w2=0.4, w3=0.3):
        self.discriminator = discriminator
        self.tul_classifier = tul_classifier
        self.w1 = w1  # Weight for adversarial reward
        self.w2 = w2  # Weight for utility reward
        self.w3 = w3  # Weight for privacy reward
        
    def compute_adversarial_reward(self, generated_traj):
        """Compute R_adv based on discriminator output"""
        with torch.no_grad():
            d_score = self.discriminator(generated_traj)
        return torch.log(d_score + 1e-8)  # Add small epsilon to avoid log(0)
    
    def compute_utility_reward(self, generated_traj, real_traj):
        """Compute R_util based on spatial, temporal, and categorical losses"""
        # Spatial loss (L2 distance)
        spatial_loss = torch.mean(torch.square(generated_traj[0] - real_traj[0]))
        
        # Temporal loss (cross-entropy on time distributions)
        temp_gen = torch.cat([generated_traj[2], generated_traj[3]], dim=-1)  # Combine day and hour
        temp_real = torch.cat([real_traj[2], real_traj[3]], dim=-1)
        temporal_loss = -torch.sum(temp_real * torch.log(temp_gen + 1e-8))
        
        # Categorical loss
        categorical_loss = -torch.sum(real_traj[1] * torch.log(generated_traj[1] + 1e-8))
        
        # Combine losses with weights
        beta, gamma, chi = 0.4, 0.3, 0.3
        utility_loss = -(beta * spatial_loss + gamma * temporal_loss + chi * categorical_loss)
        return utility_loss
    
    def compute_privacy_reward(self, generated_traj, real_user):
        """Compute R_priv based on TUL classifier output"""
        with torch.no_grad():
            link_prob = self.tul_classifier(generated_traj, real_user)
        return -link_prob  # Negative because we want to minimize linkage probability
    
    def compute_total_reward(self, generated_traj, real_traj, real_user):
        """Compute total reward as weighted sum of components"""
        r_adv = self.compute_adversarial_reward(generated_traj)
        r_util = self.compute_utility_reward(generated_traj, real_traj)
        r_priv = self.compute_privacy_reward(generated_traj, real_user)
        
        return self.w1 * r_adv + self.w2 * r_util + self.w3 * r_priv

class PPOAgent:
    def __init__(self, generator, critic, reward_function, epsilon=0.2):
        self.generator = generator
        self.critic = critic
        self.reward_function = reward_function
        self.epsilon = epsilon  # PPO clipping parameter
        
    def compute_advantages(self, values, rewards, gamma=0.99, lambda_=0.95):
        """Compute GAE (Generalized Advantage Estimation)"""
        returns = []
        gae = 0
        for r, v in zip(rewards.flip(0), values.flip(0)):
            delta = r + gamma * v - v
            gae = delta + gamma * lambda_ * gae
            returns.insert(0, gae + v)
        return torch.stack(returns) - values
    
    def update_policy(self, states, actions, old_probs, advantages):
        """Update policy using PPO clipped objective"""
        # Forward pass
        new_probs = self.generator(states)
        ratio = new_probs / (old_probs + 1e-8)
        
        # PPO clipped objective
        clip_1 = ratio * advantages
        clip_2 = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
        loss = -torch.mean(torch.min(clip_1, clip_2))
        
        # Backward pass
        self.generator.optimizer.zero_grad()
        loss.backward()
        self.generator.optimizer.step()
        
        return loss.item() 