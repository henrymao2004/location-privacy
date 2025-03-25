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
    def __init__(self, discriminator, tul_classifier):
        self.discriminator = discriminator
        self.tul_classifier = tul_classifier
        
        # Configurable weights for reward components
        self.w1 = 1.0  # Adversarial weight
        self.w2 = 1.0  # Utility weight
        self.w3 = 1.0  # Privacy weight
        
        # Weights for utility components
        self.beta = 1.0   # Spatial loss weight
        self.gamma = 1.0  # Temporal loss weight
        self.chi = 1.0    # Categorical loss weight
        
        # Privacy weight
        self.alpha = 1.0  # Privacy penalty weight
    
    def compute_adversarial_reward(self, gen_traj):
        """Compute adversarial reward based on discriminator's evaluation"""
        # Convert to tensor if needed
        if not isinstance(gen_traj, torch.Tensor):
            gen_traj = torch.tensor(gen_traj, dtype=torch.float32)
        
        # Get discriminator's probability
        d_prob = self.discriminator(gen_traj)
        
        # Compute log probability as reward
        adv_reward = torch.log(d_prob + 1e-8)  # Add small epsilon to avoid log(0)
        return adv_reward
    
    def compute_utility_reward(self, gen_traj, real_traj):
        """Compute utility reward based on spatial, temporal, and categorical losses"""
        # Convert to tensors if needed
        if not isinstance(gen_traj, torch.Tensor):
            gen_traj = torch.tensor(gen_traj, dtype=torch.float32)
        if not isinstance(real_traj, torch.Tensor):
            real_traj = torch.tensor(real_traj, dtype=torch.float32)
        
        # Spatial loss (L2 distance)
        spatial_loss = F.mse_loss(gen_traj[0], real_traj[0])  # L2 distance for coordinates
        
        # Temporal loss (cross-entropy for day and hour)
        day_loss = F.cross_entropy(gen_traj[1], torch.argmax(real_traj[1], dim=-1))
        hour_loss = F.cross_entropy(gen_traj[2], torch.argmax(real_traj[2], dim=-1))
        temporal_loss = day_loss + hour_loss
        
        # Categorical loss (cross-entropy for POI categories)
        categorical_loss = F.cross_entropy(gen_traj[3], torch.argmax(real_traj[3], dim=-1))
        
        # Combine losses with weights
        total_loss = (self.beta * spatial_loss + 
                     self.gamma * temporal_loss + 
                     self.chi * categorical_loss)
        
        # Convert loss to reward (negative loss)
        util_reward = -total_loss
        return util_reward
    
    def compute_privacy_reward(self, gen_traj, user_id):
        """Compute privacy reward based on TUL classifier's accuracy"""
        # Convert to tensor if needed
        if not isinstance(gen_traj, torch.Tensor):
            gen_traj = torch.tensor(gen_traj, dtype=torch.float32)
        if not isinstance(user_id, torch.Tensor):
            user_id = torch.tensor(user_id, dtype=torch.float32)
        
        # Get TUL classifier's probability for the correct user
        with torch.no_grad():
            link_prob = self.tul_classifier(gen_traj, user_id)
        
        # Compute privacy reward (negative probability)
        priv_reward = -self.alpha * link_prob
        return priv_reward
    
    def compute_total_reward(self, gen_traj, real_traj, user_id):
        """Compute total reward combining all components"""
        # Compute individual rewards
        adv_reward = self.compute_adversarial_reward(gen_traj)
        util_reward = self.compute_utility_reward(gen_traj, real_traj)
        priv_reward = self.compute_privacy_reward(gen_traj, user_id)
        
        # Combine rewards with weights
        total_reward = (self.w1 * adv_reward + 
                       self.w2 * util_reward + 
                       self.w3 * priv_reward)
        
        return total_reward

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