import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Generator(nn.Module):
    def __init__(self, latent_dim, vocab_size, max_length, num_heads=8, num_layers=4, d_model=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.max_length = max_length
        
        # Project latent vector to sequence length
        self.latent_proj = nn.Linear(latent_dim, max_length * d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projections for each feature type
        self.lat_lon_proj = nn.Linear(d_model, vocab_size['lat_lon'])
        self.day_proj = nn.Linear(d_model, vocab_size['day'])
        self.hour_proj = nn.Linear(d_model, vocab_size['hour'])
        self.category_proj = nn.Linear(d_model, vocab_size['category'])
        self.mask_proj = nn.Linear(d_model, vocab_size['mask'])
        
        # Store internal state for RL
        self.current_state = None

    def forward(self, z, return_state=False):
        # Project and reshape latent vector
        x = self.latent_proj(z)
        x = x.view(-1, self.max_length, self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Store internal state for RL
        self.current_state = x if return_state else None
        
        # Project to feature spaces
        lat_lon = torch.tanh(self.lat_lon_proj(x))  # Normalized coordinates
        day = F.softmax(self.day_proj(x), dim=-1)
        hour = F.softmax(self.hour_proj(x), dim=-1)
        category = F.softmax(self.category_proj(x), dim=-1)
        mask = torch.sigmoid(self.mask_proj(x))
        
        outputs = {
            'lat_lon': lat_lon,
            'day': day,
            'hour': hour,
            'category': category,
            'mask': mask
        }
        
        if return_state:
            return outputs, self.current_state
        return outputs

class Discriminator(nn.Module):
    def __init__(self, vocab_size, max_length, num_heads=8, num_layers=4, d_model=256):
        super().__init__()
        self.d_model = d_model
        
        # Feature embeddings
        self.lat_lon_embed = nn.Linear(vocab_size['lat_lon'], d_model)
        self.day_embed = nn.Linear(vocab_size['day'], d_model)
        self.hour_embed = nn.Linear(vocab_size['hour'], d_model)
        self.category_embed = nn.Linear(vocab_size['category'], d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global average pooling and output
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        # Embed features
        lat_lon = self.lat_lon_embed(x['lat_lon'])
        day = self.day_embed(x['day'])
        hour = self.hour_embed(x['hour'])
        category = self.category_embed(x['category'])
        
        # Combine embeddings
        combined = lat_lon + day + hour + category
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # Transformer encoding
        features = self.transformer_encoder(combined)
        
        # Global pooling and output
        pooled = self.global_pool(features.transpose(1, 2)).squeeze(-1)
        output = torch.sigmoid(self.output(pooled))
        
        return output
    
    def get_features(self, x):
        # Embed features
        lat_lon = self.lat_lon_embed(x['lat_lon'])
        day = self.day_embed(x['day'])
        hour = self.hour_embed(x['hour'])
        category = self.category_embed(x['category'])
        
        # Combine embeddings
        combined = lat_lon + day + hour + category
        
        # Add positional encoding
        combined = self.pos_encoder(combined)
        
        # Get transformer features
        features = self.transformer_encoder(combined)
        
        return features

class Critic(nn.Module):
    def __init__(self, vocab_size, max_length, num_heads=8, num_layers=4, d_model=256):
        super().__init__()
        self.discriminator = Discriminator(vocab_size, max_length, num_heads, num_layers, d_model)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        features = self.discriminator.get_features(x)
        pooled = F.adaptive_avg_pool1d(features.transpose(1, 2), 1).squeeze(-1)
        value = self.value_head(pooled)
        return value

class Transformer_TrajGAN(nn.Module):
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, 
                 scale_factor, w1=1.0, w2=1.0, w3=1.0, beta=1.0, gamma=1.0, chi=1.0, 
                 alpha=1.0, lr=0.0001, beta1=0.5, epsilon=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        self.keys = keys
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        
        # Reward weights
        self.w1 = w1  # Adversarial reward weight
        self.w2 = w2  # Utility reward weight
        self.w3 = w3  # Privacy reward weight
        self.beta = beta  # Spatial loss weight
        self.gamma = gamma  # Temporal loss weight
        self.chi = chi  # Categorical loss weight
        self.alpha = alpha  # Privacy penalty weight
        self.epsilon = epsilon  # PPO clip parameter
        
        # Initialize networks
        self.generator = Generator(latent_dim, vocab_size, max_length)
        self.discriminator = Discriminator(vocab_size, max_length)
        self.critic = Critic(vocab_size, max_length)
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(beta1, 0.999))
        
        self.optimizer = self.g_optimizer  # For checkpoint saving

    def compute_spatial_loss(self, gen_traj, real_traj):
        """Compute L2 distance between coordinate pairs"""
        return torch.mean(torch.norm(gen_traj['lat_lon'] - real_traj['lat_lon'], dim=-1) ** 2)

    def compute_temporal_loss(self, gen_traj, real_traj):
        """Compute cross-entropy between temporal distributions"""
        # Day distribution loss
        day_loss = F.cross_entropy(gen_traj['day'].reshape(-1, self.vocab_size['day']), 
                                 real_traj['day'].reshape(-1, self.vocab_size['day']))
        # Hour distribution loss
        hour_loss = F.cross_entropy(gen_traj['hour'].reshape(-1, self.vocab_size['hour']), 
                                  real_traj['hour'].reshape(-1, self.vocab_size['hour']))
        return day_loss + hour_loss

    def compute_categorical_loss(self, gen_traj, real_traj):
        """Compute cross-entropy between category distributions"""
        return F.cross_entropy(gen_traj['category'].reshape(-1, self.vocab_size['category']), 
                             real_traj['category'].reshape(-1, self.vocab_size['category']))

    def compute_utility_reward(self, gen_traj, real_traj):
        """Compute utility reward based on spatial, temporal, and categorical losses"""
        L_s = self.compute_spatial_loss(gen_traj, real_traj)
        L_t = self.compute_temporal_loss(gen_traj, real_traj)
        L_c = self.compute_categorical_loss(gen_traj, real_traj)
        
        return -(self.beta * L_s + self.gamma * L_t + self.chi * L_c)

    def compute_privacy_reward(self, gen_traj, tul_classifier):
        """Compute privacy reward based on TUL classifier"""
        with torch.no_grad():
            tul_pred = tul_classifier(gen_traj)
            # Use negative TUL confidence as privacy reward
            privacy_reward = -self.alpha * torch.max(tul_pred, dim=1)[0]
        return privacy_reward

    def _compute_rewards(self, real_data, gen_data, tul_classifier):
        # Adversarial reward (equation 2)
        d_fake = self.discriminator(gen_data)
        adv_reward = torch.log(d_fake + 1e-8)
        
        # Utility reward (equation 3)
        util_reward = self.compute_utility_reward(gen_data, real_data)
        
        # Privacy reward (equation 7)
        priv_reward = self.compute_privacy_reward(gen_data, tul_classifier)
        
        # Combine rewards (equation 1)
        total_reward = self.w1 * adv_reward + self.w2 * util_reward + self.w3 * priv_reward
        
        # Compute advantage for PPO
        value = self.critic(gen_data)
        advantage = total_reward - value.detach()
        
        return total_reward, advantage

    def train_step(self, batch, batch_size):
        metrics = {}
        
        # Generate trajectories with states
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        gen_data, states = self.generator(noise, return_state=True)
        
        # Train discriminator (equation 11)
        self.d_optimizer.zero_grad()
        d_real = self.discriminator(batch)
        d_fake = self.discriminator(gen_data.detach())
        d_loss = -(torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8)))
        d_loss.backward()
        self.d_optimizer.step()
        metrics['discriminator_loss'] = d_loss.item()
        
        # Compute rewards and advantages
        rewards, advantages = self._compute_rewards(batch, gen_data, self.tul_classifier)
        
        # Train critic (equation 9)
        self.c_optimizer.zero_grad()
        values = self.critic(gen_data.detach())
        c_loss = F.mse_loss(values, rewards.detach())
        c_loss.backward()
        self.c_optimizer.step()
        metrics['critic_loss'] = c_loss.item()
        
        # Train generator with PPO (equation 10)
        self.g_optimizer.zero_grad()
        log_probs = torch.log(self.discriminator(gen_data) + 1e-8)
        ratio = torch.exp(log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        g_loss = -torch.min(surr1, surr2).mean()
        g_loss.backward()
        self.g_optimizer.step()
        metrics['generator_loss'] = g_loss.item()
        
        # Record metrics
        metrics['rewards_mean'] = rewards.mean().item()
        metrics['advantages_mean'] = advantages.mean().item()
        metrics['spatial_loss'] = self.compute_spatial_loss(gen_data, batch).item()
        metrics['temporal_loss'] = self.compute_temporal_loss(gen_data, batch).item()
        metrics['categorical_loss'] = self.compute_categorical_loss(gen_data, batch).item()
        
        return metrics 