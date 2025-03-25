import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from rl_components import CriticNetwork, RewardFunction, PPOAgent
from transformer import TransformerBlock

random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)

class TrajGAN(nn.Module):
    def __init__(self, latent_dim, keys, vocab_size, max_length, lat_centroid, lon_centroid, scale_factor, tul_classifier=None):
        super(TrajGAN, self).__init__()
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.keys = keys
        self.vocab_size = vocab_size
        self.lat_centroid = lat_centroid
        self.lon_centroid = lon_centroid
        self.scale_factor = scale_factor
        self.x_train = None
        
        # Define the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, betas=(0.5, 0.999))

        # Build the trajectory generator
        self.generator = self.build_generator()

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()

        # Build the critic network for RL
        self.critic = CriticNetwork(max_length, vocab_size)
        
        # Initialize reward function with TUL classifier
        self.reward_function = RewardFunction(self.discriminator, tul_classifier)
        
        # Initialize PPO agent
        self.ppo_agent = PPOAgent(self.generator, self.critic, self.reward_function)

    def build_discriminator(self):
        class Discriminator(nn.Module):
            def __init__(self, max_length, vocab_size, keys):
                super(Discriminator, self).__init__()
                self.max_length = max_length
                self.vocab_size = vocab_size
                self.keys = keys
                
                # Embedding layers
                self.embeddings = nn.ModuleDict()
                for key in keys:
                    if key == 'mask':
                        continue
                    if key == 'lat_lon':
                        self.embeddings[key] = nn.Linear(vocab_size[key], 64)
                    else:
                        self.embeddings[key] = nn.Linear(vocab_size[key], vocab_size[key])
                
                # Feature fusion
                self.fusion = nn.Linear(sum(vocab_size.values()), 100)
                
                # Transformer
                self.transformer = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)
                
                # Output
                self.output = nn.Linear(100, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, inputs):
                # Process each input type
                embeddings = []
                for idx, key in enumerate(self.keys):
                    if key == 'mask':
                        continue
                    x = inputs[idx]
                    if key == 'lat_lon':
                        x = self.embeddings[key](x)
                    else:
                        x = self.embeddings[key](x)
                    embeddings.append(x)
                
                # Concatenate embeddings
                x = torch.cat(embeddings, dim=-1)
                x = self.fusion(x)
                
                # Transformer
                x = self.transformer(x)
                
                # Global average pooling
                x = torch.mean(x, dim=1)
                
                # Output
                x = self.output(x)
                x = self.sigmoid(x)
                return x
        
        return Discriminator(self.max_length, self.vocab_size, self.keys)

    def build_generator(self):
        class Generator(nn.Module):
            def __init__(self, max_length, vocab_size, keys, latent_dim, scale_factor):
                super(Generator, self).__init__()
                self.max_length = max_length
                self.vocab_size = vocab_size
                self.keys = keys
                self.latent_dim = latent_dim
                self.scale_factor = scale_factor
                
                # Embedding layers
                self.embeddings = nn.ModuleDict()
                for key in keys:
                    if key == 'mask':
                        continue
                    if key == 'lat_lon':
                        self.embeddings[key] = nn.Linear(vocab_size[key], 64)
                    else:
                        self.embeddings[key] = nn.Linear(vocab_size[key], vocab_size[key])
                
                # Feature fusion
                self.fusion = nn.Linear(sum(vocab_size.values()) + latent_dim, 100)
                
                # Transformer
                self.transformer = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)
                
                # Output layers
                self.outputs = nn.ModuleDict()
                for key in keys:
                    if key == 'mask':
                        continue
                    if key == 'lat_lon':
                        self.outputs[key] = nn.Linear(100, 2)
                        self.outputs[key + '_scale'] = lambda x: x * scale_factor
                    else:
                        self.outputs[key] = nn.Linear(100, vocab_size[key])
            
            def forward(self, inputs):
                # Process each input type
                embeddings = []
                noise = inputs[-1]
                
                for idx, key in enumerate(self.keys):
                    if key == 'mask':
                        embeddings.append(inputs[idx])
                        continue
                    x = inputs[idx]
                    if key == 'lat_lon':
                        x = self.embeddings[key](x)
                    else:
                        x = self.embeddings[key](x)
                    embeddings.append(x)
                
                # Concatenate embeddings with noise
                x = torch.cat(embeddings + [noise], dim=-1)
                x = self.fusion(x)
                
                # Transformer
                x = self.transformer(x)
                
                # Generate outputs
                outputs = []
                for key in self.keys:
                    if key == 'mask':
                        outputs.append(inputs[self.keys.index(key)])
                        continue
                    if key == 'lat_lon':
                        x_latlon = self.outputs[key](x)
                        x_latlon = self.outputs[key + '_scale'](x_latlon)
                        outputs.append(x_latlon)
                    else:
                        x_attr = self.outputs[key](x)
                        x_attr = F.softmax(x_attr, dim=-1)
                        outputs.append(x_attr)
                
                return outputs
        
        return Generator(self.max_length, self.vocab_size, self.keys, self.latent_dim, self.scale_factor)

    def train(self, epochs=200, batch_size=256, sample_interval=10, rl_update_interval=5):
        # Training data
        x_train = np.load('data/final_train.npy', allow_pickle=True)
        self.x_train = x_train

        # Padding zero to reach the maxlength
        X_train = [torch.tensor(pad_sequences(f, self.max_length, padding='pre', dtype='float64')) for f in x_train]
        self.X_train = X_train
        
        for epoch in range(1, epochs+1):
            # Select a random batch of real trajectories
            idx = np.random.randint(0, X_train[0].shape[0], batch_size)
            
            # Ground truths for real trajectories and synthetic trajectories
            real_bc = torch.ones((batch_size, 1))
            syn_bc = torch.zeros((batch_size, 1))

            real_trajs_bc = []
            real_trajs_bc.append(X_train[0][idx])  # latlon
            real_trajs_bc.append(X_train[1][idx])  # day
            real_trajs_bc.append(X_train[2][idx])  # hour
            real_trajs_bc.append(X_train[3][idx])  # category
            real_trajs_bc.append(X_train[4][idx])  # mask
            noise = torch.randn(batch_size, self.latent_dim)
            real_trajs_bc.append(noise)  # noise

            # Generate a batch of synthetic trajectories
            gen_trajs_bc = self.generator(real_trajs_bc)

            # Train the discriminator
            self.discriminator.optimizer.zero_grad()
            d_loss_real = F.binary_cross_entropy(self.discriminator(real_trajs_bc[:4]), real_bc)
            d_loss_syn = F.binary_cross_entropy(self.discriminator(gen_trajs_bc[:4]), syn_bc)
            d_loss = 0.5 * (d_loss_real + d_loss_syn)
            d_loss.backward()
            self.discriminator.optimizer.step()

            # Train the generator with GAN objective
            self.generator.optimizer.zero_grad()
            noise = torch.randn(batch_size, self.latent_dim)
            real_trajs_bc[5] = noise
            g_loss = self.combined_loss(real_trajs_bc, gen_trajs_bc)
            g_loss.backward()
            self.generator.optimizer.step()
            
            # RL updates every rl_update_interval epochs
            if epoch % rl_update_interval == 0:
                # Get current policy probabilities
                old_probs = self.generator(real_trajs_bc)
                
                # Generate trajectories and compute rewards
                gen_trajs = self.generator(real_trajs_bc)
                rewards = []
                for i in range(batch_size):
                    reward = self.reward_function.compute_total_reward(
                        [t[i].detach().numpy() for t in gen_trajs], 
                        [t[i].detach().numpy() for t in real_trajs_bc[:4]], 
                        idx[i]  # User ID for privacy reward
                    )
                    rewards.append(reward)
                rewards = torch.tensor(rewards)
                
                # Get value estimates from critic
                values = self.critic(gen_trajs)
                
                # Compute advantages
                advantages = self.ppo_agent.compute_advantages(values, rewards)
                
                # Update policy using PPO
                ppo_loss = self.ppo_agent.update_policy(real_trajs_bc, gen_trajs, old_probs, advantages)
                
                # Update critic
                critic_loss = F.mse_loss(self.critic(gen_trajs), rewards)
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
                
                print(f"[{epoch}/{epochs}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f} | PPO Loss: {ppo_loss:.4f} | Critic Loss: {critic_loss.item():.4f}")
            else:
                print(f"[{epoch}/{epochs}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

            # Print and save the losses/params
            if epoch % sample_interval == 0:
                self.save_checkpoint(epoch)
                print('Model params saved to the disk.')

    def combined_loss(self, real_trajs, gen_trajs):
        # Implement the combined loss function from losses.py
        traj_length = torch.sum(real_trajs[4], dim=1)
        
        # BCE loss
        bce_loss = F.binary_cross_entropy(self.discriminator(gen_trajs[:4]), torch.ones_like(self.discriminator(gen_trajs[:4])))
        
        # Spatial loss (L2 distance)
        masked_latlon_full = torch.sum(torch.sum(
            torch.multiply(
                torch.multiply(
                    (gen_trajs[0] - real_trajs[0]),
                    (gen_trajs[0] - real_trajs[0])
                ),
                torch.cat([real_trajs[4] for _ in range(2)], dim=2)
            ),
            dim=1
        ), dim=1, keepdim=True)
        masked_latlon_mse = torch.sum(torch.div(masked_latlon_full, traj_length))
        
        # Categorical losses
        ce_category = F.cross_entropy(gen_trajs[1], torch.argmax(real_trajs[1], dim=-1))
        ce_day = F.cross_entropy(gen_trajs[2], torch.argmax(real_trajs[2], dim=-1))
        ce_hour = F.cross_entropy(gen_trajs[3], torch.argmax(real_trajs[3], dim=-1))
        
        # Combine losses with weights
        p_bce, p_latlon, p_cat, p_day, p_hour = 1, 10, 1, 1, 1
        return (bce_loss * p_bce + 
                masked_latlon_mse * p_latlon + 
                ce_category * p_cat + 
                ce_day * p_day + 
                ce_hour * p_hour)

    def save_checkpoint(self, epoch):
        # Save weights
        torch.save(self.generator.state_dict(), f"params/G_model_{epoch}.pt")
        torch.save(self.discriminator.state_dict(), f"params/D_model_{epoch}.pt")
        torch.save(self.critic.state_dict(), f"params/Critic_model_{epoch}.pt")