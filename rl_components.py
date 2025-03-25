import tensorflow as tf
import keras
import numpy as np
from keras.layers import Input, Dense, LSTM, Lambda, TimeDistributed, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras.regularizers import l1

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class CriticNetwork:
    def __init__(self, max_length, vocab_size):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.optimizer = Adam(0.001)
        self.model = self.build_critic()
        
    def build_critic(self):
        # Input is the trajectory state (partial trajectory so far)
        traj_input = Input(shape=(self.max_length, sum(self.vocab_size.values())))
        
        # Transformer to process trajectory
        transformer_block = TransformerBlock(embed_dim=100, num_heads=4, ff_dim=200, rate=0.1)
        transformer_output = transformer_block(traj_input)
        
        # Global average pooling to reduce sequence dimension
        avg_pool = GlobalAveragePooling1D()(transformer_output)
        
        # Value prediction
        value = Dense(1, activation='linear')(avg_pool)
        
        model = Model(inputs=traj_input, outputs=value)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

class RewardFunction:
    def __init__(self, discriminator, tul_classifier, w1=0.3, w2=0.4, w3=0.3):
        self.discriminator = discriminator
        self.tul_classifier = tul_classifier
        self.w1 = w1  # Weight for adversarial reward
        self.w2 = w2  # Weight for utility reward
        self.w3 = w3  # Weight for privacy reward
        
    def compute_adversarial_reward(self, generated_traj):
        """Compute R_adv based on discriminator output"""
        d_score = self.discriminator.predict(generated_traj)
        return np.log(d_score + 1e-8)  # Add small epsilon to avoid log(0)
    
    def compute_utility_reward(self, generated_traj, real_traj):
        """Compute R_util based on spatial, temporal, and categorical losses"""
        # Spatial loss (L2 distance)
        spatial_loss = np.mean(np.square(generated_traj[0] - real_traj[0]))
        
        # Temporal loss (cross-entropy on time distributions)
        temp_gen = np.concatenate([generated_traj[2], generated_traj[3]], axis=-1)  # Combine day and hour
        temp_real = np.concatenate([real_traj[2], real_traj[3]], axis=-1)
        temporal_loss = -np.sum(temp_real * np.log(temp_gen + 1e-8))
        
        # Categorical loss
        categorical_loss = -np.sum(real_traj[1] * np.log(generated_traj[1] + 1e-8))
        
        # Combine losses with weights
        beta, gamma, chi = 0.4, 0.3, 0.3
        utility_loss = -(beta * spatial_loss + gamma * temporal_loss + chi * categorical_loss)
        return utility_loss
    
    def compute_privacy_reward(self, generated_traj, real_user):
        """Compute R_priv based on TUL classifier output"""
        link_prob = self.tul_classifier.predict([generated_traj, real_user])
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
        for r, v in zip(rewards[::-1], values[::-1]):
            delta = r + gamma * v - v
            gae = delta + gamma * lambda_ * gae
            returns.insert(0, gae + v)
        return np.array(returns) - values
    
    def update_policy(self, states, actions, old_probs, advantages):
        """Update policy using PPO clipped objective"""
        with tf.GradientTape() as tape:
            new_probs = self.generator(states)
            ratio = new_probs / (old_probs + 1e-8)
            
            # PPO clipped objective
            clip_1 = ratio * advantages
            clip_2 = tf.clip_by_value(ratio, 1-self.epsilon, 1+self.epsilon) * advantages
            loss = -tf.reduce_mean(tf.minimum(clip_1, clip_2))
            
        # Compute and apply gradients
        grads = tape.gradient(loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        
        return loss.numpy() 