import tensorflow as tf
from keras.losses import binary_crossentropy
import keras
import numpy as np

# BCE loss for the discriminator
def d_bce_loss(y_true, y_pred):
    d_bce_loss = binary_crossentropy(y_true, y_pred)
    return d_bce_loss

# Original TrajLoss for the GAN generator
def trajLoss(real_traj, gen_traj):
    def loss(y_true, y_pred):
        traj_length = keras.backend.sum(real_traj[4],axis=1)
        
        bce_loss = binary_crossentropy(y_true, y_pred)
        
        masked_latlon_full = keras.backend.sum(keras.backend.sum(tf.multiply(tf.multiply((gen_traj[0]-real_traj[0]),(gen_traj[0]-real_traj[0])),tf.concat([real_traj[4] for x in range(2)],axis=2)),axis=1),axis=1,keepdims=True)
        masked_latlon_mse = keras.backend.sum(tf.math.divide(masked_latlon_full,traj_length))
        
        ce_category = tf.nn.softmax_cross_entropy_with_logits_v2(gen_traj[1],real_traj[1])
        ce_day = tf.nn.softmax_cross_entropy_with_logits_v2(gen_traj[2],real_traj[2])
        ce_hour = tf.nn.softmax_cross_entropy_with_logits_v2(gen_traj[3],real_traj[3])
        
        ce_category_masked = tf.multiply(ce_category,keras.backend.sum(real_traj[4],axis=2))
        ce_day_masked = tf.multiply(ce_day,keras.backend.sum(real_traj[4],axis=2))
        ce_hour_masked = tf.multiply(ce_hour,keras.backend.sum(real_traj[4],axis=2))
        
        ce_category_mean = keras.backend.sum(tf.math.divide(ce_category_masked,traj_length))
        ce_day_mean = keras.backend.sum(tf.math.divide(ce_day_masked,traj_length))
        ce_hour_mean = keras.backend.sum(tf.math.divide(ce_hour_masked,traj_length))
        
        p_bce = 1
        p_latlon = 10
        p_cat = 1
        p_day = 1
        p_hour = 1
        
        return bce_loss*p_bce + masked_latlon_mse*p_latlon + ce_category_mean*p_cat + ce_day_mean*p_day + ce_hour_mean*p_hour

    return loss

# RL-based reward functions

# Privacy reward based on TUL task
def privacy_reward(tul_output, user_ids):
    """
    Compute privacy reward based on TUL model output
    
    Args:
        tul_output: Output probabilities from TUL model [batch_size, num_users]
        user_ids: True user IDs [batch_size]
        
    Returns:
        Privacy reward: -log(p(u_i|T_i)) for each trajectory
    """
    batch_size = tul_output.shape[0]
    privacy_rewards = np.zeros((batch_size,))
    
    for i in range(batch_size):
        # Get probability of correctly identifying the user
        user_id = user_ids[i]
        p_tul = tul_output[i, user_id]
        # Add small epsilon to avoid log(0)
        privacy_rewards[i] = -np.log(p_tul + 1e-10)
    
    return privacy_rewards

# Utility reward measuring spatial, temporal, and semantic similarity
def utility_reward(real_trajs, gen_trajs, mask):
    """
    Compute utility reward based on trajectory similarity
    
    Args:
        real_trajs: List of real trajectory features [latlon, day, hour, category]
        gen_trajs: List of generated trajectory features [latlon, day, hour, category]
        mask: Mask indicating valid trajectory points
        
    Returns:
        Utility reward: Negative distance between real and generated trajectories
    """
    batch_size = real_trajs[0].shape[0]
    utility_rewards = np.zeros((batch_size,))
    
    # Weights for different distance components
    w_spatial = 0.7
    w_temporal = 0.15
    w_semantic = 0.15
    
    for i in range(batch_size):
        # Extract valid points using mask
        valid_mask = mask[i].reshape(-1) > 0
        
        if np.sum(valid_mask) > 0:
            # Spatial distance (lat/lon)
            real_latlon = real_trajs[0][i][valid_mask]
            gen_latlon = gen_trajs[0][i][valid_mask]
            spatial_dist = np.mean(np.sqrt(np.sum((real_latlon - gen_latlon) ** 2, axis=1)))
            
            # Temporal distance (day, hour)
            temporal_dist = 0
            for j in range(1, 3):  # day, hour
                real_temp = real_trajs[j][i][valid_mask]
                gen_temp = gen_trajs[j][i][valid_mask]
                
                # Calculate Jensen-Shannon divergence
                real_dist = np.mean(real_temp, axis=0)
                gen_dist = np.mean(gen_temp, axis=0)
                
                # Normalize
                real_dist = real_dist / (np.sum(real_dist) + 1e-10)
                gen_dist = gen_dist / (np.sum(gen_dist) + 1e-10)
                
                # JS divergence
                m = 0.5 * (real_dist + gen_dist)
                js_div = 0.5 * (np.sum(real_dist * np.log((real_dist + 1e-10) / (m + 1e-10))) + 
                               np.sum(gen_dist * np.log((gen_dist + 1e-10) / (m + 1e-10))))
                
                temporal_dist += js_div
            temporal_dist /= 2  # Average over day and hour
            
            # Semantic distance (category)
            real_cat = real_trajs[3][i][valid_mask]
            gen_cat = gen_trajs[3][i][valid_mask]
            
            # Calculate JS divergence for category
            real_dist = np.mean(real_cat, axis=0)
            gen_dist = np.mean(gen_cat, axis=0)
            
            # Normalize
            real_dist = real_dist / (np.sum(real_dist) + 1e-10)
            gen_dist = gen_dist / (np.sum(gen_dist) + 1e-10)
            
            # JS divergence
            m = 0.5 * (real_dist + gen_dist)
            semantic_dist = 0.5 * (np.sum(real_dist * np.log((real_dist + 1e-10) / (m + 1e-10))) + 
                                  np.sum(gen_dist * np.log((gen_dist + 1e-10) / (m + 1e-10))))
            
            # Combine distances with weights
            total_dist = w_spatial * spatial_dist + w_temporal * temporal_dist + w_semantic * semantic_dist
            
            # Utility reward is negative distance (higher is better)
            utility_rewards[i] = -total_dist
    
    return utility_rewards

# Adversarial reward from discriminator
def adversarial_reward(disc_output):
    """
    Compute adversarial reward based on discriminator output
    
    Args:
        disc_output: Discriminator output probabilities [batch_size, 1]
        
    Returns:
        Adversarial reward: log(D(G(z))) for each trajectory
    """
    # Add small epsilon to avoid log(0)
    return np.log(disc_output + 1e-10).flatten()

# Combined reward function
def combined_reward(real_trajs, gen_trajs, disc_output, tul_output, user_ids, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Compute combined reward using weighted sum of privacy, utility, and adversarial rewards
    
    Args:
        real_trajs: List of real trajectory features
        gen_trajs: List of generated trajectory features
        disc_output: Discriminator output probabilities
        tul_output: TUL model output probabilities
        user_ids: True user IDs
        alpha: Privacy reward weight
        beta: Utility reward weight
        gamma: Adversarial reward weight
        
    Returns:
        Combined reward for each trajectory
    """
    # Get mask from real trajectories
    mask = real_trajs[4]
    
    # Compute individual rewards
    priv_reward = privacy_reward(tul_output, user_ids)
    util_reward = utility_reward(real_trajs, gen_trajs, mask)
    adv_reward = adversarial_reward(disc_output)
    
    # Combine rewards with weights
    combined = alpha * priv_reward + beta * util_reward + gamma * adv_reward
    
    return combined

# PPO clipped surrogate loss
def ppo_loss(advantages, old_probs, new_probs, epsilon=0.2):
    """
    Compute PPO clipped surrogate loss
    
    Args:
        advantages: Advantage values (R - V)
        old_probs: Action probabilities from old policy
        new_probs: Action probabilities from current policy
        epsilon: Clipping parameter
        
    Returns:
        PPO loss value
    """
    # Compute probability ratio
    ratio = new_probs / (old_probs + 1e-10)
    
    # Clipped surrogate objective
    surrogate1 = ratio * advantages
    surrogate2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # Take minimum (pessimistic estimate)
    ppo_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
    
    return ppo_loss