import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.spatial.distance import jensenshannon

class TrajRewardFunction:
    def __init__(self, tul_classifier=None, alpha=0.4, beta=0.4, gamma=0.2):
        """
        Initialize the trajectory reward function.
        
        Args:
            tul_classifier: TUL model used for privacy evaluation
            alpha: Weight for privacy reward component
            beta: Weight for utility reward component
            gamma: Weight for adversarial reward component
        """
        self.tul_classifier = tul_classifier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # Target metrics for adaptive reward balancing
        self.acc_at_1_target = 0.2  # Target re-identification accuracy (lower is better for privacy)
        self.fid_target = 20.0      # Target FID score for trajectory similarity
        self.fid_max = 100.0        # Maximum FID score for normalization
        
        # Current metrics (will be updated during training)
        self.current_acc_at_1 = 0.5  # Starting value
        self.current_fid = 50.0      # Starting value
    
    def compute_privacy_reward(self, generated_traj, user_id):
        """
        Compute privacy reward based on TUL model's ability to re-identify the user.
        Higher reward means better privacy (lower re-identification probability).
        
        Args:
            generated_traj: The generated trajectory
            user_id: True user ID
            
        Returns:
            privacy_reward: Scalar privacy reward
        """
        if self.tul_classifier is None:
            # If no TUL classifier is available, use a simple privacy heuristic
            # based on trajectory perturbation amount
            return tf.constant(0.5, dtype=tf.float32)
        
        # Get TUL model prediction
        tul_probs = self.tul_classifier.predict(generated_traj)
        
        # Extract probability for the true user
        if isinstance(user_id, (list, np.ndarray)):
            user_id = user_id[0]  # Take first element if it's batched
            
        true_user_prob = tul_probs[0, user_id]
        
        # Privacy reward is negative log probability of correct identification
        # Higher reward means lower identification probability (better privacy)
        privacy_reward = -tf.math.log(true_user_prob + 1e-8)
        
        # Normalize reward to reasonable range [0, 5]
        privacy_reward = tf.minimum(5.0, privacy_reward)
        
        return privacy_reward
    
    def compute_utility_reward(self, generated_traj, original_traj):
        """
        Compute utility reward based on spatial, temporal, and semantic similarity.
        
        Args:
            generated_traj: The generated trajectory
            original_traj: The original trajectory
            
        Returns:
            utility_reward: Scalar utility reward
        """
        # Extract components
        gen_latlon, gen_category, gen_time, gen_mask = (
            generated_traj[0], generated_traj[1], 
            generated_traj[2:4], generated_traj[4]
        )
        orig_latlon, orig_category, orig_time, orig_mask = (
            original_traj[0], original_traj[1], 
            original_traj[2:4], original_traj[4]
        )
        
        # 1. Spatial similarity (Haversine distance)
        # Simplified as Euclidean distance for efficiency
        valid_points = tf.cast(gen_mask, tf.float32)
        spatial_diff = tf.reduce_sum(tf.square(gen_latlon - orig_latlon), axis=-1)
        spatial_diff = tf.reduce_sum(spatial_diff * valid_points) / (tf.reduce_sum(valid_points) + 1e-8)
        
        # 2. Semantic similarity (Jensen-Shannon divergence between category distributions)
        # Approximate by computing category distribution similarity
        gen_cat_dist = tf.reduce_sum(gen_category * valid_points, axis=1)
        orig_cat_dist = tf.reduce_sum(orig_category * valid_points, axis=1)
        
        # Normalize to create probability distributions
        gen_cat_dist = gen_cat_dist / (tf.reduce_sum(gen_cat_dist) + 1e-8)
        orig_cat_dist = orig_cat_dist / (tf.reduce_sum(orig_cat_dist) + 1e-8)
        
        # JS divergence (approximated)
        m_dist = 0.5 * (gen_cat_dist + orig_cat_dist)
        semantic_diff = 0.5 * (
            tf.reduce_sum(gen_cat_dist * tf.math.log(gen_cat_dist / (m_dist + 1e-8) + 1e-8)) +
            tf.reduce_sum(orig_cat_dist * tf.math.log(orig_cat_dist / (m_dist + 1e-8) + 1e-8))
        )
        
        # 3. Temporal similarity
        # Simplified temporal similarity
        temporal_diff = tf.reduce_sum(tf.square(gen_time - orig_time) * valid_points) / (tf.reduce_sum(valid_points) + 1e-8)
        
        # Combined utility difference (lower is better)
        w1, w2, w3 = 0.5, 0.25, 0.25  # Weights for each component
        utility_diff = w1 * spatial_diff + w2 * semantic_diff + w3 * temporal_diff
        
        # Convert to reward (higher is better)
        # Scale to reasonable range [0, 5]
        utility_reward = 5.0 * tf.exp(-utility_diff)
        
        return utility_reward
    
    def compute_adversarial_reward(self, discriminator_output):
        """
        Compute adversarial reward based on discriminator output.
        Higher reward when discriminator is fooled (output close to 1).
        
        Args:
            discriminator_output: Output from the discriminator (0-1)
            
        Returns:
            adversarial_reward: Scalar adversarial reward
        """
        # Reward is log of discriminator output (higher when D is fooled)
        adv_reward = tf.math.log(discriminator_output + 1e-8)
        
        # Scale to reasonable range [0, 5]
        adv_reward = 5.0 * (1.0 + adv_reward) / 2.0
        
        return adv_reward
    
    def update_adaptive_weights(self, acc_at_1, fid):
        """
        Update weights based on current privacy and utility metrics.
        
        Args:
            acc_at_1: Current re-identification accuracy
            fid: Current FID score
        """
        self.current_acc_at_1 = acc_at_1
        self.current_fid = fid
        
        # Update alpha (privacy weight) - increase if privacy is worse than target
        self.alpha = self.alpha * min(1.0, self.current_acc_at_1 / self.acc_at_1_target)
        
        # Update beta (utility weight) - increase if utility is worse than target
        utility_factor = max(0.0, 1.0 - (self.current_fid - self.fid_target) / self.fid_max)
        self.beta = self.beta * utility_factor
        
        # Ensure weights sum to 1
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
    
    def compute_reward(self, generated_traj, original_traj, user_id, discriminator_output):
        """
        Compute the complete reward by combining privacy, utility, and adversarial components.
        
        Args:
            generated_traj: The generated trajectory
            original_traj: The original trajectory
            user_id: True user ID for privacy evaluation
            discriminator_output: Output from the discriminator
            
        Returns:
            total_reward: Combined reward value
        """
        privacy_reward = self.compute_privacy_reward(generated_traj, user_id)
        utility_reward = self.compute_utility_reward(generated_traj, original_traj)
        adv_reward = self.compute_adversarial_reward(discriminator_output)
        
        # Combine rewards with current weights
        total_reward = (
            self.alpha * privacy_reward + 
            self.beta * utility_reward + 
            self.gamma * adv_reward
        )
        
        # Add a small exploration bonus to avoid policy collapse
        # by rewarding entropy in the generated trajectories
        # exploration_bonus = 0.1 * self.compute_entropy_bonus(generated_traj)
        # total_reward += exploration_bonus
        
        return total_reward
    
    def compute_entropy_bonus(self, generated_traj):
        """
        Compute entropy bonus to encourage exploration.
        
        Args:
            generated_traj: The generated trajectory
            
        Returns:
            entropy_bonus: Scalar entropy bonus
        """
        # For categorical data (like POI categories)
        cat_probs = generated_traj[1]
        cat_entropy = -tf.reduce_sum(cat_probs * tf.math.log(cat_probs + 1e-8), axis=-1)
        avg_cat_entropy = tf.reduce_mean(cat_entropy)
        
        # For continuous data (like lat/lon), use a simplification
        latlon = generated_traj[0]
        # Compute variance as a proxy for entropy
        mean_latlon = tf.reduce_mean(latlon, axis=1, keepdims=True)
        latlon_var = tf.reduce_mean(tf.square(latlon - mean_latlon))
        
        # Combined entropy bonus
        entropy_bonus = 0.5 * avg_cat_entropy + 0.5 * latlon_var
        
        # Normalize to [0, 1] range
        entropy_bonus = tf.minimum(1.0, entropy_bonus)
        
        return entropy_bonus 