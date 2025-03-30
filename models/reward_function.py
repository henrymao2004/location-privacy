import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.spatial.distance import jensenshannon

class TrajRewardFunction:
    def __init__(self, tul_classifier=None, alpha=0.3, beta=0.5, gamma=0.2):
        """
        Initialize the trajectory reward function.
        
        Args:
            tul_classifier: TUL model used for privacy evaluation
            alpha: Weight for privacy reward component (decreased to 0.3 from 0.4)
            beta: Weight for utility reward component (increased to 0.5 from 0.4)
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
    
    def _maybe_convert_to_numpy(self, tensor_or_array):
        """Convert tensor to numpy array if needed"""
        if isinstance(tensor_or_array, tf.Tensor):
            return tensor_or_array.numpy()
        return tensor_or_array
    
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
            return 0.5
        
        # Convert trajectory components to numpy arrays
        generated_traj_np = [self._maybe_convert_to_numpy(comp) for comp in generated_traj]
        
        try:
            # Check if we're using a MARC model (checks for specific input layer names)
            is_marc_model = False
            if hasattr(self.tul_classifier.model, 'input_names'):
                input_names = self.tul_classifier.model.input_names
                is_marc_model = ('input_day' in input_names and 
                                 'input_hour' in input_names and 
                                 'input_category' in input_names)
            
            if is_marc_model:
                print("Using MARC-compatible conversion")
                # MARC model expects integer indices, not one-hot encoded inputs
                
                # Extract day, hour, category features
                day_features = generated_traj_np[1]  # Shape: [batch, seq_len, day_dim]
                hour_features = generated_traj_np[2]  # Shape: [batch, seq_len, hour_dim]
                category_features = generated_traj_np[3]  # Shape: [batch, seq_len, category_dim]
                
                # Convert one-hot to indices
                day_indices = np.argmax(day_features, axis=2)  # Shape: [batch, seq_len]
                hour_indices = np.argmax(hour_features, axis=2)  # Shape: [batch, seq_len]
                category_indices = np.argmax(category_features, axis=2)  # Shape: [batch, seq_len]
                
                # Expand lat_lon to match MARC input shape (from 2 to 40 features)
                # We'll use zero-padding for simplicity; in a real application, 
                # you'd need proper lat_lon feature engineering
                lat_lon = generated_traj_np[0]  # Shape: [batch, seq_len, 2]
                batch_size, seq_len, _ = lat_lon.shape
                lat_lon_expanded = np.zeros((batch_size, seq_len, 40))
                lat_lon_expanded[:, :, :2] = lat_lon  # Copy original lat_lon to first 2 dimensions
                
                # Prepare inputs for MARC model
                marc_inputs = [
                    day_indices,         # input_day: [batch, seq_len]
                    hour_indices,        # input_hour: [batch, seq_len]
                    category_indices,    # input_category: [batch, seq_len]
                    lat_lon_expanded     # input_lat_lon: [batch, seq_len, 40]
                ]
                
                print(f"Converting inputs with shapes: {[comp.shape for comp in generated_traj_np[:4]]}")
                print(f"Converted shapes: day={day_indices.shape}, hour={hour_indices.shape}, category={category_indices.shape}, latlon={lat_lon_expanded.shape}")
                
                # Get TUL model prediction
                tul_probs = self.tul_classifier.predict(marc_inputs)
            else:
                print("Using standard TUL conversion")
                # Format inputs for standard TUL classifier (4 inputs):
                # [lat_lon, category, time (day+hour), mask]
                
                # Extract day and hour features
                day_features = generated_traj_np[1]  # Shape: [batch, seq_len, day_dim]
                hour_features = generated_traj_np[2]  # Shape: [batch, seq_len, hour_dim]
                
                # Create time features with shape [batch, seq_len, 2]
                batch_size = day_features.shape[0]
                seq_len = day_features.shape[1]
                time_features = np.zeros((batch_size, seq_len, 2))
                
                # Extract the maximum value index for day and hour (one-hot to index)
                for b in range(batch_size):
                    for t in range(seq_len):
                        # Get the index of max value (convert one-hot to index)
                        if np.sum(day_features[b, t]) > 0:
                            day_idx = np.argmax(day_features[b, t])
                        else:
                            day_idx = 0
                            
                        if np.sum(hour_features[b, t]) > 0:
                            hour_idx = np.argmax(hour_features[b, t])
                        else:
                            hour_idx = 0
                            
                        # Normalize to [0,1] range
                        time_features[b, t, 0] = day_idx / 7.0  # 7 days in a week
                        time_features[b, t, 1] = hour_idx / 24.0  # 24 hours in a day
                
                # Prepare inputs for TUL classifier
                tul_inputs = [
                    generated_traj_np[0],  # lat_lon
                    generated_traj_np[3],  # category
                    time_features,         # time features (day, hour)
                    generated_traj_np[4]   # mask
                ]
                
                # Get TUL model prediction
                tul_probs = self.tul_classifier.predict(tul_inputs)
            
            # Get the number of users in the TUL classifier output
            num_users = tul_probs.shape[1]
            
            # Extract probability for the true user
            if isinstance(user_id, (list, np.ndarray)):
                user_id = user_id[0]  # Take first element if it's batched
            
            # Use modulo to ensure user_id is within valid range
            valid_user_id = user_id % num_users
            if valid_user_id != user_id:
                print(f"Adjusted user ID from {user_id} to {valid_user_id} (num_users: {num_users})")
            
            true_user_prob = tul_probs[0, valid_user_id]
            
            # Privacy reward is negative log probability of correct identification
            # Higher reward means lower identification probability (better privacy)
            privacy_reward = -np.log(true_user_prob + 1e-8)
            
            # Normalize reward to reasonable range [0, 5]
            privacy_reward = min(5.0, privacy_reward)
            
            return privacy_reward
        except Exception as e:
            print(f"Privacy reward computation error: {e}")
            return 0.5  # Default value in case of error
    
    def compute_utility_reward(self, generated_traj, original_traj):
        """
        Compute utility reward based on spatial, temporal, and semantic similarity.
        Higher scores for better utility preservation.
        
        Args:
            generated_traj: The generated trajectory
            original_traj: The original trajectory
            
        Returns:
            utility_reward: Scalar utility reward
        """
        # Convert to numpy arrays
        generated_traj_np = [self._maybe_convert_to_numpy(comp) for comp in generated_traj]
        original_traj_np = [self._maybe_convert_to_numpy(comp) for comp in original_traj]
        
        # Extract components
        gen_latlon, gen_day, gen_hour, gen_category, gen_mask = (
            generated_traj_np[0], generated_traj_np[1], 
            generated_traj_np[2], generated_traj_np[3], generated_traj_np[4]
        )
        orig_latlon, orig_day, orig_hour, orig_category, orig_mask = (
            original_traj_np[0], original_traj_np[1], 
            original_traj_np[2], original_traj_np[3], original_traj_np[4]
        )
        
        # Extract valid points mask
        valid_points = gen_mask[:,:,0]
        
        # 1. Spatial similarity (Haversine distance for geo coordinates)
        # Scale distances appropriately to match evaluation metrics
        spatial_diff = np.sum(np.square(gen_latlon - orig_latlon), axis=-1)
        spatial_diff_mean = np.sum(spatial_diff * valid_points) / (np.sum(valid_points) + 1e-8)
        spatial_similarity = np.exp(-spatial_diff_mean * 5)  # Exponential scaling to match utility score
        
        # 2. Day similarity (Jensen-Shannon divergence between day distributions)
        # Extract one-hot encoded day distributions
        gen_day_dist = np.sum(gen_day * valid_points[:,:,np.newaxis], axis=1) 
        orig_day_dist = np.sum(orig_day * valid_points[:,:,np.newaxis], axis=1)
        
        # Normalize to create probability distributions
        gen_day_dist = gen_day_dist / (np.sum(gen_day_dist, axis=-1, keepdims=True) + 1e-8)
        orig_day_dist = orig_day_dist / (np.sum(orig_day_dist, axis=-1, keepdims=True) + 1e-8)
        
        # Calculate JS divergence for days
        try:
            day_js_div = jensenshannon(gen_day_dist[0], orig_day_dist[0])
            if np.isnan(day_js_div):
                day_js_div = 1.0  # Fallback if distributions are problematic
        except:
            day_js_div = 1.0  # Fallback
        
        day_similarity = 1.0 - day_js_div
        
        # 3. Hour similarity (Jensen-Shannon divergence between hour distributions)
        # Extract one-hot encoded hour distributions
        gen_hour_dist = np.sum(gen_hour * valid_points[:,:,np.newaxis], axis=1)
        orig_hour_dist = np.sum(orig_hour * valid_points[:,:,np.newaxis], axis=1)
        
        # Normalize to create probability distributions
        gen_hour_dist = gen_hour_dist / (np.sum(gen_hour_dist, axis=-1, keepdims=True) + 1e-8)
        orig_hour_dist = orig_hour_dist / (np.sum(orig_hour_dist, axis=-1, keepdims=True) + 1e-8)
        
        # Calculate JS divergence for hours
        try:
            hour_js_div = jensenshannon(gen_hour_dist[0], orig_hour_dist[0])
            if np.isnan(hour_js_div):
                hour_js_div = 1.0  # Fallback if distributions are problematic
        except:
            hour_js_div = 1.0  # Fallback
        
        hour_similarity = 1.0 - hour_js_div
        
        # 4. Category similarity (Jensen-Shannon divergence between category distributions)
        # Extract one-hot encoded category distributions
        gen_cat_dist = np.sum(gen_category * valid_points[:,:,np.newaxis], axis=1)
        orig_cat_dist = np.sum(orig_category * valid_points[:,:,np.newaxis], axis=1)
        
        # Normalize to create probability distributions
        gen_cat_dist = gen_cat_dist / (np.sum(gen_cat_dist, axis=-1, keepdims=True) + 1e-8)
        orig_cat_dist = orig_cat_dist / (np.sum(orig_cat_dist, axis=-1, keepdims=True) + 1e-8)
        
        # Calculate JS divergence for categories
        try:
            cat_js_div = jensenshannon(gen_cat_dist[0], orig_cat_dist[0])
            if np.isnan(cat_js_div):
                cat_js_div = 1.0  # Fallback if distributions are problematic
        except:
            cat_js_div = 1.0  # Fallback
        
        category_similarity = 1.0 - cat_js_div
        
        # 5. Temporal similarity (combined day and hour)
        temporal_similarity = (day_similarity + hour_similarity) / 2.0
        
        # Weight each component to match the evaluation metrics
        w_spatial = 0.35    # Weight for spatial similarity
        w_temporal = 0.35   # Weight for temporal similarity
        w_day = 0.10        # Additional weight for day similarity
        w_hour = 0.10       # Additional weight for hour similarity
        w_category = 0.10   # Weight for category similarity
        
        # Combine similarity scores with weights
        utility_similarity = (
            w_spatial * spatial_similarity +
            w_temporal * temporal_similarity + 
            w_day * day_similarity +
            w_hour * hour_similarity +
            w_category * category_similarity
        )
        
        # Scale to reasonable range [0, 5]
        utility_reward = 5.0 * utility_similarity
        
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
        # Convert to numpy if needed
        if isinstance(discriminator_output, tf.Tensor):
            discriminator_output = discriminator_output.numpy()
        
        # Reward is log of discriminator output (higher when D is fooled)
        adv_reward = np.log(discriminator_output + 1e-8)
        
        # Scale to reasonable range [0, 5]
        adv_reward = 5.0 * (1.0 + adv_reward) / 2.0
        
        return float(adv_reward)
    
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
        try:
            privacy_reward = self.compute_privacy_reward(generated_traj, user_id)
            utility_reward = self.compute_utility_reward(generated_traj, original_traj)
            adv_reward = self.compute_adversarial_reward(discriminator_output)
            
            # Add entropy bonus for exploration
            entropy_bonus = self.compute_entropy_bonus(generated_traj)
            
            # Combine rewards with current weights
            total_reward = (
                self.alpha * privacy_reward + 
                self.beta * utility_reward + 
                self.gamma * adv_reward +
                0.05 * entropy_bonus  # Small weight for exploration
            )
            
            # Apply non-linear scaling to emphasize higher rewards
            total_reward = np.power(total_reward, 1.2)
            
            return float(total_reward)
        except Exception as e:
            print(f"Error computing reward: {e}")
            # Return a default reward to avoid breaking the training loop
            return 0.1
    
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
        cat_entropy = -np.sum(cat_probs * np.log(cat_probs + 1e-8), axis=-1)
        avg_cat_entropy = np.mean(cat_entropy)
        
        # For continuous data (like lat/lon), use a simplification
        latlon = generated_traj[0]
        # Compute variance as a proxy for entropy
        mean_latlon = np.mean(latlon, axis=1, keepdims=True)
        latlon_var = np.mean(np.square(latlon - mean_latlon))
        
        # Combined entropy bonus
        entropy_bonus = 0.5 * avg_cat_entropy + 0.5 * latlon_var
        
        # Normalize to [0, 1] range
        entropy_bonus = np.minimum(1.0, entropy_bonus)
        
        return entropy_bonus 