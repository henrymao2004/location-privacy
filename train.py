import numpy as np
import os
import pandas as pd
import wandb
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from model import KAN_TrajGAN

def compute_data_stats():
    """Compute statistics from training data."""
    # Load training data
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
    # Compute centroids
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
    # Compute scale factor
    scale_factor = max(
        max(abs(tr['lat'].max() - lat_centroid),
            abs(te['lat'].max() - lat_centroid),
            abs(tr['lat'].min() - lat_centroid),
            abs(te['lat'].min() - lat_centroid)),
        max(abs(tr['lon'].max() - lon_centroid),
            abs(te['lon'].max() - lon_centroid),
            abs(tr['lon'].min() - lon_centroid),
            abs(te['lon'].min() - lon_centroid))
    )
    
    # Load training data for category size
    x_train = np.load('data/final_train.npy', allow_pickle=True)
    category_size = x_train[3].shape[-1]  # Get category vocabulary size
    
    # Get max sequence length
    max_length = 144  # Default value, can be adjusted based on your data
    
    # Create stats dictionary
    stats = {
        'lat_centroid': lat_centroid,
        'lon_centroid': lon_centroid,
        'scale_factor': scale_factor,
        'category_size': category_size,
        'max_length': max_length
    }
    
    # Save stats
    np.save('data/data_stats.npy', stats)
    return stats

def main():
    # Initialize wandb
    wandb.init(
        project="kan-trajgan",
        config={
            "architecture": "KAN-TrajGAN",
            "dataset": "mobility-trajectories",
            "epochs": 200,
            "batch_size": 256,
            "latent_dim": 100
        }
    )
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Compute or load data statistics
    if not os.path.exists('data/data_stats.npy'):
        print("Computing data statistics...")
        data_stats = compute_data_stats()
    else:
        print("Loading data statistics...")
        data_stats = np.load('data/data_stats.npy', allow_pickle=True).item()
    
    # Initialize model parameters
    latent_dim = 100
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {
        'lat_lon': 2,
        'day': 7,
        'hour': 24,
        'category': data_stats['category_size'],
        'mask': 1
    }
    max_length = data_stats['max_length']
    lat_centroid = data_stats['lat_centroid']
    lon_centroid = data_stats['lon_centroid']
    scale_factor = data_stats['scale_factor']
    
    # Initialize and train the model
    model = KAN_TrajGAN(
        latent_dim=latent_dim,
        keys=keys,
        vocab_size=vocab_size,
        max_length=max_length,
        lat_centroid=lat_centroid,
        lon_centroid=lon_centroid,
        scale_factor=scale_factor
    )
    
    # Training parameters
    epochs = 200
    batch_size = 256
    sample_interval = 10
    
    # Train the model with wandb logging
    train_with_monitoring(model, epochs, batch_size, sample_interval)
    
    # Close wandb run
    wandb.finish()

def train_with_monitoring(model, epochs, batch_size, sample_interval):
    """Train the model with wandb monitoring - optimized for spatial and category utility"""
    # Training data
    x_train = np.load('data/final_train.npy', allow_pickle=True)
    
    # Padding with consistent dtype
    X_train = [pad_sequences(f, model.max_length, padding='pre', dtype='float32') 
               for f in x_train[:5]]  # Only use the first 5 elements (features), not lengths
    
    # Variables to track best model performance
    # Track both overall and specific metrics
    best_utility_score = float('-inf')
    best_spatial_score = float('-inf')
    best_spatial_diversity = float('-inf')  # Track best spatial diversity separately
    best_category_score = float('-inf')
    best_diversity_score = float('-inf')
    best_epoch = 0
    
    # Use a constant learning rate throughout training
    learning_rate = 0.0003  # Set to the initial learning rate
    
    # Apply the constant learning rate
    model.generator_optimizer.learning_rate = learning_rate
    model.discriminator_optimizer.learning_rate = learning_rate * 0.5
    
    print(f"Starting training for {epochs} epochs with constant learning rate {learning_rate} and enhanced spatial focus...")
    for epoch in range(epochs):
        # Multiple training iterations per epoch for more stable learning
        metrics_list = []
        for _ in range(3):  # Run 3 iterations per epoch
            # Sample batch
            indices = np.random.permutation(X_train[0].shape[0])[:batch_size]
            batch = [X[indices] for X in X_train]
            
            # Ensure batch tensors have consistent dtype
            batch = [tf.cast(tensor, tf.float32) for tensor in batch]
            
            # Training step
            metrics = model.train_step(batch, batch_size)
            metrics_list.append(metrics)
        
        # Average the metrics
        avg_metrics = {
            'd_loss_real': np.mean([m['d_loss_real'] for m in metrics_list]),
            'd_loss_fake': np.mean([m['d_loss_fake'] for m in metrics_list]),
            'g_loss': np.mean([m['g_loss'] for m in metrics_list]),
        }
        
        # Extract utility metrics from the model with the latest batch
        utility_metric, spatial_metric, category_metric, spatial_diversity, category_diversity = extract_utility_metrics(model, batch)
        avg_metrics['utility_score'] = utility_metric
        
        # Combined diversity score
        diversity_score = spatial_diversity + 0.5 * category_diversity
        
        # Create a dictionary for wandb
        to_log = {
            'epoch': epoch,
            'd_loss_real': avg_metrics['d_loss_real'],
            'd_loss_fake': avg_metrics['d_loss_fake'],
            'g_loss': avg_metrics['g_loss'],
            'utility_score': avg_metrics['utility_score'],
            'learning_rate': float(model.generator_optimizer.learning_rate),
            'spatial_metric': spatial_metric,
            'category_metric': category_metric,
            'spatial_diversity': spatial_diversity,
            'category_diversity': category_diversity,
            'diversity_score': diversity_score
        }
        
        # Track best scores and update checkpoints
        try:
            utility_score = to_log['utility_score']
            spatial_score = to_log['spatial_metric'] 
            category_score = to_log['category_metric']
            
            # Save overall best model
            if utility_score > best_utility_score:
                best_utility_score = utility_score
                best_epoch = epoch
                to_log['best_utility_score'] = best_utility_score
                to_log['best_epoch'] = best_epoch
                # Save overall best model
                model.save_checkpoint('best')
                print(f"New best model at epoch {epoch} with utility score: {utility_score:.4f}")
            
            # Save best spatial model
            if spatial_score > best_spatial_score:
                best_spatial_score = spatial_score
                to_log['best_spatial_score'] = best_spatial_score
                # Save best spatial model
                model.save_checkpoint('best_spatial')
                print(f"New best spatial model at epoch {epoch} with spatial score: {spatial_score:.4f}")
            
            # Save best spatial diversity model
            if spatial_diversity > best_spatial_diversity:
                best_spatial_diversity = spatial_diversity
                to_log['best_spatial_diversity'] = best_spatial_diversity
                # Save best spatial diversity model
                model.save_checkpoint('best_spatial_diversity')
                print(f"New best spatial diversity at epoch {epoch}: {spatial_diversity:.4f}")
                
            # Save best category model
            if category_score > best_category_score:
                best_category_score = category_score
                to_log['best_category_score'] = best_category_score
                # Save best category model
                model.save_checkpoint('best_category')
                print(f"New best category model at epoch {epoch} with category score: {category_score:.4f}")
                
            # Save best diversity model
            if diversity_score > best_diversity_score:
                best_diversity_score = diversity_score
                to_log['best_diversity_score'] = best_diversity_score
                # Save best diversity model
                model.save_checkpoint('best_diversity')
                print(f"New best diversity model at epoch {epoch} with diversity score: {diversity_score:.4f}")
                
        except Exception as e:
            print(f"Error tracking model checkpoints: {e}")
            
        wandb.log(to_log)
        
        # Print progress
        if epoch % 5 == 0:  # Increase frequency of logging for better monitoring
            print(f"Epoch {epoch}/{epochs}")
            print(f"D_real: {avg_metrics['d_loss_real']:.4f}, D_fake: {avg_metrics['d_loss_fake']:.4f}, G: {avg_metrics['g_loss']:.4f}")
            print(f"Utility: {avg_metrics['utility_score']:.4f}, Spatial: {spatial_score:.4f}, Diversity: {diversity_score:.4f}")
        
        # Save checkpoints
        if epoch % sample_interval == 0:
            model.save_checkpoint(epoch)
            
            # Log model weights to wandb
            wandb.save(f"results/generator_{epoch}.weights.h5")
            wandb.save(f"results/discriminator_{epoch}.weights.h5")
    
    print(f"\nTraining completed. Best model found at epoch {best_epoch} with utility score: {best_utility_score:.4f}")
    print(f"The best model has been saved with checkpoint name 'best'")
    print(f"Best spatial model saved as 'best_spatial', Best category model saved as 'best_category'")
    print(f"Best diversity model saved as 'best_diversity', Best spatial diversity model saved as 'best_spatial_diversity'")

def extract_utility_metrics(model, batch):
    """Extract utility metrics from a training batch with emphasis on spatial and category diversity"""
    # Sample from normal distribution for KAN model
    z = tf.random.normal([len(batch[0]), model.latent_dim])
    
    # Generate trajectories
    gen_trajs = model.generator.predict(z, verbose=0)
    
    try:
        # Ensure both batch and generated trajectories are float32
        batch_float32 = [tf.cast(tensor, tf.float32) for tensor in batch]
        gen_trajs_float32 = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
        
        # Get utility metrics from model's computation function
        metrics = model.compute_utility_metrics(batch_float32, gen_trajs_float32)
        
        # Extract and format for display
        spatial_metric = -metrics['spatial_loss'] + metrics['spatial_diversity']
        spatial_diversity = metrics['spatial_diversity']
        day_metric = -metrics['day_loss']
        hour_metric = -metrics['hour_loss']
        category_metric = -metrics['category_loss']
        category_diversity = metrics['category_diversity'] 
        utility_metric = metrics['total_utility']
        
        # Print individual metrics for debugging
        print(f"  Metrics - Spatial: {spatial_metric:.4f} (Diversity: {spatial_diversity:.4f}), " 
              f"Day: {day_metric:.4f}, Hour: {hour_metric:.4f}, "
              f"Category: {category_metric:.4f} (Diversity: {category_diversity:.4f}), "
              f"Overall: {utility_metric:.4f}")
        
        return utility_metric, spatial_metric, category_metric, spatial_diversity, category_diversity
        
    except Exception as e:
        print(f"Error in extract_utility_metrics: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to improved metrics calculation if the detailed approach fails
        try:
            # Earth radius in km for haversine distance
            EARTH_RADIUS_KM = 6371.0
            
            # Ensure consistent tensor types for fallback calculation
            batch_float32 = [tf.cast(tensor, tf.float32) for tensor in batch]
            gen_trajs_float32 = [tf.cast(tensor, tf.float32) for tensor in gen_trajs]
            
            # Spatial loss using Euclidean MSE
            spatial_loss = tf.reduce_mean(tf.square(gen_trajs_float32[0] - batch_float32[0]))
            
            # Add improved haversine calculation for geographic context
            # Convert to radians
            real_coords = batch_float32[0] * tf.constant(np.pi / 180, dtype=tf.float32)
            gen_coords = gen_trajs_float32[0] * tf.constant(np.pi / 180, dtype=tf.float32)
            
            # Extract lat/lon
            real_lat = real_coords[:,:,0]
            real_lon = real_coords[:,:,1]
            gen_lat = gen_coords[:,:,0]
            gen_lon = gen_coords[:,:,1]
            
            # Haversine formula components
            lat_diff = gen_lat - real_lat
            lon_diff = gen_lon - real_lon
            
            a = tf.sin(lat_diff/2)**2 + tf.cos(real_lat) * tf.cos(gen_lat) * tf.sin(lon_diff/2)**2
            c = 2 * tf.atan2(tf.sqrt(a), tf.sqrt(1-a))
            geo_distance = EARTH_RADIUS_KM * c
            
            # Take the mean across all points
            haversine_distance = tf.reduce_mean(geo_distance)
            
            # Combined spatial loss with both metrics
            combined_spatial_loss = spatial_loss + 0.25 * haversine_distance
            spatial_loss_np = float(combined_spatial_loss.numpy())
            
            # Enhanced diversity calculation for spatial data
            batch_size = tf.shape(gen_trajs_float32[0])[0]
            
            # Reshape for pairwise distance calculation
            points1 = tf.reshape(gen_coords, [batch_size, -1, 2])
            points2 = tf.reshape(gen_coords, [batch_size, 1, -1, 2])
            
            # Calculate pairwise distances
            pairwise_diffs = points1[:, tf.newaxis, :, :] - points2
            pairwise_distances = tf.reduce_mean(tf.sqrt(tf.reduce_sum(pairwise_diffs**2, axis=-1)), axis=-1)
            
            # Mask self-comparisons
            mask = 1.0 - tf.eye(batch_size)
            masked_distances = pairwise_distances * tf.cast(mask, tf.float32)
            
            # Calculate diversity score
            spatial_diversity_np = float(tf.reduce_sum(masked_distances).numpy()) / (batch_size * (batch_size - 1.0))
            
            # Temporal loss - lower is better
            temp_day_loss = -tf.reduce_mean(tf.reduce_sum(
                batch_float32[2] * tf.math.log(tf.clip_by_value(gen_trajs_float32[2], 1e-7, 1.0)), 
                axis=-1))
            temp_day_loss_np = float(temp_day_loss.numpy())
            
            temp_hour_loss = -tf.reduce_mean(tf.reduce_sum(
                batch_float32[3] * tf.math.log(tf.clip_by_value(gen_trajs_float32[3], 1e-7, 1.0)), 
                axis=-1))
            temp_hour_loss_np = float(temp_hour_loss.numpy())
            
            # Category loss - lower is better
            cat_loss = -tf.reduce_mean(tf.reduce_sum(
                batch_float32[1] * tf.math.log(tf.clip_by_value(gen_trajs_float32[1], 1e-7, 1.0)), 
                axis=-1))
            cat_loss_np = float(cat_loss.numpy())
            
            # Simple category diversity calculation
            cat_probs = tf.reshape(gen_trajs_float32[1], [batch_size, -1])
            
            # Normalize for cosine distance
            cat_normalized = tf.nn.l2_normalize(cat_probs, axis=1)
            similarity = tf.matmul(cat_normalized, cat_normalized, transpose_b=True)
            
            # Convert to distance
            distance = 1.0 - similarity
            mask = 1.0 - tf.eye(batch_size)
            masked_distance = distance * mask
            
            # Calculate category diversity
            category_diversity_np = float(tf.reduce_sum(masked_distance).numpy()) / (batch_size * (batch_size - 1.0))
            
            # Calculate individual components for debugging
            spatial_metric = spatial_loss_np + spatial_diversity_np
            day_metric = -temp_day_loss_np
            hour_metric = -temp_hour_loss_np
            category_metric = -cat_loss_np + 0.5 * category_diversity_np
            
            # Combine utility components with adjusted weights to prioritize spatial accuracy
            utility_metric = -(5.0 * spatial_loss_np + 2.5 * cat_loss_np + 1.0 * temp_day_loss_np + 0.5 * temp_hour_loss_np) + (2.0 * spatial_diversity_np + 1.5 * category_diversity_np)
            
            # Print individual metrics for debugging
            print(f"  Fallback Metrics - Spatial: {spatial_metric:.4f} (Diversity: {spatial_diversity_np:.4f}), " 
                  f"Day: {day_metric:.4f}, Hour: {hour_metric:.4f}, "
                  f"Category: {category_metric:.4f} (Diversity: {category_diversity_np:.4f}), "
                  f"Overall: {utility_metric:.4f}")
            
            return utility_metric, spatial_metric, category_metric, spatial_diversity_np, category_diversity_np
            
        except Exception as e2:
            print(f"Fallback utility metrics calculation also failed: {e2}")
            return -1.0, -1.0, -1.0, 0.0, 0.0

if __name__ == '__main__':
    main()