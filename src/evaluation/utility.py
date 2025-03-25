import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import entropy

def compute_fid(real_features, gen_features):
    """Compute Fréchet Inception Distance"""
    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=1)
    mu_gen = np.mean(gen_features, axis=1)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Compute FID
    diff = mu_real - mu_gen
    covmean = np.sqrt(np.matmul(sigma_real, sigma_gen))
    fid = np.sum(diff * diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return fid

def compute_jsd(p, q):
    """Compute Jensen-Shannon Divergence"""
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def evaluate_utility(model, test_data):
    """Evaluate utility preservation using FID and JSD metrics"""
    # Generate synthetic trajectories
    noise = tf.random.normal([len(test_data), model.latent_dim])
    gen_trajs = model.generator.predict([*test_data[:4], test_data[4], noise])
    
    metrics = {}
    
    # FID Score
    real_features = model.discriminator.get_layer('global_average_pooling1d').output
    gen_features = model.discriminator.get_layer('global_average_pooling1d').output
    metrics['fid'] = compute_fid(real_features, gen_features)
    
    # JSD for each feature
    for key in model.keys:
        if key != 'mask':
            real_dist = np.mean(test_data[key], axis=0)
            gen_dist = np.mean(gen_trajs[key], axis=0)
            metrics[f'jsd_{key}'] = compute_jsd(real_dist, gen_dist)
    
    return metrics 