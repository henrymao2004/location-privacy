import numpy as np
import torch
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
    model.eval()
    
    # Generate synthetic trajectories
    with torch.no_grad():
        # Convert test data to tensors and move to device
        real_tensors = {k: torch.stack([torch.from_numpy(t[k]).float() for t in test_data]).to(model.device) 
                       for k in test_data[0].keys()}
        
        # Generate synthetic data
        noise = torch.randn(len(test_data), model.latent_dim, device=model.device)
        syn_tensors = model.generator(noise)
        
        # Get features from discriminator
        real_features = model.discriminator.get_features(real_tensors)
        syn_features = model.discriminator.get_features(syn_tensors)
        
        # Convert to numpy for metric computation
        real_features = real_features.cpu().numpy()
        syn_features = syn_features.cpu().numpy()
        
        # Convert synthetic tensors to numpy for JSD computation
        syn_data = {k: v.cpu().numpy() for k, v in syn_tensors.items()}
    
    metrics = {}
    
    # FID Score
    metrics['fid'] = compute_fid(real_features, syn_features)
    
    # JSD for each feature
    for key in model.keys:
        if key != 'mask':
            real_dist = np.mean([t[key] for t in test_data], axis=0)
            syn_dist = np.mean(syn_data[key], axis=0)
            metrics[f'jsd_{key}'] = compute_jsd(real_dist, syn_dist)
    
    return metrics 