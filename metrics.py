import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import entropy
import warnings

# Suppress specific TensorFlow and NumPy warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def compute_acc_acc5_f1_prec_rec(y_true, y_pred, print_metrics=True, print_pfx=''):
    """
    Compute classification metrics for privacy evaluation.
    
    Args:
        y_true: One-hot encoded ground truth labels
        y_pred: Predicted probabilities from model
        print_metrics: Whether to print metrics
        print_pfx: Prefix to add to printed output
        
    Returns:
        tuple: (accuracy, top5_accuracy, f1_macro, precision_macro, recall_macro)
    """
    # Process predictions to get argmax
    y_pred_argmax = np.argmax(y_pred, axis=1)
    y_true_argmax = np.argmax(y_true, axis=1)
    
    # Compute accuracy
    acc = accuracy_score(y_true_argmax, y_pred_argmax)
    
    # Compute top-5 accuracy
    acc_top5 = _compute_top_k_accuracy(y_true, y_pred, k=5)
    
    # Convert to binary format for f1, precision, recall
    y_pred_one_hot = np.zeros_like(y_pred)
    for i, idx in enumerate(y_pred_argmax):
        y_pred_one_hot[i, idx] = 1
    
    # Compute macro metrics
    f1_macro_val = f1_score(y_true_argmax, y_pred_argmax, average='macro')
    prec_macro_val = precision_score(y_true_argmax, y_pred_argmax, average='macro')
    rec_macro_val = recall_score(y_true_argmax, y_pred_argmax, average='macro')
    
    if print_metrics:
        pfx = '' if print_pfx == '' else print_pfx + '\t'
        print(f"{pfx}acc: {acc:.6f}\tacc_top5: {acc_top5:.6f}\tf1_macro: {f1_macro_val:.6f}\t"
              f"prec_macro: {prec_macro_val:.6f}\trec_macro: {rec_macro_val:.6f}")
    
    return acc, acc_top5, f1_macro_val, prec_macro_val, rec_macro_val


def _compute_top_k_accuracy(y_true, y_pred, k=5):
    """
    Compute top-k accuracy.
    
    Args:
        y_true: One-hot encoded ground truth labels
        y_pred: Predicted probabilities from model
        k: Number of top predictions to consider
        
    Returns:
        float: Top-k accuracy
    """
    y_true_argmax = np.argmax(y_true, axis=1)
    top_k_indices = np.argsort(y_pred, axis=1)[:, -k:]
    
    correct = 0
    for i, true_idx in enumerate(y_true_argmax):
        if true_idx in top_k_indices[i]:
            correct += 1
    
    return correct / len(y_true)


# === SPATIAL UTILITY METRICS ===

def spatial_utility_mse(real_traj, gen_traj, mask=None):
    """
    Calculate Mean Squared Error between real and generated spatial coordinates.
    
    Args:
        real_traj: Real trajectory coordinates, shape [batch, seq_len, 2]
        gen_traj: Generated trajectory coordinates, shape [batch, seq_len, 2]
        mask: Optional mask for valid points, shape [batch, seq_len, 1]
        
    Returns:
        float: MSE between trajectories
    """
    if mask is None:
        # If no mask is provided, consider all points valid
        mask = np.ones((real_traj.shape[0], real_traj.shape[1], 1))
        
    # Repeat mask to match spatial dimensions
    mask_repeated = np.repeat(mask, 2, axis=2)
    
    # Calculate MSE
    diff = gen_traj - real_traj
    squared_diff = diff * diff
    masked_squared_diff = squared_diff * mask_repeated
    
    # Calculate trajectory lengths
    traj_lengths = np.sum(mask, axis=(1, 2)) + 1e-8
    
    # Calculate MSE per trajectory
    spatial_mse = np.sum(np.sum(masked_squared_diff, axis=1), axis=1) / traj_lengths
    
    return np.mean(spatial_mse)


def hausdorff_distance(real_traj, gen_traj, mask=None):
    """
    Calculate Hausdorff distance between real and generated trajectories.
    
    Args:
        real_traj: Real trajectory coordinates, shape [batch, seq_len, 2]
        gen_traj: Generated trajectory coordinates, shape [batch, seq_len, 2]
        mask: Optional mask for valid points, shape [batch, seq_len, 1]
        
    Returns:
        float: Average Hausdorff distance across batch
    """
    batch_size = real_traj.shape[0]
    distances = []
    
    for i in range(batch_size):
        # Extract valid points based on mask if provided
        if mask is not None:
            valid_indices = np.where(mask[i, :, 0] > 0)[0]
            real_points = real_traj[i, valid_indices]
            gen_points = gen_traj[i, valid_indices]
        else:
            real_points = real_traj[i]
            gen_points = gen_traj[i]
        
        # Skip if there are too few points
        if len(real_points) < 2 or len(gen_points) < 2:
            continue
            
        # Calculate directed Hausdorff distances
        d1 = directed_hausdorff(real_points, gen_points)
        d2 = directed_hausdorff(gen_points, real_points)
        
        # Hausdorff distance is the maximum of the two directed distances
        distances.append(max(d1, d2))
    
    return np.mean(distances) if distances else float('inf')


def directed_hausdorff(A, B):
    """
    Calculate directed Hausdorff distance from A to B.
    
    Args:
        A: First set of points, shape [n, 2]
        B: Second set of points, shape [m, 2]
        
    Returns:
        float: Directed Hausdorff distance
    """
    max_min_dist = 0
    for a in A:
        min_dist = float('inf')
        for b in B:
            dist = euclidean(a, b)
            min_dist = min(min_dist, dist)
        max_min_dist = max(max_min_dist, min_dist)
    
    return max_min_dist


def dtw_distance(real_traj, gen_traj, mask=None):
    """
    Calculate Dynamic Time Warping distance between real and generated trajectories.
    
    Args:
        real_traj: Real trajectory coordinates, shape [batch, seq_len, 2]
        gen_traj: Generated trajectory coordinates, shape [batch, seq_len, 2]
        mask: Optional mask for valid points, shape [batch, seq_len, 1]
        
    Returns:
        float: Average DTW distance across batch
    """
    batch_size = real_traj.shape[0]
    distances = []
    
    for i in range(batch_size):
        # Extract valid points based on mask if provided
        if mask is not None:
            valid_indices = np.where(mask[i, :, 0] > 0)[0]
            real_points = real_traj[i, valid_indices]
            gen_points = gen_traj[i, valid_indices]
        else:
            real_points = real_traj[i]
            gen_points = gen_traj[i]
        
        # Skip if there are too few points
        if len(real_points) < 2 or len(gen_points) < 2:
            continue
            
        # Calculate DTW distance
        distances.append(_compute_dtw(real_points, gen_points))
    
    return np.mean(distances) if distances else float('inf')


def _compute_dtw(A, B):
    """
    Compute DTW distance between two sequences.
    
    Args:
        A: First sequence, shape [n, 2]
        B: Second sequence, shape [m, 2]
        
    Returns:
        float: DTW distance
    """
    n, m = len(A), len(B)
    
    # Initialize distance matrix
    dtw_matrix = np.zeros((n+1, m+1))
    dtw_matrix[0, :] = np.inf
    dtw_matrix[:, 0] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Fill the matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = euclidean(A[i-1], B[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],     # insertion
                dtw_matrix[i, j-1],     # deletion
                dtw_matrix[i-1, j-1]    # match
            )
    
    return dtw_matrix[n, m]


# === TEMPORAL UTILITY METRICS ===

def temporal_utility_cross_entropy(real_day, gen_day, real_hour, gen_hour, mask=None):
    """
    Calculate cross entropy between real and generated temporal patterns.
    
    Args:
        real_day: Real day one-hot encodings, shape [batch, seq_len, 7]
        gen_day: Generated day one-hot encodings, shape [batch, seq_len, 7]
        real_hour: Real hour one-hot encodings, shape [batch, seq_len, 24]
        gen_hour: Generated hour one-hot encodings, shape [batch, seq_len, 24]
        mask: Optional mask for valid points, shape [batch, seq_len, 1]
        
    Returns:
        dict: Dictionary with day and hour cross entropy and combined score
    """
    if mask is None:
        # If no mask is provided, consider all points valid
        mask = np.ones((real_day.shape[0], real_day.shape[1], 1))
    
    # Clip generated values to avoid numerical issues
    gen_day_clipped = np.clip(gen_day, 1e-7, 1.0)
    gen_hour_clipped = np.clip(gen_hour, 1e-7, 1.0)
    
    # Calculate cross-entropy
    day_ce = _categorical_crossentropy(real_day, gen_day_clipped)
    hour_ce = _categorical_crossentropy(real_hour, gen_hour_clipped)
    
    # Apply mask
    mask_flat = mask.squeeze(axis=2)
    day_ce_masked = day_ce * mask_flat
    hour_ce_masked = hour_ce * mask_flat
    
    # Calculate trajectory lengths
    traj_lengths = np.sum(mask, axis=(1, 2)) + 1e-8
    
    # Calculate average CE per trajectory
    day_ce_avg = np.sum(day_ce_masked, axis=1) / traj_lengths
    hour_ce_avg = np.sum(hour_ce_masked, axis=1) / traj_lengths
    
    # Weight hour consistency more than day (0.4 for day, 0.6 for hour)
    combined_ce = 0.4 * np.mean(day_ce_avg) + 0.6 * np.mean(hour_ce_avg)
    
    return {
        'day_ce': np.mean(day_ce_avg),
        'hour_ce': np.mean(hour_ce_avg),
        'combined_ce': combined_ce
    }


def _categorical_crossentropy(y_true, y_pred):
    """
    Calculate categorical cross entropy between true and predicted distributions.
    
    Args:
        y_true: True distribution, shape [batch, seq_len, n_classes]
        y_pred: Predicted distribution, shape [batch, seq_len, n_classes]
        
    Returns:
        numpy.ndarray: Cross entropy per sample, shape [batch, seq_len]
    """
    return -np.sum(y_true * np.log(y_pred), axis=2)


def temporal_distribution_similarity(real_day, gen_day, real_hour, gen_hour):
    """
    Calculate JS divergence between real and generated temporal distributions.
    
    Args:
        real_day: Real day one-hot encodings, shape [batch, seq_len, 7]
        gen_day: Generated day one-hot encodings, shape [batch, seq_len, 7]
        real_hour: Real hour one-hot encodings, shape [batch, seq_len, 24]
        gen_hour: Generated hour one-hot encodings, shape [batch, seq_len, 24]
        
    Returns:
        dict: Dictionary with day and hour JS divergence scores
    """
    # Aggregate distributions across sequence dimension
    real_day_dist = np.sum(real_day, axis=1)
    gen_day_dist = np.sum(gen_day, axis=1)
    real_hour_dist = np.sum(real_hour, axis=1)
    gen_hour_dist = np.sum(gen_hour, axis=1)
    
    # Normalize to get probability distributions
    real_day_dist = real_day_dist / (np.sum(real_day_dist, axis=1, keepdims=True) + 1e-8)
    gen_day_dist = gen_day_dist / (np.sum(gen_day_dist, axis=1, keepdims=True) + 1e-8)
    real_hour_dist = real_hour_dist / (np.sum(real_hour_dist, axis=1, keepdims=True) + 1e-8)
    gen_hour_dist = gen_hour_dist / (np.sum(gen_hour_dist, axis=1, keepdims=True) + 1e-8)
    
    # Calculate JS divergence
    day_js_div = []
    hour_js_div = []
    
    for i in range(real_day_dist.shape[0]):
        day_js_div.append(_js_divergence(real_day_dist[i], gen_day_dist[i]))
        hour_js_div.append(_js_divergence(real_hour_dist[i], gen_hour_dist[i]))
    
    return {
        'day_js': np.mean(day_js_div),
        'hour_js': np.mean(hour_js_div),
        'combined_js': 0.4 * np.mean(day_js_div) + 0.6 * np.mean(hour_js_div)
    }


def _js_divergence(p, q):
    """
    Calculate Jensen-Shannon divergence between two distributions.
    
    Args:
        p: First distribution
        q: Second distribution
        
    Returns:
        float: JS divergence
    """
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


# === CATEGORY UTILITY METRICS ===

def category_utility_cross_entropy(real_cat, gen_cat, mask=None):
    """
    Calculate cross entropy between real and generated category distributions.
    
    Args:
        real_cat: Real category one-hot encodings, shape [batch, seq_len, n_cat]
        gen_cat: Generated category one-hot encodings, shape [batch, seq_len, n_cat]
        mask: Optional mask for valid points, shape [batch, seq_len, 1]
        
    Returns:
        float: Average cross entropy
    """
    if mask is None:
        # If no mask is provided, consider all points valid
        mask = np.ones((real_cat.shape[0], real_cat.shape[1], 1))
    
    # Clip generated values to avoid numerical issues
    gen_cat_clipped = np.clip(gen_cat, 1e-7, 1.0)
    
    # Calculate cross-entropy
    cat_ce = _categorical_crossentropy(real_cat, gen_cat_clipped)
    
    # Apply mask
    mask_flat = mask.squeeze(axis=2)
    cat_ce_masked = cat_ce * mask_flat
    
    # Calculate trajectory lengths
    traj_lengths = np.sum(mask, axis=(1, 2)) + 1e-8
    
    # Calculate average CE per trajectory
    cat_ce_avg = np.sum(cat_ce_masked, axis=1) / traj_lengths
    
    return np.mean(cat_ce_avg)


def category_distribution_similarity(real_cat, gen_cat):
    """
    Calculate JS divergence between real and generated category distributions.
    
    Args:
        real_cat: Real category one-hot encodings, shape [batch, seq_len, n_cat]
        gen_cat: Generated category one-hot encodings, shape [batch, seq_len, n_cat]
        
    Returns:
        float: JS divergence between category distributions
    """
    # Aggregate distributions across sequence dimension
    real_cat_dist = np.sum(real_cat, axis=1)
    gen_cat_dist = np.sum(gen_cat, axis=1)
    
    # Normalize to get probability distributions
    real_cat_dist = real_cat_dist / (np.sum(real_cat_dist, axis=1, keepdims=True) + 1e-8)
    gen_cat_dist = gen_cat_dist / (np.sum(gen_cat_dist, axis=1, keepdims=True) + 1e-8)
    
    # Calculate JS divergence
    cat_js_div = []
    
    for i in range(real_cat_dist.shape[0]):
        cat_js_div.append(_js_divergence(real_cat_dist[i], gen_cat_dist[i]))
    
    return np.mean(cat_js_div)


def category_transition_similarity(real_cat, gen_cat, mask=None):
    """
    Calculate similarity between category transition matrices.
    
    Args:
        real_cat: Real category one-hot encodings, shape [batch, seq_len, n_cat]
        gen_cat: Generated category one-hot encodings, shape [batch, seq_len, n_cat]
        mask: Optional mask for valid points, shape [batch, seq_len, 1]
        
    Returns:
        float: Frobenius norm of difference between transition matrices
    """
    if mask is None:
        # If no mask is provided, consider all points valid
        mask = np.ones((real_cat.shape[0], real_cat.shape[1], 1))
    
    n_cat = real_cat.shape[2]
    batch_size = real_cat.shape[0]
    
    # Convert one-hot to indices
    real_cat_idx = np.argmax(real_cat, axis=2)
    gen_cat_idx = np.argmax(gen_cat, axis=2)
    
    # Compute transition matrices
    real_trans = np.zeros((batch_size, n_cat, n_cat))
    gen_trans = np.zeros((batch_size, n_cat, n_cat))
    
    for b in range(batch_size):
        # Get valid indices based on mask
        valid_indices = np.where(mask[b, :, 0] > 0)[0]
        if len(valid_indices) < 2:
            continue
            
        for i in range(len(valid_indices) - 1):
            idx1 = valid_indices[i]
            idx2 = valid_indices[i + 1]
            real_trans[b, real_cat_idx[b, idx1], real_cat_idx[b, idx2]] += 1
            gen_trans[b, gen_cat_idx[b, idx1], gen_cat_idx[b, idx2]] += 1
        
        # Normalize to get transition probabilities
        row_sums_real = np.sum(real_trans[b], axis=1, keepdims=True)
        row_sums_gen = np.sum(gen_trans[b], axis=1, keepdims=True)
        
        # Avoid division by zero
        row_sums_real[row_sums_real == 0] = 1
        row_sums_gen[row_sums_gen == 0] = 1
        
        real_trans[b] = real_trans[b] / row_sums_real
        gen_trans[b] = gen_trans[b] / row_sums_gen
    
    # Compute Frobenius norm of difference
    norm_diffs = []
    for b in range(batch_size):
        # Skip if there are no transitions
        if np.sum(real_trans[b]) == 0 or np.sum(gen_trans[b]) == 0:
            continue
            
        norm_diff = np.linalg.norm(real_trans[b] - gen_trans[b], 'fro')
        norm_diffs.append(norm_diff)
    
    return np.mean(norm_diffs) if norm_diffs else float('inf')


# === COMBINED UTILITY METRICS ===

def compute_utility_metrics(real_trajs, gen_trajs, mask=None):
    """
    Compute comprehensive utility metrics for trajectories.
    
    Args:
        real_trajs: List of real trajectory tensors [lat_lon, day, hour, category, ...]
        gen_trajs: List of generated trajectory tensors [lat_lon, day, hour, category, ...]
        mask: Optional mask for valid points
        
    Returns:
        dict: Dictionary with all utility metrics
    """
    # Extract components from trajectory tensors
    real_latlon, real_day, real_hour, real_cat = real_trajs[:4]
    gen_latlon, gen_day, gen_hour, gen_cat = gen_trajs[:4]
    
    # Use mask from real_trajs if not provided
    if mask is None and len(real_trajs) > 4:
        mask = real_trajs[4]
    
    # Compute spatial metrics
    spatial_mse = spatial_utility_mse(real_latlon, gen_latlon, mask)
    hausdorff = hausdorff_distance(real_latlon, gen_latlon, mask)
    dtw = dtw_distance(real_latlon, gen_latlon, mask)
    
    # Compute temporal metrics
    temporal_ce = temporal_utility_cross_entropy(real_day, gen_day, real_hour, gen_hour, mask)
    temporal_js = temporal_distribution_similarity(real_day, gen_day, real_hour, gen_hour)
    
    # Compute category metrics
    category_ce = category_utility_cross_entropy(real_cat, gen_cat, mask)
    category_js = category_distribution_similarity(real_cat, gen_cat)
    category_trans = category_transition_similarity(real_cat, gen_cat, mask)
    
    # Compile all metrics
    metrics = {
        # Spatial metrics
        'spatial_mse': spatial_mse,
        'hausdorff_distance': hausdorff,
        'dtw_distance': dtw,
        
        # Temporal metrics
        'day_ce': temporal_ce['day_ce'],
        'hour_ce': temporal_ce['hour_ce'],
        'temporal_ce': temporal_ce['combined_ce'],
        'day_js': temporal_js['day_js'],
        'hour_js': temporal_js['hour_js'],
        'temporal_js': temporal_js['combined_js'],
        
        # Category metrics
        'category_ce': category_ce,
        'category_js': category_js,
        'category_trans_diff': category_trans,
        
        # Combined metrics (weighted average - adjust weights as needed)
        'combined_utility': -(
            0.6 * (spatial_mse / 10.0) +  # Normalize spatial MSE
            0.2 * temporal_ce['combined_ce'] + 
            0.2 * category_ce
        )
    }
    
    return metrics


# === PRIVACY METRICS HELPER FUNCTIONS ===

def compute_privacy_metrics(classifier, gen_trajs, true_labels, print_metrics=True):
    """
    Compute privacy metrics using a classifier.
    
    Args:
        classifier: Trained classifier model
        gen_trajs: Generated trajectories
        true_labels: True labels for classification task
        print_metrics: Whether to print metrics
        
    Returns:
        dict: Dictionary with privacy metrics
    """
    # Use classifier to predict labels
    pred_probs = classifier.predict(gen_trajs)
    
    # Compute privacy metrics
    acc, acc5, f1_macro, prec_macro, rec_macro = compute_acc_acc5_f1_prec_rec(
        true_labels, pred_probs, print_metrics=print_metrics, print_pfx='PRIVACY'
    )
    
    # Return metrics dictionary
    return {
        'test_acc': acc,
        'test_acc5': acc5,
        'test_f1_macro': f1_macro,
        'test_prec_macro': prec_macro,
        'test_rec_macro': rec_macro
    }


def evaluate_trajectory_privacy(gen_trajs, cls_y_test, pred_y_test=None, classifier=None):
    """
    Evaluate trajectory privacy using either predictions or a classifier.
    
    Args:
        gen_trajs: Generated trajectories
        cls_y_test: True test labels
        pred_y_test: Pre-computed predictions (optional)
        classifier: Classifier model (optional if pred_y_test is provided)
        
    Returns:
        dict: Dictionary with privacy metrics
    """
    if pred_y_test is None:
        if classifier is None:
            raise ValueError("Either pred_y_test or classifier must be provided")
        
        # Generate predictions using classifier
        pred_y_test = classifier.predict(gen_trajs)
    
    # Compute privacy metrics
    test_acc, test_acc5, test_f1_macro, test_prec_macro, test_rec_macro = compute_acc_acc5_f1_prec_rec(
        cls_y_test, pred_y_test, print_metrics=True, print_pfx='TEST'
    )
    
    print(f"test_acc: {test_acc}")
    print(f"test_acc5: {test_acc5}")
    print(f"test_f1_macro: {test_f1_macro}")
    print(f"test_prec_macro: {test_prec_macro}")
    print(f"test_rec_macro: {test_rec_macro}")
    
    return {
        'test_acc': test_acc,
        'test_acc5': test_acc5,
        'test_f1_macro': test_f1_macro,
        'test_prec_macro': test_prec_macro, 
        'test_rec_macro': test_rec_macro
    } 