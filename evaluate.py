import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.preprocessing.sequence import pad_sequences
from scipy.linalg import sqrtm
from scipy.spatial.distance import jensenshannon

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import relevant modules
try:
    from MARC.marc import MARC
except ImportError:
    print("Warning: MARC module not found. TUL evaluation may not work properly.")


def prepare_trajectories_for_tul(traj_df, max_length=144):
    """
    Prepare trajectory data for the TUL classifier.
    
    Args:
        traj_df: DataFrame containing trajectory data
        max_length: Maximum length of trajectories for padding
        
    Returns:
        inputs: List of inputs formatted for the TUL classifier
    """
    # Ensure the dataframe has the expected columns
    required_cols = ['tid', 'label', 'lat', 'lon', 'day', 'hour', 'category']
    for col in required_cols:
        if col not in traj_df.columns:
            raise ValueError(f"Required column '{col}' not found in trajectory data")
    
    # Group by trajectory ID
    traj_grouped = traj_df.groupby('tid')
    
    # Initialize lists for each feature
    lat_lon_list = []
    day_list = []
    hour_list = []
    category_list = []
    labels = []
    
    # Process each trajectory
    for tid, group in traj_grouped:
        # Extract features
        lat_lon = np.array(group[['lat', 'lon']])
        day = np.array(group['day'])
        hour = np.array(group['hour'])
        category = np.array(group['category'])
        label = group['label'].iloc[0]
        
        # Append to lists
        lat_lon_list.append(lat_lon)
        day_list.append(day)
        hour_list.append(hour)
        category_list.append(category)
        labels.append(label)
    
    # Convert day, hour, category to indices format
    day_indices = [np.array(d) for d in day_list]
    hour_indices = [np.array(h) for h in hour_list]
    category_indices = [np.array(c) for c in category_list]
    
    # Pad sequences
    day_padded = pad_sequences(day_indices, maxlen=max_length, padding='pre', dtype='int32')
    hour_padded = pad_sequences(hour_indices, maxlen=max_length, padding='pre', dtype='int32')
    category_padded = pad_sequences(category_indices, maxlen=max_length, padding='pre', dtype='int32')
    
    # Pad lat_lon and expand to 40 dimensions (as expected by MARC)
    lat_lon_padded_list = []
    for ll in lat_lon_list:
        # Pad to max_length
        padded = np.zeros((max_length, 2))
        padded[-len(ll):] = ll
        # Expand to 40 dimensions
        expanded = np.pad(padded, ((0, 0), (0, 38)))
        lat_lon_padded_list.append(expanded)
    
    lat_lon_padded = np.array(lat_lon_padded_list)
    
    return [day_padded, hour_padded, category_padded, lat_lon_padded], np.array(labels)


def load_tul_classifier():
    """
    Load the pre-trained TUL classifier.
    
    Returns:
        tul_model: Loaded TUL classifier model
    """
    try:
        # Create and initialize the MARC model
        marc_model = MARC()
        marc_model.build_model()
        
        # Load pre-trained weights
        marc_model.load_weights('MARC/weights/MARC_Weight.h5')
        
        print("Successfully loaded TUL classifier")
        return marc_model
    except Exception as e:
        print(f"Error loading TUL classifier: {e}")
        raise


def evaluate_privacy(real_inputs, synthetic_inputs, real_labels, synthetic_labels):
    """
    Evaluate privacy metrics using the Trajectory-User Linking (TUL) task.
    
    Args:
        real_inputs: Prepared inputs for real trajectories
        synthetic_inputs: Prepared inputs for synthetic trajectories
        real_labels: Ground truth labels for real trajectories
        synthetic_labels: Ground truth labels for synthetic trajectories
        
    Returns:
        metrics: Dictionary of privacy metrics
    """
    # Load TUL classifier
    tul_classifier = load_tul_classifier()
    
    # Get predictions
    print("Getting TUL predictions for real trajectories...")
    real_preds = tul_classifier(real_inputs)
    
    print("Getting TUL predictions for synthetic trajectories...")
    synthetic_preds = tul_classifier(synthetic_inputs)
    
    # Get the top-k predictions
    k_values = [1, 5]
    
    privacy_metrics = {}
    
    # Calculate metrics for real and synthetic data
    for dataset_name, preds, labels in [("real", real_preds, real_labels), 
                                        ("synthetic", synthetic_preds, synthetic_labels)]:
        
        # Top-k accuracy
        for k in k_values:
            top_k_indices = np.argsort(preds, axis=1)[:, -k:]
            
            correct = np.array([labels[i] in top_k_indices[i] for i in range(len(labels))])
            acc_k = np.mean(correct)
            
            privacy_metrics[f"{dataset_name}_acc@{k}"] = acc_k
        
        # Get the predicted labels (top-1)
        pred_labels = np.argmax(preds, axis=1)
        
        # Calculate precision, recall, and F1 score
        precision_macro = precision_score(labels, pred_labels, average='macro')
        recall_macro = recall_score(labels, pred_labels, average='macro')
        f1_macro = f1_score(labels, pred_labels, average='macro')
        
        privacy_metrics[f"{dataset_name}_macro_precision"] = precision_macro
        privacy_metrics[f"{dataset_name}_macro_recall"] = recall_macro
        privacy_metrics[f"{dataset_name}_macro_f1"] = f1_macro
    
    return privacy_metrics


def calculate_fid(real_features, synthetic_features):
    """
    Calculate Fréchet Inception Distance between real and synthetic features.
    
    Args:
        real_features: Features from real trajectories
        synthetic_features: Features from synthetic trajectories
        
    Returns:
        fid: Fréchet Inception Distance score
    """
    # Calculate mean and covariance for real features
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    
    # Calculate mean and covariance for synthetic features
    mu2 = np.mean(synthetic_features, axis=0)
    sigma2 = np.cov(synthetic_features, rowvar=False)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary component if necessary
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    
    return fid


def extract_features(trajectories):
    """
    Extract features from trajectories for utility evaluation.
    
    Args:
        trajectories: DataFrame containing trajectory data
        
    Returns:
        features: Extracted features
    """
    # Group by trajectory ID
    traj_grouped = trajectories.groupby('tid')
    
    features = []
    
    for tid, group in traj_grouped:
        # Extract statistical features
        lat_mean = group['lat'].mean()
        lat_std = group['lat'].std()
        lon_mean = group['lon'].mean()
        lon_std = group['lon'].std()
        
        # Calculate trajectory length
        traj_length = len(group)
        
        # Calculate average displacement
        lat_diffs = np.diff(group['lat'].values)
        lon_diffs = np.diff(group['lon'].values)
        avg_displacement = np.mean(np.sqrt(lat_diffs**2 + lon_diffs**2)) if len(lat_diffs) > 0 else 0
        
        # Day and hour distribution
        day_hist = np.bincount(group['day'].astype(int), minlength=7)
        day_hist = day_hist / (np.sum(day_hist) + 1e-10)  # Normalize
        
        hour_hist = np.bincount(group['hour'].astype(int), minlength=24)
        hour_hist = hour_hist / (np.sum(hour_hist) + 1e-10)  # Normalize
        
        # Category distribution
        max_category = max(trajectories['category'].max(), 1)
        category_hist = np.bincount(group['category'].astype(int), minlength=max_category+1)
        category_hist = category_hist / (np.sum(category_hist) + 1e-10)  # Normalize
        
        # Combine features
        traj_features = np.concatenate([
            [lat_mean, lat_std, lon_mean, lon_std, traj_length, avg_displacement],
            day_hist,
            hour_hist,
            category_hist
        ])
        
        features.append(traj_features)
    
    return np.array(features)


def calculate_jsd(real_dist, synthetic_dist):
    """
    Calculate Jensen-Shannon Divergence between real and synthetic distributions.
    
    Args:
        real_dist: Distribution from real trajectories
        synthetic_dist: Distribution from synthetic trajectories
        
    Returns:
        jsd: Jensen-Shannon Divergence score
    """
    # Ensure distributions sum to 1
    real_dist = real_dist / (np.sum(real_dist) + 1e-10)
    synthetic_dist = synthetic_dist / (np.sum(synthetic_dist) + 1e-10)
    
    # Calculate JSD
    jsd = jensenshannon(real_dist, synthetic_dist)
    
    return jsd


def evaluate_utility(real_df, synthetic_df):
    """
    Evaluate utility metrics between real and synthetic trajectories.
    
    Args:
        real_df: DataFrame containing real trajectory data
        synthetic_df: DataFrame containing synthetic trajectory data
        
    Returns:
        metrics: Dictionary of utility metrics
    """
    # Extract features for FID calculation
    real_features = extract_features(real_df)
    synthetic_features = extract_features(synthetic_df)
    
    # Calculate FID
    fid_score = calculate_fid(real_features, synthetic_features)
    
    # Calculate JSD for different attributes
    utility_metrics = {'fid_score': fid_score}
    
    # Day distribution
    real_day_dist = np.bincount(real_df['day'].astype(int), minlength=7)
    synthetic_day_dist = np.bincount(synthetic_df['day'].astype(int), minlength=7)
    day_jsd = calculate_jsd(real_day_dist, synthetic_day_dist)
    utility_metrics['day_jsd'] = day_jsd
    
    # Hour distribution
    real_hour_dist = np.bincount(real_df['hour'].astype(int), minlength=24)
    synthetic_hour_dist = np.bincount(synthetic_df['hour'].astype(int), minlength=24)
    hour_jsd = calculate_jsd(real_hour_dist, synthetic_hour_dist)
    utility_metrics['hour_jsd'] = hour_jsd
    
    # Category distribution
    max_category = max(real_df['category'].max(), synthetic_df['category'].max(), 1)
    real_category_dist = np.bincount(real_df['category'].astype(int), minlength=max_category+1)
    synthetic_category_dist = np.bincount(synthetic_df['category'].astype(int), minlength=max_category+1)
    category_jsd = calculate_jsd(real_category_dist, synthetic_category_dist)
    utility_metrics['category_jsd'] = category_jsd
    
    # Calculate overall JSD as average
    overall_jsd = (day_jsd + hour_jsd + category_jsd) / 3
    utility_metrics['overall_jsd'] = overall_jsd
    
    return utility_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate privacy and utility of synthetic trajectories')
    parser.add_argument('--real', type=str, default='/root/autodl-tmp/location-privacy/data/test_latlon.csv', 
                        help='Path to real trajectory data CSV')
    parser.add_argument('--synthetic', type=str, default='/root/autodl-tmp/location-privacy/results/syn_traj_test.csv', 
                        help='Path to synthetic trajectory data CSV')
    parser.add_argument('--output', type=str, default='results/evaluation_metrics.csv', 
                        help='Path to save evaluation metrics')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Loading real data from {args.real}")
    print(f"Loading synthetic data from {args.synthetic}")
    
    # Load trajectory data
    real_df = pd.read_csv(args.real)
    synthetic_df = pd.read_csv(args.synthetic)
    
    # Ensure column order matches (handling the difference in column order)
    if list(synthetic_df.columns) != list(real_df.columns):
        print(f"Reordering synthetic columns to match real data")
        print(f"Real columns: {real_df.columns.tolist()}")
        print(f"Synthetic columns: {synthetic_df.columns.tolist()}")
        
        # Try to reorder synthetic columns to match real data
        try:
            synthetic_df = synthetic_df[real_df.columns]
        except KeyError:
            # If the column names don't match exactly, handle it
            print("Column names don't match exactly. Mapping columns...")
            
            # The test csv includes columns: tid,label,lat,lon,day,hour,category
            # The synthetic csv includes column: label,tid,lat,lon,day,hour,category
            synthetic_df = synthetic_df[['label', 'tid', 'lat', 'lon', 'day', 'hour', 'category']]
            synthetic_df = synthetic_df[['tid', 'label', 'lat', 'lon', 'day', 'hour', 'category']]
    
    print(f"Real data shape: {real_df.shape}")
    print(f"Synthetic data shape: {synthetic_df.shape}")
    
    # Prepare data for TUL evaluation
    print("Preparing data for privacy evaluation...")
    real_inputs, real_labels = prepare_trajectories_for_tul(real_df)
    synthetic_inputs, synthetic_labels = prepare_trajectories_for_tul(synthetic_df)
    
    # Evaluate privacy
    print("Evaluating privacy metrics...")
    privacy_metrics = evaluate_privacy(real_inputs, synthetic_inputs, real_labels, synthetic_labels)
    
    # Evaluate utility
    print("Evaluating utility metrics...")
    utility_metrics = evaluate_utility(real_df, synthetic_df)
    
    # Combine metrics
    all_metrics = {**privacy_metrics, **utility_metrics}
    
    # Save metrics
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(args.output, index=False)
    
    # Print evaluation results
    print("\nPrivacy Evaluation Metrics:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Real':<10} {'Synthetic':<10}")
    print("-" * 50)
    for k in [1, 5]:
        print(f"ACC@{k:<24} {privacy_metrics[f'real_acc@{k}']:.4f}    {privacy_metrics[f'synthetic_acc@{k}']:.4f}")
    
    print(f"Macro Precision{' ':14} {privacy_metrics['real_macro_precision']:.4f}    {privacy_metrics['synthetic_macro_precision']:.4f}")
    print(f"Macro Recall{' ':17} {privacy_metrics['real_macro_recall']:.4f}    {privacy_metrics['synthetic_macro_recall']:.4f}")
    print(f"Macro F1{' ':21} {privacy_metrics['real_macro_f1']:.4f}    {privacy_metrics['synthetic_macro_f1']:.4f}")
    
    print("\nUtility Evaluation Metrics:")
    print("-" * 50)
    print(f"FID Score: {utility_metrics['fid_score']:.4f}")
    print(f"Day JSD: {utility_metrics['day_jsd']:.4f}")
    print(f"Hour JSD: {utility_metrics['hour_jsd']:.4f}")
    print(f"Category JSD: {utility_metrics['category_jsd']:.4f}")
    print(f"Overall JSD: {utility_metrics['overall_jsd']:.4f}")
    
    print(f"\nAll metrics saved to {args.output}")


if __name__ == '__main__':
    main() 