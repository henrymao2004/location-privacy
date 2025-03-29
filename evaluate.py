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


def prepare_trajectories_for_tul(traj_df, max_length=144, label_col='label'):
    """
    Prepare trajectory data for the TUL classifier.
    
    Args:
        traj_df: DataFrame containing trajectory data
        max_length: Maximum length of trajectories for padding
        label_col: Name of the column containing label information
        
    Returns:
        inputs: List of inputs formatted for the TUL classifier
    """
    # Create a copy of the dataframe to avoid modifying the original
    df = traj_df.copy()
    
    # If the specified label column doesn't exist but 'point_idx' does, use that instead
    if label_col not in df.columns and 'point_idx' in df.columns:
        df[label_col] = df['point_idx']
        print(f"Using 'point_idx' as the label column")
    
    # Ensure the dataframe has the expected columns
    required_cols = ['tid', label_col, 'lat', 'lon', 'day', 'hour', 'category']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in trajectory data. Available columns: {df.columns.tolist()}")
    
    # Validate and fix values to valid ranges
    # Days should be 0-6 (0=Monday, 6=Sunday)
    if df['day'].max() > 6:
        print(f"Warning: 'day' column contains values outside range [0,6]. Max value found: {df['day'].max()}")
        # For values > 6, we should take modulo 7 to wrap around to valid days
        df['day'] = df['day'] % 7
        print(f"Fixed day values distribution: {df['day'].value_counts().sort_index()}")
    
    # Hours should be 0-23
    if df['hour'].max() > 23:
        print(f"Warning: 'hour' column contains values outside range [0,23]. Max value found: {df['hour'].max()}")
        # For values > 23, we should take modulo 24 to wrap around to valid hours
        df['hour'] = df['hour'] % 24
        print(f"Fixed hour values distribution: {df['hour'].value_counts().sort_index()}")
    
    # Group by trajectory ID
    traj_grouped = df.groupby('tid')
    
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
        day = np.array(group['day']).astype(np.int32)  # Ensure correct dtype
        hour = np.array(group['hour']).astype(np.int32)  # Ensure correct dtype
        category = np.array(group['category']).astype(np.int32)  # Ensure correct dtype
        label = group[label_col].iloc[0]
        
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
        
        try:
            # Calculate precision, recall, and F1 score with zero_division=0
            precision_macro = precision_score(labels, pred_labels, average='macro', zero_division=0)
            recall_macro = recall_score(labels, pred_labels, average='macro', zero_division=0)
            f1_macro = f1_score(labels, pred_labels, average='macro', zero_division=0)
            
            privacy_metrics[f"{dataset_name}_macro_precision"] = precision_macro
            privacy_metrics[f"{dataset_name}_macro_recall"] = recall_macro
            privacy_metrics[f"{dataset_name}_macro_f1"] = f1_macro
        except Exception as e:
            print(f"Error calculating metrics for {dataset_name}: {e}")
            # Set default values
            privacy_metrics[f"{dataset_name}_macro_precision"] = 0.0
            privacy_metrics[f"{dataset_name}_macro_recall"] = 0.0
            privacy_metrics[f"{dataset_name}_macro_f1"] = 0.0
    
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
    try:
        # Calculate mean and covariance for real features
        mu1 = np.mean(real_features, axis=0)
        sigma1 = np.cov(real_features, rowvar=False)
        
        # Calculate mean and covariance for synthetic features
        mu2 = np.mean(synthetic_features, axis=0)
        sigma2 = np.cov(synthetic_features, rowvar=False)
        
        # Calculate FID
        diff = mu1 - mu2
        
        # Handle potential numerical issues in computing square root
        try:
            covmean = sqrtm(sigma1.dot(sigma2))
            # Check and correct imaginary component if necessary
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        except Exception as e:
            print(f"Failed to compute square root: {e}")
            # Fallback calculation without square root term
            fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2)
            print(f"Using alternative FID calculation: {fid}")
        
        return fid
    except Exception as e:
        print(f"Error calculating FID: {e}")
        return float('inf')  # Return worst-case value on error


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
    
    # Determine maximum category value across all trajectories
    max_category = int(trajectories['category'].max())
    print(f"Maximum category value: {max_category}")
    
    # Use a fixed category size to ensure consistent feature dimensions
    fixed_category_size = 100  # Choose a size that's large enough for all possible categories
    
    features = []
    
    for tid, group in traj_grouped:
        try:
            # Extract statistical features
            lat_mean = group['lat'].mean()
            lat_std = group['lat'].std() if len(group) > 1 else 0
            lon_mean = group['lon'].mean()
            lon_std = group['lon'].std() if len(group) > 1 else 0
            
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
            
            # Category distribution with fixed size
            category_hist = np.zeros(fixed_category_size)
            actual_hist = np.bincount(group['category'].astype(int), minlength=max_category+1)
            category_hist[:len(actual_hist)] = actual_hist[:fixed_category_size]
            category_hist = category_hist / (np.sum(category_hist) + 1e-10)  # Normalize
            
            # Combine features
            traj_features = np.concatenate([
                [lat_mean, lat_std, lon_mean, lon_std, traj_length, avg_displacement],
                day_hist,
                hour_hist,
                category_hist
            ])
            
            features.append(traj_features)
        except Exception as e:
            print(f"Error processing trajectory {tid}: {e}")
            # Skip this trajectory
    
    if not features:
        raise ValueError("No valid features could be extracted from trajectories")
    
    # Verify all features have the same length
    feature_lengths = [len(f) for f in features]
    if len(set(feature_lengths)) > 1:
        print(f"Warning: inconsistent feature lengths detected: {set(feature_lengths)}")
        # Find the most common length
        from collections import Counter
        most_common_length = Counter(feature_lengths).most_common(1)[0][0]
        # Filter features to keep only those with the most common length
        features = [f for f in features if len(f) == most_common_length]
        print(f"Kept {len(features)} features with length {most_common_length}")
    
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
    # Ensure both distributions have the same length
    max_len = max(len(real_dist), len(synthetic_dist))
    
    # Pad both distributions to the same length
    padded_real = np.zeros(max_len)
    padded_real[:len(real_dist)] = real_dist
    
    padded_synthetic = np.zeros(max_len)
    padded_synthetic[:len(synthetic_dist)] = synthetic_dist
    
    # Ensure distributions sum to 1
    padded_real = padded_real / (np.sum(padded_real) + 1e-10)
    padded_synthetic = padded_synthetic / (np.sum(padded_synthetic) + 1e-10)
    
    # Calculate JSD
    try:
        jsd = jensenshannon(padded_real, padded_synthetic)
        return jsd
    except Exception as e:
        print(f"Error calculating JSD: {e}")
        return 1.0  # Return worst-case value on error


def evaluate_utility(real_df, synthetic_df):
    """
    Evaluate utility metrics between real and synthetic trajectories.
    
    Args:
        real_df: DataFrame containing real trajectory data
        synthetic_df: DataFrame containing synthetic trajectory data
        
    Returns:
        metrics: Dictionary of utility metrics
    """
    try:
        # Extract features
        real_features = extract_features(real_df)
        synthetic_features = extract_features(synthetic_df)
        
        print(f"Real features shape: {real_features.shape}")
        print(f"Synthetic features shape: {synthetic_features.shape}")
        
        # Ensure features have the same dimensions
        min_features = min(real_features.shape[1], synthetic_features.shape[1])
        
        # Truncate features to the same length
        real_features_truncated = real_features[:, :min_features]
        synthetic_features_truncated = synthetic_features[:, :min_features]
        
        print(f"Using truncated features with length {min_features}")
        
        # Calculate FID
        fid_score = calculate_fid(real_features_truncated, synthetic_features_truncated)
        
        # Calculate JSD for different attributes
        utility_metrics = {'fid_score': fid_score}
        
        # Ensure day values are in [0, 6]
        real_df_day = real_df.copy()
        synthetic_df_day = synthetic_df.copy()
        real_df_day['day'] = real_df_day['day'].clip(0, 6)
        synthetic_df_day['day'] = synthetic_df_day['day'].clip(0, 6)
        
        # Day distribution - ensure same length
        real_day_dist = np.bincount(real_df_day['day'].astype(int), minlength=7)
        synthetic_day_dist = np.bincount(synthetic_df_day['day'].astype(int), minlength=7)
        day_jsd = calculate_jsd(real_day_dist, synthetic_day_dist)
        utility_metrics['day_jsd'] = day_jsd
        
        # Ensure hour values are in [0, 23]
        real_df_hour = real_df.copy()
        synthetic_df_hour = synthetic_df.copy()
        real_df_hour['hour'] = real_df_hour['hour'].clip(0, 23)
        synthetic_df_hour['hour'] = synthetic_df_hour['hour'].clip(0, 23)
        
        # Hour distribution - ensure same length
        real_hour_dist = np.bincount(real_df_hour['hour'].astype(int), minlength=24)
        synthetic_hour_dist = np.bincount(synthetic_df_hour['hour'].astype(int), minlength=24)
        hour_jsd = calculate_jsd(real_hour_dist, synthetic_hour_dist)
        utility_metrics['hour_jsd'] = hour_jsd
        
        # Category distribution - use the same max category for both
        max_category = max(max(real_df['category'].max(), 1), max(synthetic_df['category'].max(), 1))
        real_category_dist = np.bincount(real_df['category'].astype(int), minlength=max_category+1)
        synthetic_category_dist = np.bincount(synthetic_df['category'].astype(int), minlength=max_category+1)
        category_jsd = calculate_jsd(real_category_dist, synthetic_category_dist)
        utility_metrics['category_jsd'] = category_jsd
        
        # Calculate overall JSD as average
        overall_jsd = (day_jsd + hour_jsd + category_jsd) / 3
        utility_metrics['overall_jsd'] = overall_jsd
        
        return utility_metrics
    except Exception as e:
        print(f"Error in evaluate_utility: {e}")
        # Return default metrics
        return {
            'fid_score': float('inf'),
            'day_jsd': 1.0,
            'hour_jsd': 1.0,
            'category_jsd': 1.0,
            'overall_jsd': 1.0
        }


def main():
    parser = argparse.ArgumentParser(description='Evaluate privacy and utility of synthetic trajectories')
    parser.add_argument('--real', type=str, default='/root/autodl-tmp/location-privacy/data/test_latlon.csv', 
                        help='Path to real trajectory data CSV')
    parser.add_argument('--synthetic', type=str, default='/root/autodl-tmp/location-privacy/results/synthetic_trajectories_epoch110.csv', 
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
    
    print(f"Real data shape: {real_df.shape}")
    print(f"Synthetic data shape: {synthetic_df.shape}")
    
    # Print column names for debugging
    print(f"Real data columns: {real_df.columns.tolist()}")
    print(f"Synthetic data columns: {synthetic_df.columns.tolist()}")
    
    # Prepare data for TUL evaluation
    print("Preparing data for privacy evaluation...")
    real_inputs, real_labels = prepare_trajectories_for_tul(real_df)
    synthetic_inputs, synthetic_labels = prepare_trajectories_for_tul(synthetic_df, label_col='point_idx')
    
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