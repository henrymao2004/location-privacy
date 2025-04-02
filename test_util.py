import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from math import radians
import os
import argparse

def calculate_spatial_utility(real_traj, gen_traj):
    """
    Calculate spatial utility between real and generated trajectories.
    
    Args:
        real_traj: DataFrame containing real trajectory with 'lat' and 'lon' columns
        gen_traj: DataFrame containing generated trajectory with 'lat' and 'lon' columns
        
    Returns:
        Float representing spatial similarity (1.0 = identical, 0.0 = completely different)
    """
    # Check if trajectories have the same length
    if len(real_traj) != len(gen_traj):
        min_length = min(len(real_traj), len(gen_traj))
        real_traj = real_traj.iloc[:min_length]
        gen_traj = gen_traj.iloc[:min_length]
    
    # Convert lat/lon to radians for haversine distance calculation
    real_coords = np.array([[radians(lat), radians(lon)] for lat, lon in zip(real_traj['lat'], real_traj['lon'])])
    gen_coords = np.array([[radians(lat), radians(lon)] for lat, lon in zip(gen_traj['lat'], gen_traj['lon'])])
    
    # Calculate distances in km (earth radius ≈ 6371 km)
    distances = haversine_distances(real_coords, gen_coords) * 6371.0
    
    # Extract distances between corresponding points
    point_distances = np.diagonal(distances)
    
    # Convert distances to similarity scores (1.0 for 0km, decreasing as distance increases)
    # Using an exponential decay function: e^(-distance/scale)
    # where scale is a parameter that determines how quickly similarity drops with distance
    scale = 1.0  # 1km scale - adjust based on your application
    similarities = np.exp(-point_distances/scale)
    
    # Overall spatial utility is the average similarity
    spatial_utility = np.mean(similarities)
    
    return spatial_utility

def calculate_temporal_utility(real_traj, gen_traj):
    """
    Calculate temporal utility between real and generated trajectories.
    
    Args:
        real_traj: DataFrame containing real trajectory with temporal features
        gen_traj: DataFrame containing generated trajectory with temporal features
        
    Returns:
        Float representing temporal similarity (1.0 = identical, 0.0 = completely different)
    """
    # Match length if needed
    if len(real_traj) != len(gen_traj):
        min_length = min(len(real_traj), len(gen_traj))
        real_traj = real_traj.iloc[:min_length]
        gen_traj = gen_traj.iloc[:min_length]
    
    # Hour difference considering circular nature (23 and 0 are 1 hour apart, not 23)
    hour_diffs = np.minimum(
        np.abs(real_traj['hour'].values - gen_traj['hour'].values),
        24 - np.abs(real_traj['hour'].values - gen_traj['hour'].values)
    )
    
    # Convert to similarity (1.0 for 0 hours difference, 0.0 for 12 hours difference)
    hour_sim = np.mean(1.0 - (hour_diffs / 12.0))
    
    # Day difference considering circular nature of the week
    day_diffs = np.minimum(
        np.abs(real_traj['day'].values - gen_traj['day'].values),
        7 - np.abs(real_traj['day'].values - gen_traj['day'].values)
    )
    
    # Convert to similarity (1.0 for 0 days difference, 0.0 for 3-4 days difference)
    day_sim = np.mean(1.0 - (day_diffs / 3.5))
    
    # Overall temporal utility is the average of day and hour similarities
    temporal_utility = (day_sim + hour_sim) / 2.0
    
    return temporal_utility, day_sim, hour_sim

def calculate_category_utility(real_traj, gen_traj):
    """
    Calculate category utility between real and generated trajectories.
    
    Args:
        real_traj: DataFrame containing real trajectory with 'category' column
        gen_traj: DataFrame containing generated trajectory with 'category' column
        
    Returns:
        Float representing category similarity (1.0 = identical, 0.0 = completely different)
    """
    # Match length if needed
    if len(real_traj) != len(gen_traj):
        min_length = min(len(real_traj), len(gen_traj))
        real_traj = real_traj.iloc[:min_length]
        gen_traj = gen_traj.iloc[:min_length]
    
    # Calculate exact category matches
    exact_matches = (real_traj['category'].values == gen_traj['category'].values)
    category_utility = np.mean(exact_matches)
    
    return category_utility

def calculate_overall_utility(spatial_utility, temporal_utility, category_utility):
    """
    Calculate overall utility as a weighted combination of individual utilities.
    
    Args:
        spatial_utility: Float representing spatial utility
        temporal_utility: Float representing temporal utility
        category_utility: Float representing category utility
        
    Returns:
        Float representing overall utility
    """
    # Define weights for each utility component
    spatial_weight = 1.5    # Higher weight for spatial aspects
    temporal_weight = 1.0   # Standard weight for temporal aspects
    category_weight = 1.0   # Standard weight for category aspects
    
    # Calculate weighted average
    weighted_sum = (spatial_utility * spatial_weight + 
                   temporal_utility * temporal_weight + 
                   category_utility * category_weight)
    
    total_weight = spatial_weight + temporal_weight + category_weight
    
    overall_utility = weighted_sum / total_weight
        
    return overall_utility

def match_by_label(real_df, gen_df):
    """
    Match trajectories by label when tids don't match.
    
    Args:
        real_df: DataFrame containing real trajectories
        gen_df: DataFrame containing generated trajectories
        
    Returns:
        List of tuples (real_tid, gen_tid, label) for matched trajectories
    """
    # Get unique labels
    real_labels = real_df['label'].unique()
    gen_labels = gen_df['label'].unique()
    
    # Find common labels
    common_labels = np.intersect1d(real_labels, gen_labels)
    print(f"Found {len(common_labels)} common labels to evaluate")
    
    # Create matched pairs
    matched_pairs = []
    
    for label in common_labels:
        real_tids = real_df[real_df['label'] == label]['tid'].unique()
        gen_tids = gen_df[gen_df['label'] == label]['tid'].unique()
        
        # Match one real trajectory with one generated trajectory for each label
        for i in range(min(len(real_tids), len(gen_tids))):
            matched_pairs.append((real_tids[i], gen_tids[i], label))
    
    print(f"Created {len(matched_pairs)} matched trajectory pairs based on labels")
    return matched_pairs

def evaluate_trajectory_utility(real_traj_file, gen_traj_file, output_file=None):
    """
    Evaluate utility performance between real and generated trajectories.
    
    Args:
        real_traj_file: Path to CSV file containing real trajectories
        gen_traj_file: Path to CSV file containing generated trajectories
        output_file: Optional path to save results
        
    Returns:
        Dictionary containing utility metrics
    """
    # Load trajectory data
    real_df = pd.read_csv(real_traj_file)
    gen_df = pd.read_csv(gen_traj_file)
    
    # Ensure column order is consistent
    required_cols = ['tid', 'label', 'lat', 'lon', 'day', 'hour', 'category']
    
    # Fix column order if needed
    for df in [real_df, gen_df]:
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")
    
    # Print dataset information
    print(f"Real trajectories file: {real_traj_file}")
    print(f"Generated trajectories file: {gen_traj_file}")
    print(f"Real data rows: {len(real_df)}, columns: {real_df.columns.tolist()}")
    print(f"Generated data rows: {len(gen_df)}, columns: {gen_df.columns.tolist()}")
    
    # Get unique trajectory IDs
    real_tids = real_df['tid'].unique()
    gen_tids = gen_df['tid'].unique()
    
    print(f"Real trajectories: {len(real_tids)}")
    print(f"Generated trajectories: {len(gen_tids)}")
    
    # Find common trajectory IDs
    common_tids = np.intersect1d(real_tids, gen_tids)
    print(f"Found {len(common_tids)} common trajectory IDs")
    
    # If no common tids, try matching by label
    if len(common_tids) == 0:
        print("No common trajectory IDs found. Matching trajectories by label...")
        matched_pairs = match_by_label(real_df, gen_df)
        
        if not matched_pairs:
            raise ValueError("Could not match any trajectories by label")
    else:
        # Use common tids directly
        matched_pairs = [(tid, tid, real_df[real_df['tid'] == tid]['label'].iloc[0]) for tid in common_tids]
    
    # Initialize results
    results = {
        'real_tid': [],
        'gen_tid': [],
        'label': [],
        'spatial_utility': [],
        'temporal_utility': [],
        'day_utility': [],
        'hour_utility': [],
        'category_utility': [],
        'overall_utility': []
    }
    
    # Process each trajectory pair
    for real_tid, gen_tid, label in matched_pairs:
        real_traj = real_df[real_df['tid'] == real_tid]
        gen_traj = gen_df[gen_df['tid'] == gen_tid]
        
        # Calculate utilities
        spatial_util = calculate_spatial_utility(real_traj, gen_traj)
        temporal_util, day_util, hour_util = calculate_temporal_utility(real_traj, gen_traj)
        category_util = calculate_category_utility(real_traj, gen_traj)
        overall_util = calculate_overall_utility(spatial_util, temporal_util, category_util)
        
        # Store results
        results['real_tid'].append(real_tid)
        results['gen_tid'].append(gen_tid)
        results['label'].append(label)
        results['spatial_utility'].append(spatial_util)
        results['temporal_utility'].append(temporal_util)
        results['day_utility'].append(day_util)
        results['hour_utility'].append(hour_util)
        results['category_utility'].append(category_util)
        results['overall_utility'].append(overall_util)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average utilities
    avg_results = {
        'avg_spatial_utility': np.mean(results['spatial_utility']),
        'avg_temporal_utility': np.mean(results['temporal_utility']),
        'avg_day_utility': np.mean(results['day_utility']),
        'avg_hour_utility': np.mean(results['hour_utility']),
        'avg_category_utility': np.mean(results['category_utility']),
        'avg_overall_utility': np.mean(results['overall_utility'])
    }
    
    # Calculate per-label utilities
    label_results = results_df.groupby('label').mean()
    
    # Print average results
    print("\n===== OVERALL UTILITY METRICS =====")
    print(f"SPATIAL UTILITY:  {avg_results['avg_spatial_utility']:.4f}")
    print(f"TEMPORAL UTILITY: {avg_results['avg_temporal_utility']:.4f}")
    print(f"DAY UTILITY:      {avg_results['avg_day_utility']:.4f}")
    print(f"HOUR UTILITY:     {avg_results['avg_hour_utility']:.4f}")
    print(f"CATEGORY UTILITY: {avg_results['avg_category_utility']:.4f}")
    print(f"OVERALL UTILITY:  {avg_results['avg_overall_utility']:.4f}")
    
    # Print per-label results
    print("\n===== UTILITY METRICS BY LABEL =====")
    print(label_results[['spatial_utility', 'temporal_utility', 'category_utility', 'overall_utility']])
    
    # Save results if output file is provided
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Detailed results saved to {output_file}")
        
        # Save summary statistics
        summary_file = os.path.splitext(output_file)[0] + "_summary.csv"
        label_results.to_csv(summary_file)
        print(f"Summary by label saved to {summary_file}")
    
    return {**avg_results, 'label_results': label_results, 'detailed_results': results_df}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate utility metrics between real and generated trajectories')
    parser.add_argument('real_traj_file', type=str, help='Path to CSV file containing real trajectories')
    parser.add_argument('gen_traj_file', type=str, help='Path to CSV file containing generated trajectories')
    parser.add_argument('--output', '-o', type=str, help='Path to save results CSV', default="utility_results.csv")
    parser.add_argument('--max_pairs', '-m', type=int, help='Maximum number of trajectory pairs to evaluate per label', default=5)
    
    args = parser.parse_args()
    
    evaluate_trajectory_utility(args.real_traj_file, args.gen_traj_file, args.output)