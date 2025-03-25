"""Convert encoded csv files to one-hot-encoded npy files."""

import pandas as pd
import numpy as np
import argparse

def data_conversion(df, tid_col, max_length=144):
    """Converts input panda dataframe to one-hot-encoded Numpy array (locations are still in float).
    
    Args:
        df: Input dataframe
        tid_col: Column name for trajectory ID
        max_length: Maximum length to pad/truncate trajectories to
    """
    
    x = [[] for i in ['lat_lon', 'day', 'hour', 'category', 'mask']]
    for tid in df[tid_col].unique():
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, ['lat', 'lon', 'day', 'hour', 'category']].values)
        
        # Get sequence length
        seq_len = min(len(traj), max_length)
        
        # Create and pad/truncate lat_lon
        loc_list = []
        for i in range(seq_len):
            lat = traj['lat'].values[i]
            lon = traj['lon'].values[i]
            loc_list.append(np.array([lat, lon], dtype=np.float64))
        loc_array = np.array(loc_list)
        if seq_len < max_length:
            padding = np.zeros((max_length - seq_len, 2))
            loc_array = np.vstack([loc_array, padding])
        else:
            loc_array = loc_array[:max_length]
        x[0].append(loc_array)
        
        # Create and pad/truncate one-hot encodings
        day_onehot = np.eye(7)[features[2].astype(np.int32)][:seq_len]
        hour_onehot = np.eye(24)[features[3].astype(np.int32)][:seq_len]
        category_onehot = np.eye(10)[features[4].astype(np.int32)][:seq_len]
        
        # Pad if necessary
        if seq_len < max_length:
            day_padding = np.zeros((max_length - seq_len, 7))
            hour_padding = np.zeros((max_length - seq_len, 24))
            category_padding = np.zeros((max_length - seq_len, 10))
            
            day_onehot = np.vstack([day_onehot, day_padding])
            hour_onehot = np.vstack([hour_onehot, hour_padding])
            category_onehot = np.vstack([category_onehot, category_padding])
        
        x[1].append(day_onehot)
        x[2].append(hour_onehot)
        x[3].append(category_onehot)
        
        # Create and pad/truncate mask
        mask = np.ones((seq_len, 1))
        if seq_len < max_length:
            mask_padding = np.zeros((max_length - seq_len, 1))
            mask = np.vstack([mask, mask_padding])
        else:
            mask = mask[:max_length]
        x[4].append(mask)
    
    # Convert lists to numpy arrays
    converted_data = []
    for feature_list in x:
        converted_data.append(np.stack(feature_list))
    
    return np.array(converted_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", type=str, default="dev_train_encoded_final.csv")
    parser.add_argument("--save_path", type=str, default="train_encoded.npy")
    parser.add_argument("--tid_col", type=str, default="tid")
    parser.add_argument("--max_length", type=int, default=144)
    args = parser.parse_args()
    
    df = pd.read_csv(args.load_path)
    converted_data = data_conversion(df, args.tid_col, args.max_length)
    np.save(args.save_path, converted_data)