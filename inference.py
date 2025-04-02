import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from model import KAN_TrajGAN
from keras.preprocessing.sequence import pad_sequences
import os

if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_weights_path> <output_csv_path>")
        print("Example: python inference.py results/generator_50.weights.h5 results/syn_traj_test.csv")
        sys.exit(1)
    
    # Get command line arguments
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Model parameters
    latent_dim = 100
    max_length = 144
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon": 2, "day": 7, "hour": 24, "category": 10, "mask": 1}
    
    # Load centroid and scale data
    print("Loading geographic data...")
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
    lat_centroid = (tr['lat'].sum() + te['lat'].sum())/(len(tr)+len(te))
    lon_centroid = (tr['lon'].sum() + te['lon'].sum())/(len(tr)+len(te))
    
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
    
    print(f"Geographic stats - Lat centroid: {lat_centroid}, Lon centroid: {lon_centroid}, Scale factor: {scale_factor}")
    
    # Initialize the model
    print("Initializing KAN_TrajGAN model...")
    model = KAN_TrajGAN(
        latent_dim=latent_dim,
        keys=keys,
        vocab_size=vocab_size,
        max_length=max_length,
        lat_centroid=lat_centroid,
        lon_centroid=lon_centroid,
        scale_factor=scale_factor
    )
    
    # Load test data
    print("Loading test data...")
    x_test = np.load('data/final_test.npy', allow_pickle=True)
    
    # Verify the data structure
    print(f"Test data structure: {len(x_test)} elements")
    for i, x in enumerate(x_test):
        if isinstance(x, np.ndarray):
            print(f"  Element {i}: shape {x.shape}, dtype {x.dtype}")
        else:
            print(f"  Element {i}: type {type(x)}")
    
    # Extract trajectory lengths for post-processing
    traj_lengths = x_test[6] if len(x_test) > 6 else None
    
    # Get batch size from test data
    batch_size = len(x_test[0]) if isinstance(x_test[0], list) else x_test[0].shape[0]
    print(f"Generating synthetic trajectories for {batch_size} samples...")
    
    # Load the generator weights
    print(f"Loading generator weights from {model_path}...")
    try:
        model.generator.load_weights(model_path)
        print("Generator weights loaded successfully!")
    except Exception as e:
        print(f"Error loading generator weights: {e}")
        sys.exit(1)
    
    # Generate synthetic trajectories using KAN model
    print("Generating predictions with KAN model...")
    try:
        # Sample from standard normal distribution
        z = tf.random.normal([batch_size, latent_dim])
        
        # Generate trajectories
        predictions = model.generator.predict(z)
        
        print(f"Generated {len(predictions)} prediction arrays")
        for i, pred in enumerate(predictions):
            print(f"  Prediction {i}: shape {pred.shape}, dtype {pred.dtype}")
    except Exception as e:
        print(f"Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Process the predicted trajectories
    print("Processing predicted trajectories...")
    traj_attr_concat_list = []
    for attributes in predictions:
        traj_attr_list = []
        idx = 0
        for row in attributes:
            if idx < len(traj_lengths):  # Make sure we don't go out of bounds
                # Get trajectory length (handle both int and array formats)
                if isinstance(traj_lengths[idx], (list, tuple, np.ndarray)):
                    traj_len = traj_lengths[idx][0]
                else:
                    traj_len = traj_lengths[idx]  # Assume it's a simple integer
                
                if row.shape == (max_length, 2):
                    traj_attr_list.append(row[max_length-traj_len:])
                else:
                    traj_attr_list.append(np.argmax(row[max_length-traj_len:], axis=1).reshape(traj_len, 1))
                idx += 1
            else:
                break
        
        # Skip if we couldn't process any rows
        if len(traj_attr_list) == 0:
            continue
            
        try:
            traj_attr_concat = np.concatenate(traj_attr_list)
            traj_attr_concat_list.append(traj_attr_concat)
        except Exception as e:
            print(f"Error concatenating attributes: {e}")
            print(f"Shapes of traj_attr_list: {[a.shape for a in traj_attr_list]}")
            continue
    
    # Skip if no trajectories were processed successfully
    if len(traj_attr_concat_list) == 0:
        print("No trajectories were processed successfully. Check the data format.")
        sys.exit(1)
        
    # Combine all trajectory attributes
    try:
        traj_data = np.concatenate(traj_attr_concat_list, axis=1)
        print(f"Combined trajectory data shape: {traj_data.shape}")
    except Exception as e:
        print(f"Error concatenating trajectory data: {e}")
        print(f"Shapes of traj_attr_concat_list: {[a.shape for a in traj_attr_concat_list]}")
        sys.exit(1)
    
    # Load test metadata
    print("Loading metadata...")
    try:
        df_test = pd.read_csv('data/dev_test_encoded_final.csv')
        print(f"Loaded metadata with {len(df_test)} rows")
        label = np.array(df_test['label']).reshape(-1, 1)
        tid = np.array(df_test['tid']).reshape(-1, 1)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        sys.exit(1)
    
    # Combine with metadata
    print("Combining data with metadata...")
    if len(label) != len(traj_data):
        print(f"Warning: Mismatch between label length ({len(label)}) and trajectory data length ({len(traj_data)})")
        # Truncate to the shorter length
        min_len = min(len(label), len(traj_data))
        label = label[:min_len]
        tid = tid[:min_len]
        traj_data = traj_data[:min_len]
    
    # Concatenate with labels and IDs
    try:
        traj_data = np.concatenate([label, tid, traj_data], axis=1)
        print(f"Final data shape after adding metadata: {traj_data.shape}")
    except Exception as e:
        print(f"Error adding metadata: {e}")
        sys.exit(1)
    
    # Create DataFrame
    print("Creating DataFrame...")
    column_names = ['label', 'tid', 'lat', 'lon', 'day', 'hour', 'category', 'mask']
    if traj_data.shape[1] != len(column_names):
        print(f"Warning: Column count mismatch. Expected {len(column_names)}, got {traj_data.shape[1]}")
        # Adjust column names if necessary
        if traj_data.shape[1] < len(column_names):
            column_names = column_names[:traj_data.shape[1]]
        else:
            # Add extra columns with generic names
            for i in range(len(column_names), traj_data.shape[1]):
                column_names.append(f"extra_{i}")
    
    df_traj_fin = pd.DataFrame(traj_data, columns=column_names)
    
    # Convert location deviations to actual latitude and longitude
    print("Converting coordinate deviations to actual coordinates...")
    if 'lat' in df_traj_fin.columns and 'lon' in df_traj_fin.columns:
        df_traj_fin['lat'] = df_traj_fin['lat'].astype(float) + lat_centroid
        df_traj_fin['lon'] = df_traj_fin['lon'].astype(float) + lon_centroid
    
    # Remove mask column if present
    if 'mask' in df_traj_fin.columns:
        df_traj_fin = df_traj_fin.drop(columns=['mask'])
    
    # Convert columns to appropriate types
    print("Converting column types...")
    try:
        df_traj_fin['tid'] = df_traj_fin['tid'].astype(np.int32)
        df_traj_fin['day'] = df_traj_fin['day'].astype(np.int32)
        df_traj_fin['hour'] = df_traj_fin['hour'].astype(np.int32)
        df_traj_fin['category'] = df_traj_fin['category'].astype(np.int32)
        df_traj_fin['label'] = df_traj_fin['label'].astype(np.int32)
    except KeyError as e:
        print(f"Warning: Could not convert column {e} to int32. Skipping.")
    except ValueError as e:
        print(f"Warning: Value error while converting columns: {e}")
    
    # Print sample data
    print("\nSample of synthetic trajectory data:")
    print(df_traj_fin.head())
    
    # Save to CSV
    print(f"Saving synthetic trajectories to {output_path}...")
    df_traj_fin.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df_traj_fin)} synthetic trajectories!")