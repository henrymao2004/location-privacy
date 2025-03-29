import sys
import pandas as pd
import numpy as np

from model import RL_Enhanced_Transformer_TrajGAN

from keras.preprocessing.sequence import pad_sequences

if __name__ == '__main__':
    # Get epoch from command line argument or default to 150
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    latent_dim = 100
    max_length = 144
    
    keys = ['lat_lon', 'day', 'hour', 'category', 'mask']
    vocab_size = {"lat_lon":2, "day":7, "hour":24, "category":10, "mask":1}
    
    # Load training data statistics
    tr = pd.read_csv('data/train_latlon.csv')
    te = pd.read_csv('data/test_latlon.csv')
    
    # Calculate centroids and scale factor
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
    
    # Initialize the GAN model
    gan = RL_Enhanced_Transformer_TrajGAN(
        latent_dim=latent_dim, 
        keys=keys, 
        vocab_size=vocab_size, 
        max_length=max_length, 
        lat_centroid=lat_centroid, 
        lon_centroid=lon_centroid, 
        scale_factor=scale_factor
    )
    
    print(f"Loading model weights from epoch {n_epochs}")
    
    # Load test data
    x_test = np.load('data/final_test.npy', allow_pickle=True)
    
    # Format test data - same as the original code
    x_test = [x_test[0], x_test[1], x_test[2], x_test[3], x_test[4], x_test[5].reshape(-1,1), x_test[6].reshape(-1,1)]
    X_test = [pad_sequences(f, max_length, padding='pre', dtype='float64') for f in x_test[:5]]
    
    # Add random noise for generator
    noise = np.random.normal(0, 1, (len(X_test[0]), latent_dim))
    X_test.append(noise)
    
    # Load model weights
    gan.generator.load_weights(f'results/generator_{n_epochs}.weights.h5')
    
    print("Generating synthetic trajectories...")
    
    # Make predictions with the generator
    prediction = gan.generator.predict(X_test)
    
    # Process the predictions to create trajectory data
    traj_attr_concat_list = []
    for attributes in prediction:
        traj_attr_list = []
        idx = 0
        for row in attributes:
            if row.shape == (max_length, 2):  # For lat_lon
                traj_attr_list.append(row[max_length-x_test[6][idx][0]:])
            else:  # For categorical data (day, hour, category)
                traj_attr_list.append(np.argmax(row[max_length-x_test[6][idx][0]:], axis=1).reshape(x_test[6][idx][0], 1))
            idx += 1
        traj_attr_concat = np.concatenate(traj_attr_list)
        traj_attr_concat_list.append(traj_attr_concat)
    
    traj_data = np.concatenate(traj_attr_concat_list, axis=1)
    
    # Load test metadata
    df_test = pd.read_csv('data/dev_test_encoded_final.csv')
    label = np.array(df_test['label']).reshape(-1, 1)
    tid = np.array(df_test['tid']).reshape(-1, 1)
    
    # Combine data
    traj_data = np.concatenate([label, tid, traj_data], axis=1)
    df_traj_fin = pd.DataFrame(traj_data)
    
    # Set column names
    df_traj_fin.columns = ['label', 'tid', 'lat', 'lon', 'day', 'hour', 'category', 'mask']
    
    # Convert location deviation to actual latitude and longitude
    # In the transformer model, coordinates are scaled during generation and need to be unscaled
    df_traj_fin['lat'] = df_traj_fin['lat'] * gan.scale_factor + gan.lat_centroid
    df_traj_fin['lon'] = df_traj_fin['lon'] * gan.scale_factor + gan.lon_centroid
    
    # Remove mask column
    del df_traj_fin['mask']
    
    # Convert data types
    df_traj_fin['tid'] = df_traj_fin['tid'].astype(np.int32)
    df_traj_fin['day'] = df_traj_fin['day'].astype(np.int32)
    df_traj_fin['hour'] = df_traj_fin['hour'].astype(np.int32)
    df_traj_fin['category'] = df_traj_fin['category'].astype(np.int32)
    df_traj_fin['label'] = df_traj_fin['label'].astype(np.int32)
    
    # Save synthetic trajectory data
    output_file = f'results/syn_traj_test_{n_epochs}.csv'
    df_traj_fin.to_csv(output_file, index=False)
    
    print(f"Saved synthetic trajectories to {output_file}")
    print(f"Generated {len(df_traj_fin)} trajectory points")
    
    # Print summary statistics
    print("\nSummary statistics of generated data:")
    print(f"Latitude range: {df_traj_fin['lat'].min():.6f} to {df_traj_fin['lat'].max():.6f}")
    print(f"Longitude range: {df_traj_fin['lon'].min():.6f} to {df_traj_fin['lon'].max():.6f}")
    print(f"Days distribution: {df_traj_fin['day'].value_counts().sort_index().to_dict()}")
    print(f"Hours distribution: {df_traj_fin['hour'].value_counts().sort_index().head(5).to_dict()} ... (showing first 5)")
    print(f"Category distribution: {df_traj_fin['category'].value_counts().sort_index().head(5).to_dict()} ... (showing first 5)")
