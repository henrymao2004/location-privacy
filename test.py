import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import json

# Updated model import
from model import KAN_TrajGAN 

# Removed MARC logger and related imports
# from MARC.core.logger import Logger
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# from MARC.core.utils.geohash import bin_geohash

# Placeholder for evaluation metrics - replace with actual GAN evaluation
def evaluate_generated_data(real_data_list, generated_data_list, keys):
    """Placeholder function to compare real and generated data statistics."""
    print("\n--- Basic Statistical Comparison ---   ")
    if len(real_data_list) != len(generated_data_list):
        print("Error: Real and generated data lists have different lengths.")
        return

    for i, key in enumerate(keys):
        real_feature = real_data_list[i]
        gen_feature = generated_data_list[i]
        
        print(f"Feature: '{key}'")
        try:
            if key == 'lat_lon': # Continuous data
                 real_mean = np.mean(real_feature, axis=(0, 1))
                 gen_mean = np.mean(gen_feature, axis=(0, 1))
                 real_std = np.std(real_feature, axis=(0, 1))
                 gen_std = np.std(gen_feature, axis=(0, 1))
                 print(f"  Real Mean: {real_mean}, Gen Mean: {gen_mean}")
                 print(f"  Real Std:  {real_std}, Gen Std:  {gen_std}")
                 # Add more sophisticated distance metrics like MMD or Wasserstein if needed
            else: # Categorical data (assuming one-hot)
                 # Compare distributions (e.g., frequency of categories)
                 real_counts = np.sum(np.sum(real_feature, axis=0), axis=0)
                 gen_counts = np.sum(np.sum(gen_feature, axis=0), axis=0)
                 real_dist = real_counts / np.sum(real_counts)
                 gen_dist = gen_counts / np.sum(gen_counts)
                 print(f"  Real Dist: {np.round(real_dist, 3)}")
                 print(f"  Gen Dist:  {np.round(gen_dist, 3)}")
                 # Calculate JS Divergence or Chi-Squared test for distribution similarity
                 try:
                     from scipy.spatial.distance import jensenshannon
                     jsd = jensenshannon(real_dist, gen_dist, base=2)
                     print(f"  Jensen-Shannon Divergence: {jsd:.4f}")
                 except ImportError:
                     print("  (Install scipy for Jensen-Shannon Divergence calculation)")
                 except ValueError as e:
                      print(f"  Could not calculate JSD: {e}") # e.g. if distributions have different lengths

        except Exception as e:
            print(f"  Error evaluating feature {key}: {e}")

# Removed compute_utility_metrics as it was specific to the previous model/task
# def compute_utility_metrics(...):
#    ...

# Removed recreate_model_structure as model is loaded directly
# def recreate_model_structure():
#    ...
    
# Removed get_trajectories as data loading is simplified
# def get_trajectories(...):
#    ...

def load_and_pad_data(file_path, keys, max_length):
    """Loads data from npz file and pads sequences."""
    try:
        data_dict = np.load(file_path, allow_pickle=True)
        print(f"Loaded data keys from {file_path}: {list(data_dict.keys())}")

        missing_keys = [k for k in keys if k not in data_dict]
        if missing_keys:
            raise ValueError(f"Missing keys in data file {file_path}: {missing_keys}")

        data_list = [data_dict[key] for key in keys]
        data_dict.close() # Close the file

        # Padding
        padded_data = []
        num_samples = -1
        for i, key in enumerate(keys):
            feature_data = data_list[i]
            # Pad sequences individually using float32
            padded_feature = pad_sequences(feature_data, max_length, padding='pre', dtype='float32', truncating='pre')
            
            if num_samples == -1: num_samples = padded_feature.shape[0]
            elif num_samples != padded_feature.shape[0]:
                raise ValueError(f"Inconsistent sample count after padding for key '{key}'")
                
            padded_data.append(padded_feature)
            
        print(f"Successfully loaded and padded data from {file_path}. Samples: {num_samples}")
        return padded_data, num_samples

    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return None, 0
    except Exception as e:
        print(f"Error loading or padding data from {file_path}: {e}")
        return None, 0

def main(config_file='results/model_config_best.json', 
         weights_epoch='best', 
         num_generate=1000,
         output_file='results/generated_trajectories.npz'):

    print('====================================', 'TESTING KAN-TrajGAN', 
          '====================================')
    print(f"Using config: {config_file}")
    print(f"Using weights epoch: {weights_epoch}")
    print(f"Generating {num_generate} samples.")

    # Load model configuration
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Model config file not found at {config_file}")
        return
    except Exception as e:
        print(f"Error loading model config: {e}")
        return
        
    # Load the model using the configuration and specified epoch weights
    try:
        model = KAN_TrajGAN.from_saved_checkpoint(
            epoch=weights_epoch, 
            checkpoint_dir='results' # Assuming checkpoints are in 'results'
        )
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    # --- Load Real Test Data (for comparison) --- 
    # Define keys needed based on model config or structure
    # These should match the keys used during training and expected by the model
    data_keys = model.discriminator_output_keys # Use keys defined in the loaded model instance
    max_length = model.max_length
    
    # Assuming a preprocessed test file exists
    test_data_path = 'data/final_test.npz' 
    X_test_padded, num_test_samples = load_and_pad_data(test_data_path, data_keys, max_length)

    if X_test_padded is None:
        print("Could not load test data for comparison. Skipping evaluation.")
        # Optionally, continue with generation only
        # return 

    # --- Generate Synthetic Data --- 
    print(f"\nGenerating {num_generate} synthetic trajectories...")
    try:
        # Sample from standard normal distribution for KAN model
        z = tf.random.normal([num_generate, model.latent_dim])
        
        # Generate trajectories
        generated_trajs_list = model.generator.predict(z)
        
        # Ensure output is a list
        if not isinstance(generated_trajs_list, list):
            generated_trajs_list = [generated_trajs_list]
            
        print(f"Generated {len(generated_trajs_list[0])} trajectories.")

        # --- Save Generated Data --- 
        # Save as a dictionary in npz format, matching the keys
        gen_data_dict = {key: data for key, data in zip(data_keys, generated_trajs_list)}
        np.savez(output_file, **gen_data_dict)
        print(f"Generated trajectories saved to {output_file}")

        # --- Evaluate Generated Data (Basic Comparison) --- 
        if X_test_padded is not None:
            # Optionally select a subset of real data for comparison if sizes differ greatly
            if num_generate < num_test_samples:
                 print(f"Comparing generated data ({num_generate} samples) with a subset of real test data ({num_generate} samples)")
                 indices = np.random.choice(num_test_samples, num_generate, replace=False)
                 real_data_subset = [data[indices] for data in X_test_padded]
                 evaluate_generated_data(real_data_subset, generated_trajs_list, data_keys)
            else:
                 evaluate_generated_data(X_test_padded, generated_trajs_list, data_keys)
        else:
             print("Skipping evaluation as real test data was not loaded.")

    except Exception as e:
        print(f"An error occurred during generation or evaluation: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting script finished.")

if __name__ == '__main__':
    # Simple argument parsing (optional)
    import argparse
    parser = argparse.ArgumentParser(description="Test KAN-TrajGAN Model")
    parser.add_argument("--config", type=str, default="results/model_config_best.json", help="Path to model config JSON file")
    parser.add_argument("--epoch", type=str, default="best", help="Epoch number or 'best' to load weights from")
    parser.add_argument("--num_generate", type=int, default=1000, help="Number of trajectories to generate")
    parser.add_argument("--output", type=str, default="results/generated_trajectories.npz", help="Output file for generated data (.npz)")
    args = parser.parse_args()

    main(config_file=args.config, 
         weights_epoch=args.epoch, 
         num_generate=args.num_generate,
         output_file=args.output)