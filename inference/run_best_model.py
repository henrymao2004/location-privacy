import os
import json
import argparse
import subprocess
import sys

def find_best_model(criterion='combined_score', batch_size=64):
    """
    Find the best model using the find_best_model.py script.
    
    Args:
        criterion: Criterion to use for selecting the best model
        batch_size: Batch size for evaluation
        
    Returns:
        best_epoch: Epoch number of the best model
        best_path: Path to the best model
    """
    # Run the find_best_model.py script
    print("Finding the best model checkpoint...")
    
    cmd = [
        sys.executable, 'inference/find_best_model.py',
        '--criterion', criterion,
        '--batch_size', str(batch_size),
        '--output', 'inference/best_model.json'
    ]
    
    # If criterion is discriminator_loss, we want lower values
    if criterion == 'discriminator_loss':
        cmd.append('--higher_is_better')
    
    try:
        subprocess.run(cmd, check=True)
        
        # Read the results
        with open('inference/best_model.json', 'r') as f:
            results = json.load(f)
        
        return results['best_epoch'], results['best_path']
    
    except Exception as e:
        print(f"Error finding best model: {e}")
        return None, None

def generate_trajectories(epoch, batch_size=256, output_file='inference/best_model_trajectories.csv', evaluate=True):
    """
    Generate synthetic trajectories using the specified model epoch.
    
    Args:
        epoch: Epoch number of the model to use
        batch_size: Number of trajectories to generate
        output_file: Path to save the generated trajectories
        evaluate: Whether to evaluate the generated trajectories
    """
    print(f"Generating trajectories using model from epoch {epoch}...")
    
    cmd = [
        sys.executable, 'inference/generate_trajectories.py',
        '--epoch', str(epoch),
        '--batch_size', str(batch_size),
        '--output_file', output_file
    ]
    
    if evaluate:
        cmd.append('--evaluate')
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Trajectories generated and saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error generating trajectories: {e}")
        return False

def evaluate_privacy(real_file, synthetic_file, output_file='inference/best_model_privacy.csv'):
    """
    Evaluate the privacy protection of the generated trajectories.
    
    Args:
        real_file: Path to the real trajectory data
        synthetic_file: Path to the synthetic trajectory data
        output_file: Path to save the privacy metrics
    """
    print("Evaluating privacy protection...")
    
    cmd = [
        sys.executable, 'inference/evaluate_privacy.py',
        '--real', real_file,
        '--synthetic', synthetic_file,
        '--output', output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Privacy metrics saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error evaluating privacy: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Find the best model and run inference with it')
    parser.add_argument('--criterion', type=str, default='combined_score',
                       choices=['discriminator_loss', 'realism_score', 'privacy_score', 'utility_score', 'combined_score'],
                       help='Criterion to use for selecting the best model')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                       help='Batch size for model evaluation during search')
    parser.add_argument('--gen_batch_size', type=int, default=256,
                       help='Batch size for trajectory generation with the best model')
    parser.add_argument('--output_dir', type=str, default='inference/best_model_results',
                       help='Directory to save the generated trajectories and evaluation metrics')
    parser.add_argument('--real_data', type=str, default='data/test_latlon.csv',
                       help='Path to the real trajectory data for privacy evaluation')
    parser.add_argument('--skip_search', action='store_true',
                       help='Skip the model search and use the specified epoch')
    parser.add_argument('--epoch', type=int, default=None,
                       help='Epoch to use if skipping the search')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Output file paths
    trajectories_file = os.path.join(args.output_dir, 'synthetic_trajectories.csv')
    evaluation_file = os.path.join(args.output_dir, 'trajectory_metrics.csv')
    privacy_file = os.path.join(args.output_dir, 'privacy_metrics.csv')
    
    # Find best model
    best_epoch = args.epoch
    if not args.skip_search or best_epoch is None:
        best_epoch, best_path = find_best_model(args.criterion, args.eval_batch_size)
        
        if best_epoch is None:
            print("Could not find a valid model. Exiting.")
            return
        
        print(f"Best model found: Epoch {best_epoch}")
        
        # Save the best epoch to a file
        with open(os.path.join(args.output_dir, 'best_epoch.txt'), 'w') as f:
            f.write(str(best_epoch))
    else:
        print(f"Using specified epoch: {best_epoch}")
    
    # Generate trajectories
    success = generate_trajectories(
        best_epoch, 
        batch_size=args.gen_batch_size, 
        output_file=trajectories_file,
        evaluate=True
    )
    
    if not success:
        print("Failed to generate trajectories. Exiting.")
        return
    
    # Evaluate privacy
    evaluate_privacy(
        args.real_data,
        trajectories_file,
        output_file=privacy_file
    )
    
    print("\nAll done!")
    print(f"Best model epoch: {best_epoch}")
    print(f"Synthetic trajectories: {trajectories_file}")
    print(f"Privacy metrics: {privacy_file}")

if __name__ == '__main__':
    main() 