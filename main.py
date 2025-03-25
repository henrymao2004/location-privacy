import os
import yaml
import tensorflow as tf
import numpy as np
from src.data.preprocess import preprocess_data
from src.data.dataset import TrajectoryDataset
from src.models.transformer import Transformer_TrajGAN
from src.training.trainer import PPOTrainer
from src.evaluation.privacy import evaluate_privacy
from src.evaluation.utility import evaluate_utility

def load_config(config_path='configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Load configuration
    config = load_config()
    
    # Create necessary directories
    for path in config['paths'].values():
        os.makedirs(path, exist_ok=True)
    
    # Preprocess data
    print("Preprocessing data...")
    train_data, val_data, test_data = preprocess_data(
        data_path=config['paths']['raw_data'],
        max_length=config['data']['max_length'],
        lat_centroid=config['data']['lat_centroid'],
        lon_centroid=config['data']['lon_centroid'],
        scale_factor=config['data']['scale_factor']
    )
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_data, batch_size=config['training']['batch_size'])
    val_dataset = TrajectoryDataset(val_data, batch_size=config['training']['batch_size'])
    
    # Initialize model
    print("Initializing model...")
    model = Transformer_TrajGAN(
        latent_dim=config['model']['latent_dim'],
        keys=config['model']['keys'],
        vocab_size=config['model']['vocab_size'],
        max_length=config['data']['max_length'],
        lat_centroid=config['data']['lat_centroid'],
        lon_centroid=config['data']['lon_centroid'],
        scale_factor=config['data']['scale_factor'],
        w1=config['training']['reward_weights']['w1'],
        w2=config['training']['reward_weights']['w2'],
        w3=config['training']['reward_weights']['w3']
    )
    
    # Load TUL classifier
    tul_classifier = tf.keras.models.load_model(config['paths']['tul_classifier'])
    
    # Initialize trainer
    trainer = PPOTrainer(
        model=model,
        tul_classifier=tul_classifier,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config['training']
    )
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        epochs=config['training']['epochs'],
        save_dir=config['paths']['checkpoints']
    )
    
    # Evaluate model
    print("Evaluating model...")
    privacy_metrics = evaluate_privacy(model, test_data, tul_classifier)
    utility_metrics = evaluate_utility(model, test_data)
    
    # Save results
    results = {
        'privacy_metrics': privacy_metrics,
        'utility_metrics': utility_metrics,
        'training_history': history
    }
    np.save(os.path.join(config['paths']['results'], 'evaluation_metrics.npy'), results)
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 