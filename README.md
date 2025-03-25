# RL-Enhanced Transformer-TrajGAN

This repository implements a Reinforcement Learning (RL) enhanced Transformer-based Trajectory GAN for privacy-preserving trajectory generation. The model uses a transformer architecture with PPO optimization to balance privacy and utility in generated trajectories.

## Project Structure

```
.
├── configs/
│   └── config.yaml         # Configuration parameters
├── data/
│   └── raw_data.npy        # Raw trajectory data
├── src/
│   ├── data/
│   │   ├── preprocess.py   # Data preprocessing
│   │   └── dataset.py      # Dataset class
│   ├── models/
│   │   └── transformer.py  # Transformer model implementation
│   ├── training/
│   │   └── trainer.py      # PPO training implementation
│   └── evaluation/
│       ├── privacy.py      # Privacy evaluation metrics
│       └── utility.py      # Utility evaluation metrics
├── results/                # Training results and metrics
└── main.py                # Main training script
```

## Prerequisites

- Python 3.8+
- TensorFlow 2.8+
- NumPy
- Pandas
- Matplotlib
- PyYAML
- scikit-learn
- scipy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/RL-Transformer-TrajGAN.git
cd RL-Transformer-TrajGAN
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Place your raw trajectory data in `data/raw_data.npy`
   - The data should be a numpy array with shape (N, T, F)
   - N: number of trajectories
   - T: maximum sequence length
   - F: number of features (latitude, longitude, day, hour, category)

2. Configure data parameters in `configs/config.yaml`:
```yaml
data:
  max_length: 24
  lat_centroid: 39.9042
  lon_centroid: 116.4074
  scale_factor: 1000
```

## Model Configuration

Adjust model parameters in `configs/config.yaml`:

```yaml
model:
  latent_dim: 64
  num_heads: 8
  num_transformer_blocks: 6
  dff: 256
  dropout_rate: 0.1
```

## Training Configuration

Configure training parameters in `configs/config.yaml`:

```yaml
training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  reward_weights:
    privacy: 0.4
    utility: 0.3
    adversarial: 0.3
  ppo:
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    c1: 1.0
    c2: 0.01
```

## Running the Experiment

1. Start training:
```bash
python main.py
```

2. Monitor training progress:
   - Training metrics will be printed to console
   - Model checkpoints will be saved in `results/checkpoints/`
   - Training history will be saved in `results/training_history.npy`

3. View results:
   - Evaluation metrics will be saved in `results/evaluation_metrics.npy`
   - Generated trajectories will be saved in `results/generated_trajectories.npy`

## Evaluation Metrics

The model is evaluated using:

### Privacy Metrics
- ACC@1: Top-1 accuracy of TUL classifier
- ACC@5: Top-5 accuracy of TUL classifier
- Macro Precision, Recall, and F1

### Utility Metrics
- FID (Fréchet Inception Distance)
- JSD (Jensen-Shannon Divergence) for each feature

## Ablation Study

To run ablation studies with different reward weight combinations:

```python
from src.training.trainer import PPOTrainer
from src.evaluation.privacy import evaluate_privacy
from src.evaluation.utility import evaluate_utility

# Define different weight combinations
weight_combinations = [
    {'privacy': 0.5, 'utility': 0.3, 'adversarial': 0.2},
    {'privacy': 0.3, 'utility': 0.5, 'adversarial': 0.2},
    {'privacy': 0.3, 'utility': 0.3, 'adversarial': 0.4}
]

# Run experiments for each combination
for weights in weight_combinations:
    trainer = PPOTrainer(model, train_dataset, val_dataset, tul_classifier, weights)
    trainer.train(epochs=100)
    
    # Evaluate
    privacy_metrics = evaluate_privacy(model, test_data, tul_classifier)
    utility_metrics = evaluate_utility(model, test_data)
    
    # Save results
    save_results(weights, privacy_metrics, utility_metrics)
```

## Results Analysis

1. Load evaluation metrics:
```python
metrics = np.load('results/evaluation_metrics.npy', allow_pickle=True).item()
```

2. Plot results:
```python
plot_results(metrics)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
