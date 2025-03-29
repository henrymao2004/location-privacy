# PPO-Enhanced Trajectory GAN

This repository implements a Privacy-Preserving Trajectory Generation model using Reinforcement Learning, specifically Proximal Policy Optimization (PPO). The model extends the traditional LSTM-TrajGAN architecture by adding a policy optimization component that fine-tunes the generator using a reward signal that balances privacy, utility, and realism.

## Model Architecture

The framework consists of these key components:

1. **Transformer-based Generator**: Acts as the policy network in RL, generating synthetic trajectories
2. **Transformer-based Discriminator**: Provides adversarial feedback on trajectory realism
3. **Critic Network**: Assists in policy learning by estimating state-value functions
4. **Reward Function**: Balances privacy (via TUL), utility, and realism objectives

## RL Formulation

Trajectory generation is formulated as a sequential decision process where:

- **State**: The Transformer's internal state after generating t points
- **Action**: The next trajectory point (coordinates, timestamp, POI category)
- **Reward**: Computed using TUL (Trajectory-User Linking) privacy score, spatial-temporal utility, and realism metrics

## Reward Function

The reward function is a weighted combination of three components:

```
R(T_i, T_i^{orig}) = α · R_{priv}(T_i) + β · R_{util}(T_i, T_i^{orig}) + γ · R_{adv}(T_i)
```

where:
- `R_{priv}`: Privacy reward measured using TUL classifier
- `R_{util}`: Utility reward measuring spatial, temporal, and semantic similarity
- `R_{adv}`: Adversarial reward from discriminator feedback
- `α`, `β`, and `γ` are hyperparameters balancing the objectives

## Installation

1. Clone the repository
```bash
git clone [repository-url]
cd location-privacy
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare the data
```bash
# Make sure the trajectory data is in the data/ directory
# The data should be pre-processed into the expected format
```

## Usage

### Training the Model

To train the model with default parameters:

```bash
python ppo_demo.py --train --epochs 100 --batch_size 32
```

Advanced training options:
```bash
python ppo_demo.py --train --epochs 200 --batch_size 64 --latent_dim 128
```

### Generating Privacy-Enhanced Trajectories

To generate trajectories using a trained model:

```bash
# Generate using the best checkpoint
python ppo_demo.py --generate --load_checkpoint best --num_samples 10

# Generate using a specific checkpoint
python ppo_demo.py --generate --load_checkpoint 50 --num_samples 5
```

### Model Evaluation

To evaluate the model's privacy-utility tradeoff:

```bash
# Run the evaluation script (separate implementation)
python evaluate.py --checkpoint best
```

## Hyperparameter Configuration

Key hyperparameters that can be tuned:

1. **PPO parameters**:
   - `clip_epsilon`: Clipping parameter for PPO (default: 0.2)
   - `entropy_beta`: Entropy bonus coefficient (default: 0.01)
   
2. **Reward weights**:
   - `alpha`: Weight for privacy reward (default: 0.4)
   - `beta`: Weight for utility reward (default: 0.4)
   - `gamma`: Weight for adversarial reward (default: 0.2)

3. **Model architecture**:
   - `d_model`: Dimension of transformer model (default: 128)
   - `num_heads`: Number of attention heads (default: 4)
   - `dff`: Dimension of feed-forward network (default: 512)

## Results

The PPO-enhanced model demonstrates significant improvements over the baseline LSTM-TrajGAN:

1. **Better Privacy Protection**: Lower re-identification rates in TUL tasks
2. **Improved Utility Preservation**: Better maintenance of essential trajectory patterns
3. **More Realistic Trajectories**: Increased realism in generated trajectories
4. **Flexible Privacy-Utility Tradeoff**: Dynamic adjustment of privacy-utility balance via reward weights

## Model Components

- `models/ppo_trajgan.py`: Main implementation of the PPO-enhanced TrajGAN
- `models/transformer_components.py`: Transformer architecture components
- `models/critic.py`: Critic network for value function estimation
- `models/reward_function.py`: Implementation of the multi-component reward function
- `models/tul_classifier.py`: Trajectory-User Linking classifier for privacy evaluation

## Citation

If you use this code in your research, please cite our work:

```
@article{...}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 