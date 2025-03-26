# Trajectory Generation and Evaluation

This directory contains scripts for generating synthetic trajectories using the trained RL-Enhanced Transformer-TrajGAN model and evaluating their quality and privacy protection.

## Overview

The RL-Enhanced Transformer-TrajGAN model is designed to generate synthetic trajectories that:
1. Look realistic (fooling a discriminator)
2. Preserve the utility of the data (statistical properties)
3. Enhance privacy protection (reducing re-identification risk)

## Scripts

This directory contains the following scripts:

- `generate_trajectories.py`: Generate synthetic trajectories using the trained model
- `evaluate_privacy.py`: Evaluate the privacy protection of synthetic trajectories
- `visualize_trajectories.py`: Visualize real and synthetic trajectories (coming soon)

## Generating Synthetic Trajectories

To generate synthetic trajectories using the trained model:

```bash
python inference/generate_trajectories.py --epoch [EPOCH_NUMBER] --batch_size 256 --output_file inference/synthetic_trajectories.csv --evaluate
```

Arguments:
- `--epoch`: Required. Epoch number of the saved model weights.
- `--batch_size`: Optional. Number of trajectories to generate (default: 256).
- `--output_file`: Optional. Path to save synthetic trajectories (default: 'inference/synthetic_trajectories.csv').
- `--use_test`: Optional flag. Use test data as conditioning input (default: True).
- `--evaluate`: Optional flag. Evaluate the generated trajectories against real data.

The script will:
1. Load the model from the specified epoch
2. Generate synthetic trajectories
3. Save the trajectories to the specified output file
4. If `--evaluate` is specified, evaluate the quality of the generated trajectories

## Evaluating Privacy Protection

To evaluate the privacy protection of the synthetic trajectories:

```bash
python inference/evaluate_privacy.py --real data/test_latlon.csv --synthetic inference/synthetic_trajectories.csv --output inference/privacy_metrics.csv
```

Arguments:
- `--real`: Required. Path to real trajectory data CSV.
- `--synthetic`: Required. Path to synthetic trajectory data CSV.
- `--output`: Optional. Path to save privacy metrics (default: 'inference/privacy_metrics.csv').

The script will:
1. Load the real and synthetic trajectory data
2. Load the TUL (Trajectory-User Linking) classifier
3. Evaluate the privacy protection of the synthetic trajectories
4. Save the privacy metrics to the specified output file

## Privacy Metrics

The privacy evaluation produces the following metrics:

- `real_accuracy`: Accuracy of the TUL classifier on real trajectories
- `synthetic_accuracy`: Accuracy of the TUL classifier on synthetic trajectories
- `privacy_gain`: Reduction in re-identification accuracy (real_accuracy - synthetic_accuracy)
- `normalized_privacy_gain`: Privacy gain normalized by real accuracy (0-1 scale)
- `real_avg_max_prob`: Average maximum probability of the TUL classifier on real trajectories
- `synthetic_avg_max_prob`: Average maximum probability of the TUL classifier on synthetic trajectories
- `real_entropy`: Entropy of the TUL classifier predictions on real trajectories
- `synthetic_entropy`: Entropy of the TUL classifier predictions on synthetic trajectories
- `entropy_increase`: Increase in prediction entropy (synthetic_entropy - real_entropy)

Higher values for `privacy_gain`, `normalized_privacy_gain`, and `entropy_increase` indicate better privacy protection.

## Example Workflow

1. Train the model:
```bash
python train.py --epochs 200 --batch_size 256 --sample_interval 10
```

2. Generate synthetic trajectories:
```bash
python inference/generate_trajectories.py --epoch 200 --batch_size 256 --output_file inference/synthetic_trajectories.csv --evaluate
```

3. Evaluate privacy protection:
```bash
python inference/evaluate_privacy.py --real data/test_latlon.csv --synthetic inference/synthetic_trajectories.csv
```

4. Analyze the results:
- Check the evaluation metrics in 'inference/synthetic_trajectories_metrics.csv'
- Check the privacy metrics in 'inference/privacy_metrics.csv'

## Next Steps

After generating synthetic trajectories, you might want to:

1. Visualize them using visualization tools or libraries like matplotlib, folium, or kepler.gl
2. Use them as input to other models or applications
3. Compare them with other trajectory generation methods
4. Fine-tune the model by adjusting rewards weights or model architecture 