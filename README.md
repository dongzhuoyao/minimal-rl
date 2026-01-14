# FlowGRPO Tutorial

This tutorial provides a minimal implementation of FlowGRPO (Flow Matching with Group Relative Policy Optimization) for learning the basics of training flow matching models with reinforcement learning.

## Overview

FlowGRPO is an online RL method for training flow matching models (like diffusion models). This tutorial includes:

- **Toy Dataset**: Simple prompts for image generation tasks
- **Toy Model**: A simplified 1D flow matching model (easy to visualize and understand)
- **GRPO Training**: Implementation of the Group Relative Policy Optimization algorithm
- **Visualization**: Real-time plots of training progress, rewards, and generated samples
- **Evaluation**: Metrics to track model performance

## Structure

```
.
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── dataset/              # Toy dataset
│   ├── train.txt        # Training prompts
│   └── test.txt         # Test prompts
├── models/              # Model implementations
│   └── toy_flow_model.py  # Simplified flow matching model
├── rewards/             # Reward functions
│   └── simple_reward.py   # Simple reward function
├── training/            # Training utilities
│   ├── grpo.py          # GRPO algorithm implementation
│   └── trainer.py       # Main training loop
├── visualization/       # Visualization utilities
│   └── plotter.py       # Plotting functions
├── evaluation/          # Evaluation scripts
│   └── evaluator.py     # Evaluation metrics
└── train.py             # Main training script
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate toy dataset:
```bash
python -m dataset.generate_dataset
```

3. Run training:
```bash
python train.py
```

4. View results:
The training will generate visualizations in `outputs/` directory.

## Key Concepts

### Flow Matching
Flow matching models learn to transform noise into data by following a probability flow. Unlike diffusion models that use discrete steps, flow matching uses continuous flows.

### GRPO (Group Relative Policy Optimization)
GRPO is similar to PPO but uses group-based advantages:
- Samples are grouped by prompt
- Advantages are computed relative to the group mean
- This reduces variance and improves training stability

### Training Loop
1. **Sample**: Generate trajectories (images) from the current policy
2. **Reward**: Compute rewards for each sample
3. **Advantage**: Compute group-relative advantages
4. **Update**: Update policy using clipped policy gradient

## Tutorial Components

### 1. Toy Dataset
Simple prompts like "a red circle", "a blue square" that are easy to evaluate.

### 2. Toy Model
A 1D flow matching model that generates simple shapes. This makes it easy to:
- Visualize the generation process
- Understand how flow matching works
- Debug training issues

### 3. Simple Reward
A reward function that measures how well the generated shape matches the prompt.

### 4. Visualization
Real-time plots showing:
- Training loss
- Reward curves
- Generated samples
- Advantage distributions

### 5. Evaluation
Metrics including:
- Average reward
- Sample quality
- Training stability

## Next Steps

After understanding this tutorial, you can:
1. Extend to 2D image generation
2. Use more complex reward functions
3. Experiment with different hyperparameters
4. Try the full FlowGRPO implementation in `original_impl/`
