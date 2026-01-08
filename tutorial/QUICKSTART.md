# FlowGRPO Tutorial - Quick Start Guide

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Tutorial

### Step 1: Generate the Dataset

```bash
python -m tutorial.dataset.generate_dataset
```

This creates `tutorial/dataset/train.txt` and `tutorial/dataset/test.txt` with simple prompts.

### Step 2: Run a Quick Example

To see how the model works without training:

```bash
python tutorial/example.py
```

This will:
- Load the dataset
- Create a toy model
- Generate some sample signals
- Compute rewards
- Visualize the results

### Step 3: Train the Model

```bash
python tutorial/train.py --num_epochs 20
```

This will:
- Train the FlowGRPO model
- Generate visualizations every 5 epochs
- Save outputs to `tutorial/outputs/`

### Step 4: View Results

Check the `tutorial/outputs/` directory for:
- `training_curves.png`: Loss and reward curves
- `samples_epoch_X.png`: Generated samples at each evaluation epoch

## Understanding the Code

### Key Components

1. **Toy Flow Model** (`models/toy_flow_model.py`):
   - Simplified 1D flow matching model
   - Generates 1D signals instead of images
   - Easy to visualize and understand

2. **GRPO Algorithm** (`training/grpo.py`):
   - Group Relative Policy Optimization
   - Computes advantages relative to group means
   - Clips importance ratios for stability

3. **Reward Function** (`rewards/simple_reward.py`):
   - Evaluates how well generated signals match prompts
   - Checks both shape (circle/square/triangle) and color

4. **Training Loop** (`training/trainer.py`):
   - Samples trajectories from current policy
   - Computes rewards and advantages
   - Updates model using GRPO loss

## Customization

### Change Model Architecture

Edit `models/toy_flow_model.py`:
- Modify `signal_dim` to change output size
- Adjust `hidden_dim` to change model capacity

### Add New Prompts

Edit `dataset/generate_dataset.py`:
- Add prompts to `train_prompts` or `test_prompts` lists

### Modify Reward Function

Edit `rewards/simple_reward.py`:
- Adjust shape/color reward weights
- Add new evaluation criteria

### Training Hyperparameters

Modify `train.py` or pass arguments:
```bash
python tutorial/train.py \
    --num_epochs 50 \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --clip_range 1e-3
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the repository root:
```bash
cd /path/to/minimal-rl
python tutorial/train.py
```

### CUDA Errors

If you have CUDA available but want to use CPU:
```bash
python tutorial/train.py --device cpu
```

### Memory Issues

Reduce batch size or number of samples per prompt:
```bash
python tutorial/train.py --batch_size 2 --num_samples_per_prompt 2
```

## Next Steps

After completing this tutorial:

1. **Extend to 2D**: Modify the model to generate 2D images
2. **Use Real Models**: Try the full FlowGRPO implementation in `original_impl/`
3. **Experiment**: Try different reward functions and hyperparameters
4. **Read the Paper**: Understand the theoretical foundations

## Resources

- FlowGRPO Paper: https://arxiv.org/abs/2505.05470
- Original Implementation: https://github.com/yifan123/flow_grpo
- Flow Matching: https://arxiv.org/abs/2210.02747
