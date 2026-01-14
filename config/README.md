# Hydra Configuration Guide

This directory contains Hydra configuration files for managing hyperparameters in the FlowGRPO tutorial.

## Directory Structure

```
config/
├── config.yaml              # Main config file (composes all sub-configs)
├── model/
│   ├── default.yaml         # Default model architecture
│   └── large.yaml          # Larger model variant
├── training/
│   ├── default.yaml         # Default training hyperparameters
│   ├── fast.yaml           # Fast training (fewer epochs, smaller batches)
│   └── gpu.yaml            # GPU-optimized training
├── dataset/
│   └── default.yaml         # Dataset configuration
└── paths/
    └── default.yaml         # Output paths
```

## Usage Examples

### Basic Usage

```bash
# Use default configuration
python train.py

# Use a different training config
python train.py training=fast
python train.py training=gpu

# Use a different model config
python train.py model=large

# Combine different configs
python train.py training=gpu model=large
```

### Override Specific Parameters

```bash
# Override single parameter
python train.py training.num_epochs=100
python train.py training.batch_size=8
python train.py training.learning_rate=0.0005

# Override multiple parameters
python train.py training.num_epochs=100 training.batch_size=8 training.learning_rate=0.0005

# Override model parameters
python train.py model.signal_dim=128 model.hidden_dim=256

# Override paths
python train.py paths.output_dir=my_outputs
```

### Advanced Usage

```bash
# Use fast training with large model
python train.py training=fast model=large

# Use GPU config but override batch size
python train.py training=gpu training.batch_size=16

# Override multiple nested parameters
python train.py training.num_epochs=200 training.batch_size=16 training.learning_rate=0.0005 model.hidden_dim=256
```

## Configuration Files

### Main Config (`config.yaml`)

Composes all sub-configurations. You can add global settings here like `run_name` and `seed`.

### Model Config (`model/default.yaml`)

- `signal_dim`: Dimension of the generated signal (default: 64)
- `prompt_dim`: Dimension of prompt embeddings (default: 32)
- `hidden_dim`: Hidden dimension of the model (default: 128)
- `vocab_size`: Vocabulary size for prompt encoding (default: 20)

### Training Config (`training/default.yaml`)

- `num_epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size (default: 4)
- `num_samples_per_prompt`: Number of samples per prompt (default: 4)
- `num_steps`: Number of flow steps (default: 20)
- `eval_num_steps`: Number of steps for evaluation (default: 20)
- `learning_rate`: Learning rate (default: 0.001)
- `clip_range`: GRPO clip range (default: 0.0001)
- `beta`: KL penalty coefficient (default: 0.0)
- `eval_freq`: Evaluation frequency in epochs (default: 5)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)
- `device`: Device to use - "cpu", "cuda", or "auto" (default: "auto")

### Dataset Config (`dataset/default.yaml`)

- `dataset_dir`: Directory containing dataset files (default: "dataset")
- `split`: Dataset split to use (default: "train")

### Paths Config (`paths/default.yaml`)

- `output_dir`: Output directory for checkpoints and visualizations (default: "outputs")

## Creating Custom Configs

You can create custom config files by copying existing ones and modifying them:

```bash
# Create a custom training config
cp config/training/default.yaml config/training/my_config.yaml
# Edit my_config.yaml with your settings
python train.py training=my_config
```

## Hydra Outputs

Hydra automatically creates output directories for each run:
- Default: `outputs/YYYY-MM-DD_HH-MM-SS/`
- You can change this with `paths.output_dir` or by setting `hydra.run.dir` in the config

## Tips

1. **Use config groups**: Instead of overriding many parameters, create a new config file in the appropriate group
2. **Check config before running**: Hydra prints the full configuration at the start of training
3. **Reproducibility**: Set `seed` in the main config for reproducible results
4. **Device auto-detection**: Use `device: "auto"` to automatically use CUDA if available
