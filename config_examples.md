# Hydra Configuration Examples

This document provides practical examples of using Hydra to configure the FlowGRPO training.

## Basic Usage

### Default Configuration
```bash
python train.py
```
Uses all default settings from `config/config.yaml`.

### Using Predefined Configs

**Fast Training** (for quick testing):
```bash
python train.py training=fast
```
- Fewer epochs (10)
- Smaller batch size (2)
- Fewer samples per prompt (2)

**GPU-Optimized Training**:
```bash
python train.py training=gpu
```
- Larger batch size (8)
- More samples per prompt (8)
- Uses CUDA device

**Large Model**:
```bash
python train.py model=large
```
- Larger signal dimension (128)
- Larger hidden dimension (256)
- More vocabulary (30)

**Combining Configs**:
```bash
python train.py training=gpu model=large
```

## Parameter Overrides

### Training Parameters

**Change number of epochs**:
```bash
python train.py training.num_epochs=100
```

**Change batch size**:
```bash
python train.py training.batch_size=8
```

**Change learning rate**:
```bash
python train.py training.learning_rate=0.0005
```

**Change multiple training parameters**:
```bash
python train.py training.num_epochs=100 training.batch_size=8 training.learning_rate=0.0005
```

**Change flow steps**:
```bash
python train.py training.num_steps=30 training.eval_num_steps=30
```

**Change GRPO parameters**:
```bash
python train.py training.clip_range=0.0005 training.beta=0.01
```

### Model Parameters

**Change model architecture**:
```bash
python train.py model.signal_dim=128 model.hidden_dim=256
```

**Change prompt encoding**:
```bash
python train.py model.prompt_dim=64 model.vocab_size=30
```

### Paths and Dataset

**Change output directory**:
```bash
python train.py paths.output_dir=my_experiment_outputs
```

**Change dataset directory**:
```bash
python train.py dataset.dataset_dir=my_dataset
```

## Advanced Examples

### Long Training Run
```bash
python train.py training.num_epochs=200 training.eval_freq=10
```

### High-Throughput Training
```bash
python train.py training.batch_size=16 training.num_samples_per_prompt=16
```

### Fine-Tuning Existing Model
```bash
python train.py training.num_epochs=50 training.learning_rate=0.0001 training.beta=0.01
```

### Experiment with Different Model Sizes
```bash
# Small model
python train.py model.signal_dim=32 model.hidden_dim=64

# Medium model (default)
python train.py

# Large model
python train.py model=large
```

### CPU Training with Custom Settings
```bash
python train.py training.device=cpu training.batch_size=2 training.num_samples_per_prompt=2
```

## Creating Custom Configs

You can create your own config files:

1. **Create a custom training config**:
   ```bash
   cp config/training/default.yaml config/training/my_experiment.yaml
   ```
   Edit `config/training/my_experiment.yaml` with your settings.

2. **Use your custom config**:
   ```bash
   python train.py training=my_experiment
   ```

## Viewing Configuration

Hydra automatically prints the full configuration at the start of training. You can also use:

```bash
python train.py --cfg job  # Print config and exit
```

## Hydra Output Directories

By default, Hydra creates output directories with timestamps:
- `outputs/YYYY-MM-DD_HH-MM-SS/`

To use a custom output directory:
```bash
python train.py paths.output_dir=my_outputs
```

Or disable Hydra's output directory management:
```bash
python train.py hydra.output_subdir=null hydra.run.dir=.
```

## Tips

1. **Start with defaults**: Always start with `python train.py` to ensure everything works
2. **Use config groups**: Create config files for common experiment setups
3. **Override selectively**: Only override parameters you need to change
4. **Check config output**: The printed configuration shows exactly what will be used
5. **Reproducibility**: Set `seed` in config for reproducible results
