# Running FlowGRPO Tutorial in Google Colab

This guide shows you how to run the FlowGRPO trainer in Google Colab.

## Quick Start (3 Methods)

### Method 1: Using the Notebook (Recommended)

1. **Upload the notebook** `colab_setup.ipynb` to Colab
2. **Run cells sequentially** - Each cell will:
   - Install dependencies
   - Setup directories
   - Generate dataset
   - Run training

### Method 2: Copy-Paste Script

1. **Upload all code files** to Colab (or clone from GitHub)
2. **Run this in a Colab cell**:

```python
# Install dependencies
!pip install torch>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0 pillow>=10.0.0 scipy>=1.10.0

# Run training
exec(open('colab_train.py').read())
```

### Method 3: Step-by-Step

#### Step 1: Install Dependencies

```python
!pip install torch>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 tqdm>=4.65.0 pillow>=10.0.0 scipy>=1.10.0
```

#### Step 2: Upload Code Files

**Option A: From GitHub** (if you have a repo):
```python
!git clone https://github.com/your-username/minimal-rl.git
%cd minimal-rl
```

**Option B: Manual Upload**:
- Use Colab's file menu: `File > Upload > Upload to session storage`
- Upload the code files to the root directory

#### Step 3: Setup and Run

```python
import sys
from pathlib import Path
import torch

# Add to path
sys.path.insert(0, str(Path.cwd()))

# Import
from dataset.dataset import PromptDataset
from dataset.generate_dataset import generate_dataset
from models.toy_flow_model import create_toy_model
from rewards.simple_reward import SimpleReward
from training.trainer import FlowGRPOTrainer

# Setup directories
import os
os.makedirs("dataset", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Generate dataset
generate_dataset()

# Load datasets
train_dataset = PromptDataset("dataset", split="train")
test_dataset = PromptDataset("dataset", split="test")

# Create model
model, prompt_encoder = create_toy_model(
    signal_dim=64,
    prompt_dim=32,
    hidden_dim=128,
    vocab_size=20,
)

# Create reward function
reward_fn = SimpleReward()

# Training config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

config = {
    "batch_size": 4,
    "num_samples_per_prompt": 4,
    "num_steps": 20,
    "eval_num_steps": 20,
    "learning_rate": 1e-3,
    "clip_range": 1e-4,
    "beta": 0.0,
    "device": device,
    "output_dir": "outputs",
    "eval_freq": 5,
    "max_grad_norm": 1.0,
}

# Create trainer
trainer = FlowGRPOTrainer(
    model=model,
    prompt_encoder=prompt_encoder,
    reward_fn=reward_fn,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    config=config,
)

# Train
trainer.train(num_epochs=50)
```

#### Step 4: View Results

```python
from IPython.display import Image, display
from pathlib import Path

# Display plots
output_dir = Path("outputs")
if (output_dir / "training_curves.png").exists():
    display(Image(str(output_dir / "training_curves.png")))

# List files
for f in sorted(output_dir.glob("*")):
    print(f"  - {f.name}")
```

## Directory Structure Needed

Make sure you have this structure in Colab:

```
.
├── __init__.py
├── dataset/
│   ├── __init__.py
│   ├── dataset.py
│   └── generate_dataset.py
├── models/
│   ├── __init__.py
│   └── toy_flow_model.py
├── rewards/
│   ├── __init__.py
│   └── simple_reward.py
├── training/
│   ├── __init__.py
│   ├── grpo.py
│   └── trainer.py
└── visualization/
    ├── __init__.py
    └── plotter.py
```

## Important Notes

1. **GPU**: Colab provides free GPU - the code will automatically use CUDA if available
2. **File Persistence**: Files uploaded to Colab are temporary. Download outputs if you want to keep them
3. **Runtime**: Training takes ~5-10 minutes on GPU, ~20-30 minutes on CPU
4. **Outputs**: Visualizations are saved to `outputs/` directory

## Troubleshooting

### Import Errors
- Make sure all `__init__.py` files exist in subdirectories
- Check that `sys.path` includes the project root

### CUDA Out of Memory
- Reduce `batch_size` in config (e.g., from 4 to 2)
- Reduce `num_samples_per_prompt` (e.g., from 4 to 2)

### File Not Found Errors
- Run `generate_dataset()` before loading datasets
- Check that directory structure is correct

## Customization

You can modify training parameters:

```python
config = {
    "batch_size": 8,              # Increase batch size
    "num_samples_per_prompt": 8,  # More samples per prompt
    "num_steps": 30,              # More flow steps
    "learning_rate": 5e-4,        # Lower learning rate
    "num_epochs": 100,            # More epochs
    # ... etc
}
```

## Download Results

To download outputs from Colab:

```python
from google.colab import files
from pathlib import Path

# Download specific file
files.download('outputs/training_curves.png')

# Or download entire outputs directory
!zip -r outputs.zip outputs/
files.download('outputs.zip')
```
