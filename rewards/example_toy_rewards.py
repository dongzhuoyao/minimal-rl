"""
Example: Using Toy Rewards with FlowGRPO

This script demonstrates how to use the toy reward functions for experimentation.
These rewards are simple, fun, and great for understanding how different reward
shapes affect model behavior.

Toy Rewards Available:
1. 'bright' - Rewards brighter/more visible digits
2. 'centered' - Rewards digits centered in the image
3. 'sparse' - Rewards sparse/thin digits (fewer pixels)
4. 'large' - Rewards large/bold digits (more pixels)
5. 'contrast' - Rewards high contrast images
6. 'tiny' - Rewards tiny/compact digits (small bounding box)
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rewards.mnist_rewards import (
    get_recommended_reward_config,
    MNISTBrightDigitReward,
    MNISTCenteredDigitReward,
    MNISTSparseDigitReward,
    MNISTLargeDigitReward,
    MNISTHighContrastReward,
    MNISTTinyDigitReward,
)
from dataset import MNISTDataset


def example_toy_rewards():
    """Demonstrate different toy rewards."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load a few sample images
    dataset = MNISTDataset('dataset', split='test', download=True)
    sample_indices = [0, 1, 2, 3]
    images = []
    labels = []
    
    for idx in sample_indices:
        sample = dataset[idx]
        img = sample['image'].view(1, 28, 28)  # [1, 28, 28]
        images.append(img)
        labels.append(sample['label'].item())
    
    images_tensor = torch.stack(images).to(device)  # [4, 1, 28, 28]
    prompts = [f"digit {label}" for label in labels]
    
    print("=" * 60)
    print("Toy Rewards Demo")
    print("=" * 60)
    print(f"Testing with {len(images_tensor)} sample images")
    print(f"Labels: {labels}\n")
    
    # Test each toy reward
    toy_rewards = {
        'bright': MNISTBrightDigitReward(device=device),
        'centered': MNISTCenteredDigitReward(device=device),
        'sparse': MNISTSparseDigitReward(device=device),
        'large': MNISTLargeDigitReward(device=device),
        'contrast': MNISTHighContrastReward(device=device),
        'tiny': MNISTTinyDigitReward(device=device),
    }
    
    print("Reward Scores:")
    print("-" * 60)
    for name, reward_fn in toy_rewards.items():
        rewards = reward_fn(images_tensor, prompts)
        avg_reward = rewards.mean().item()
        print(f"{name:12s}: {rewards.cpu().numpy()} (avg: {avg_reward:.4f})")
    
    print("\n" + "=" * 60)


def example_combined_toy_reward():
    """Example: Combine toy reward with classifier for correctness."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from rewards.mnist_rewards import MNISTClassifierReward
    
    # Create classifier and toy reward
    classifier = MNISTClassifierReward(device=device)
    bright_reward = MNISTBrightDigitReward(device=device)
    
    # Load sample images
    dataset = MNISTDataset('dataset', split='test', download=True)
    sample = dataset[0]
    img = sample['image'].view(1, 1, 28, 28).to(device)
    label = sample['label'].item()
    prompt = f"digit {label}"
    
    # Compute both rewards
    classifier_reward = classifier(img, [prompt])
    bright_reward_val = bright_reward(img, [prompt])
    
    # Combine (60% correctness, 40% brightness)
    combined = 0.6 * classifier_reward + 0.4 * bright_reward_val
    
    print("=" * 60)
    print("Combined Reward Example")
    print("=" * 60)
    print(f"Image label: {label}")
    print(f"Classifier reward: {classifier_reward.item():.4f}")
    print(f"Bright reward: {bright_reward_val.item():.4f}")
    print(f"Combined (60% correct + 40% bright): {combined.item():.4f}")
    print("=" * 60)


def example_training_with_toy_reward():
    """Example: How to use toy rewards in training."""
    print("""
To use a toy reward in training, modify train.yaml:

```yaml
reward:
  config: "bright"  # or "centered", "sparse", "large", "contrast", "tiny"
```

Or use via command line:
```bash
python train.py reward.config=bright
python train.py reward.config=sparse
python train.py reward.config=contrast
```

You can also combine toy rewards with classifier:

```python
from rewards.mnist_rewards import (
    MNISTClassifierReward,
    MNISTBrightDigitReward
)

classifier = MNISTClassifierReward(device='cuda')
bright = MNISTBrightDigitReward(device='cuda')

def combined_reward(images, prompts):
    correct = classifier(images, prompts)
    bright_val = bright(images, prompts)
    return 0.7 * correct + 0.3 * bright_val

# Use in training
reward_fn = combined_reward
```
""")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Toy Rewards Examples")
    print("=" * 60 + "\n")
    
    # Example 1: Test all toy rewards
    example_toy_rewards()
    
    print("\n")
    
    # Example 2: Combine toy reward with classifier
    example_combined_toy_reward()
    
    print("\n")
    
    # Example 3: Usage in training
    example_training_with_toy_reward()
