# MNIST Reward Functions for FlowGRPO

This document describes several reward function options for training flow matching models on MNIST using GRPO (Group Relative Policy Optimization).

## Overview

When training on MNIST, you need reward functions that:
1. **Encourage correct digit generation** (classification accuracy)
2. **Promote visual quality** (sharpness, contrast, realism)
3. **Prevent mode collapse** (diversity)
4. **Work well with GRPO** (group-based advantage computation)

## Reward Function Options

### 1. Classification Accuracy Reward ⭐ **Recommended as Base**

**Class**: `MNISTClassifierReward`

**What it does**: Uses a pre-trained MNIST classifier to check if the generated digit matches the target class.

**Pros**:
- Directly optimizes for correctness
- Fast and efficient computation
- Clear optimization signal
- Essential for digit generation tasks

**Cons**:
- May not penalize visual artifacts if classifier is lenient
- Doesn't ensure high visual quality on its own

**Usage**:
```python
from rewards.mnist_rewards import MNISTClassifierReward

reward_fn = MNISTClassifierReward(device='cuda', pretrained=True)
rewards = reward_fn(images, prompts=["digit 5", "digit 3", ...])
```

**When to use**: Always include this as a base reward. It's the most important for MNIST.

---

### 2. Confidence Reward

**Class**: `MNISTConfidenceReward`

**What it does**: Rewards high classifier confidence, encouraging clear, unambiguous digit generation.

**Pros**:
- Encourages sharp, clear outputs
- Reduces ambiguous generations
- Complements classification accuracy

**Cons**:
- Doesn't directly optimize for correctness
- Best used in combination with other rewards

**Usage**:
```python
from rewards.mnist_rewards import MNISTConfidenceReward, MNISTClassifierReward

classifier = MNISTClassifierReward(device='cuda')
reward_fn = MNISTConfidenceReward(classifier)
```

**When to use**: Add this when you want sharper, more confident digit generation.

---

### 3. Perceptual Quality Reward

**Class**: `MNISTPerceptualReward`

**What it does**: Measures perceptual similarity to real MNIST samples using SSIM or statistical properties.

**Pros**:
- Encourages realistic digit appearance
- Captures visual quality beyond classification
- Good for avoiding artifacts

**Cons**:
- More computationally expensive
- Requires reference samples (or statistical properties)
- May not directly optimize for correctness

**Usage**:
```python
from rewards.mnist_rewards import MNISTPerceptualReward

reward_fn = MNISTPerceptualReward(device='cuda', metric='ssim')
```

**When to use**: Use when visual quality is important and you have computational budget.

---

### 4. Image Quality Reward

**Class**: `MNISTQualityReward`

**What it does**: Measures sharpness, contrast, and brightness without requiring references.

**Pros**:
- Fast computation
- No external dependencies
- Encourages sharp, high-contrast images
- Works well as a regularizer

**Cons**:
- Doesn't ensure correctness
- May not capture all quality aspects

**Usage**:
```python
from rewards.mnist_rewards import MNISTQualityReward

reward_fn = MNISTQualityReward(device='cuda')
```

**When to use**: Good lightweight option for improving visual quality.

---

### 5. Diversity Reward

**Class**: `MNISTDiversityReward`

**What it does**: Encourages diversity in generated samples for the same prompt, preventing mode collapse.

**Pros**:
- Prevents mode collapse
- Encourages exploration
- Good for multi-modal generation

**Cons**:
- May conflict with quality rewards
- Should be used with small weight
- Requires grouping by prompt_id

**Usage**:
```python
from rewards.mnist_rewards import MNISTDiversityReward

reward_fn = MNISTDiversityReward(device='cuda')
```

**When to use**: Add with small weight (0.05-0.1) if you notice mode collapse.

---

### 6. Combined Multi-Objective Reward ⭐ **Recommended**

**Class**: `CombinedMNISTReward`

**What it does**: Combines multiple reward signals with configurable weights.

**Pros**:
- Balances multiple objectives
- Flexible and configurable
- Typically best performance
- Easy to tune

**Cons**:
- Requires tuning weights
- More complex than single rewards

**Usage**:
```python
from rewards.mnist_rewards import CombinedMNISTReward, get_recommended_reward_config

# Use recommended configuration
reward_fn = get_recommended_reward_config('balanced', device='cuda')

# Or customize
reward_fn = CombinedMNISTReward(
    device='cuda',
    weights={
        'classification': 0.5,
        'confidence': 0.2,
        'quality': 0.25,
        'perceptual': 0.05,
        'diversity': 0.0,
    },
    use_classifier=True,
    use_confidence=True,
    use_quality=True,
    use_perceptual=False,
    use_diversity=False,
)
```

**When to use**: This is the recommended approach for most training scenarios.

---

## Recommended Configurations

### Configuration 1: "Accuracy" (Fast Training)
Focus on classification accuracy only.

```python
reward_fn = get_recommended_reward_config('accuracy', device='cuda')
```

**Weights**:
- Classification: 0.7
- Confidence: 0.3
- Quality: 0.0

**Use when**: You want fast training and correctness is the main goal.

---

### Configuration 2: "Balanced" ⭐ **Recommended**
Balance between accuracy and quality.

```python
reward_fn = get_recommended_reward_config('balanced', device='cuda')
```

**Weights**:
- Classification: 0.5
- Confidence: 0.2
- Quality: 0.25
- Perceptual: 0.05

**Use when**: You want both correct and visually appealing digits (most common case).

---

### Configuration 3: "Quality" (High Visual Quality)
Focus on visual quality.

```python
reward_fn = get_recommended_reward_config('quality', device='cuda')
```

**Weights**:
- Classification: 0.3
- Confidence: 0.2
- Quality: 0.4
- Perceptual: 0.1

**Use when**: Visual quality is more important than pure accuracy.

---

### Configuration 4: "Diverse" (Prevent Mode Collapse)
Include diversity reward.

```python
reward_fn = get_recommended_reward_config('diverse', device='cuda')
```

**Weights**:
- Classification: 0.4
- Confidence: 0.2
- Quality: 0.25
- Perceptual: 0.05
- Diversity: 0.1

**Use when**: You notice the model generating similar outputs for the same prompt.

---

## Integration with GRPO

All reward functions are designed to work with GRPO's group-based advantage computation:

1. **Group by prompt**: Samples with the same prompt (e.g., "digit 5") are grouped together
2. **Compute group mean**: Average reward within each group
3. **Compute advantages**: `advantage = reward - group_mean`
4. **Update policy**: Use advantages for policy gradient updates

The reward functions return rewards for each sample, and GRPO handles the grouping automatically based on `prompt_ids`.

---

## Implementation Notes

### Pre-training the Classifier

Before using `MNISTClassifierReward`, you should train a classifier on MNIST:

```python
# Train a simple classifier on MNIST
# This is a one-time setup step
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Load MNIST
train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train classifier (simplified example)
classifier = MNISTClassifierReward(device='cuda', pretrained=False)
# ... training loop ...
# Save the trained classifier
torch.save(classifier.classifier.state_dict(), 'mnist_classifier.pth')
```

### Prompt Format

Prompts should be in the format:
- `"digit 0"`, `"digit 1"`, ..., `"digit 9"`
- `"number 0"`, `"number 1"`, etc.

The reward functions will parse these to extract the target digit.

### Image Format

Reward functions expect images in one of these formats:
- `[batch_size, 1, 28, 28]` (grayscale)
- `[batch_size, 3, H, W]` (RGB, will be converted to grayscale)
- `[batch_size, 28, 28]` (grayscale, will add channel dimension)

Images will be automatically resized to 28x28 if needed.

---

## Example Training Setup

```python
from rewards.mnist_rewards import get_recommended_reward_config
from training.trainer import FlowGRPOTrainer

# Create reward function
reward_fn = get_recommended_reward_config('balanced', device='cuda')

# Use in trainer (same as tutorial)
trainer = FlowGRPOTrainer(
    model=model,
    prompt_encoder=prompt_encoder,
    reward_fn=reward_fn,  # Use MNIST reward here
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    config=trainer_config,
)

trainer.train(num_epochs=100)
```

---

## Tuning Tips

1. **Start with "balanced" configuration**: It's a good default
2. **Monitor reward components**: Log individual reward components to see which ones are driving improvements
3. **Adjust weights gradually**: Change weights by 0.1 at a time
4. **Classification weight**: Keep this high (≥0.4) for MNIST
5. **Diversity weight**: Keep low (≤0.1) or zero unless you see mode collapse
6. **Quality vs Accuracy trade-off**: Increase quality weight if digits look blurry, increase classification if accuracy is low

---

## References

- GRPO paper: [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
- MNIST dataset: [Yann LeCun's MNIST](http://yann.lecun.com/exdb/mnist/)
- SSIM: Structural Similarity Index
- LPIPS: Learned Perceptual Image Patch Similarity
