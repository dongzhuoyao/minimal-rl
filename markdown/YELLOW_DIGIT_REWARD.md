# Yellow Digit Reward Function

This document describes reward functions for generating yellow digits in MNIST using FlowGRPO.

## Overview

The yellow reward functions encourage the model to generate digits with yellow color. This is useful when you want to bias generation towards a specific color while maintaining digit correctness.

## Reward Functions

### 1. MNISTYellowReward ⭐ **Simple Yellow Reward**

**What it does**: Rewards images with yellow color.

**Features**:
- Works with both RGB and grayscale images
- For RGB: Detects yellow directly (high R+G, low B)
- For grayscale: Interprets bright pixels as "yellow"

**Usage**:
```python
from rewards.mnist_rewards import MNISTYellowReward

# Create yellow reward
yellow_reward = MNISTYellowReward(
    device='cuda',
    yellow_threshold=0.6,      # Threshold for yellow detection in RGB
    brightness_threshold=0.7    # Threshold for brightness in grayscale
)

# Compute rewards
rewards = yellow_reward(images, prompts)
```

**Parameters**:
- `yellow_threshold` (default 0.6): Threshold for yellow detection in RGB images
- `brightness_threshold` (default 0.7): Threshold for brightness in grayscale images

**How it works**:
- **RGB images**: Detects yellow as pixels with high R+G and low B
- **Grayscale images**: Interprets bright pixels as "yellow" (since yellow is bright)

---

### 2. MNISTColorReward 🎨 **General Color Reward**

**What it does**: More flexible - can target any color (yellow, red, blue, etc.).

**Usage**:
```python
from rewards.mnist_rewards import MNISTColorReward

# For yellow
yellow_reward = MNISTColorReward(device='cuda', target_color='yellow')

# For red
red_reward = MNISTColorReward(device='cuda', target_color='red')

# For custom RGB color
custom_reward = MNISTColorReward(
    device='cuda', 
    target_rgb=(1.0, 0.8, 0.2)  # Custom orange-yellow
)
```

**Supported colors**:
- `'yellow'`, `'red'`, `'green'`, `'blue'`
- `'orange'`, `'purple'`, `'cyan'`, `'magenta'`
- `'white'`, `'black'`
- Or use `target_rgb` for custom colors

---

### 3. MNISTYellowDigitReward ⭐⭐ **Recommended: Yellow + Correct**

**What it does**: Combines yellow color reward with digit correctness reward.

**Why use this**: Ensures you get yellow digits that are still recognizable and correct!

**Usage**:
```python
from rewards.mnist_rewards import (
    MNISTClassifierReward,
    MNISTYellowReward,
    MNISTYellowDigitReward
)

# Create component rewards
classifier = MNISTClassifierReward(device='cuda')
yellow = MNISTYellowReward(device='cuda')

# Create combined reward
combined_reward = MNISTYellowDigitReward(
    classifier_reward=classifier,
    yellow_reward=yellow,
    digit_weight=0.6,    # 60% weight on correctness
    yellow_weight=0.4    # 40% weight on yellow color
)

# Use in training
rewards = combined_reward(images, prompts)
```

**Weight tuning**:
- Higher `digit_weight` (e.g., 0.7-0.8): Prioritize correctness, yellow is bonus
- Higher `yellow_weight` (e.g., 0.5-0.6): Prioritize yellow color, correctness is bonus
- Balanced (0.5/0.5): Equal importance

---

## Example: Training with Yellow Reward

### Option 1: Pure Yellow Reward (may sacrifice correctness)

```python
from rewards.mnist_rewards import MNISTYellowReward
from training.trainer import FlowGRPOTrainer

# Create yellow reward
reward_fn = MNISTYellowReward(device='cuda')

# Use in trainer
trainer = FlowGRPOTrainer(
    model=model,
    prompt_encoder=prompt_encoder,
    reward_fn=reward_fn,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    config=trainer_config,
)
```

**Warning**: This may generate yellow blobs that aren't recognizable digits!

---

### Option 2: Yellow + Correctness (Recommended)

```python
from rewards.mnist_rewards import (
    MNISTClassifierReward,
    MNISTYellowReward,
    MNISTYellowDigitReward
)

# Create combined reward
classifier = MNISTClassifierReward(device='cuda')
yellow = MNISTYellowReward(device='cuda')
reward_fn = MNISTYellowDigitReward(
    classifier_reward=classifier,
    yellow_reward=yellow,
    digit_weight=0.6,
    yellow_weight=0.4
)

# Use in trainer
trainer = FlowGRPOTrainer(
    model=model,
    prompt_encoder=prompt_encoder,
    reward_fn=reward_fn,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    config=trainer_config,
)
```

**Result**: Generates yellow digits that are still recognizable!

---

### Option 3: Multi-Objective with Yellow

```python
from rewards.mnist_rewards import (
    MNISTClassifierReward,
    MNISTConfidenceReward,
    MNISTYellowReward,
    MNISTQualityReward
)

classifier = MNISTClassifierReward(device='cuda')
confidence = MNISTConfidenceReward(classifier)
yellow = MNISTYellowReward(device='cuda')
quality = MNISTQualityReward(device='cuda')

class YellowMultiObjectiveReward:
    def __init__(self):
        self.classifier = classifier
        self.confidence = confidence
        self.yellow = yellow
        self.quality = quality
    
    def __call__(self, images, prompts, **kwargs):
        cls_rewards = self.classifier(images, prompts, **kwargs)
        conf_rewards = self.confidence(images, prompts, **kwargs)
        yellow_rewards = self.yellow(images, prompts, **kwargs)
        quality_rewards = self.quality(images, prompts, **kwargs)
        
        # Weighted combination
        total = (0.4 * cls_rewards +      # Correctness
                 0.2 * conf_rewards +     # Confidence
                 0.3 * yellow_rewards +    # Yellow color
                 0.1 * quality_rewards)    # Quality
        return total

reward_fn = YellowMultiObjectiveReward()
```

---

## Tuning Tips

### 1. Adjust Yellow Threshold

```python
# More strict yellow detection (fewer pixels count as yellow)
yellow_reward = MNISTYellowReward(device='cuda', yellow_threshold=0.7)

# More lenient yellow detection (more pixels count as yellow)
yellow_reward = MNISTYellowReward(device='cuda', yellow_threshold=0.5)
```

### 2. Balance Yellow vs Correctness

```python
# Prioritize correctness
combined = MNISTYellowDigitReward(
    classifier_reward=classifier,
    yellow_reward=yellow,
    digit_weight=0.8,  # High weight on correctness
    yellow_weight=0.2  # Low weight on yellow
)

# Prioritize yellow color
combined = MNISTYellowDigitReward(
    classifier_reward=classifier,
    yellow_reward=yellow,
    digit_weight=0.3,  # Low weight on correctness
    yellow_weight=0.7  # High weight on yellow
)
```

### 3. Monitor Both Metrics

During training, log both:
- Classification accuracy (digit correctness)
- Yellow ratio (fraction of yellow pixels)

This helps you tune the weights appropriately.

---

## How Yellow Detection Works

### RGB Images

Yellow in RGB space:
- **High Red** (R ≈ 1.0)
- **High Green** (G ≈ 1.0)
- **Low Blue** (B ≈ 0.0)

Formula: `yellow_score = (R + G) / 2 - B`

Pixels with `yellow_score > threshold` are considered yellow.

### Grayscale Images

Since grayscale can't represent color:
- Yellow is interpreted as **bright pixels**
- Brightness threshold determines what counts as "yellow"

This is an approximation - for true yellow color, use RGB images!

---

## Expected Results

### With Pure Yellow Reward:
- ✅ High yellow ratio
- ❌ May generate unrecognizable yellow blobs
- ❌ Digits may not be correct

### With Yellow + Correctness:
- ✅ High yellow ratio
- ✅ Digits are still recognizable
- ✅ Good balance between color and correctness

### With Multi-Objective:
- ✅ High yellow ratio
- ✅ Correct digits
- ✅ Good visual quality
- ✅ Best overall results

---

## Troubleshooting

### Problem: Digits aren't yellow enough
**Solution**: 
- Increase `yellow_weight` in combined reward
- Decrease `yellow_threshold` to be more lenient
- Ensure you're generating RGB images (not grayscale)

### Problem: Yellow but digits are wrong
**Solution**:
- Increase `digit_weight` in combined reward
- Use `MNISTYellowDigitReward` instead of pure yellow reward

### Problem: Correct digits but not yellow
**Solution**:
- Increase `yellow_weight` in combined reward
- Decrease `yellow_threshold` to be more lenient
- Check if model can generate colors (may need RGB output)

---

## Research Applications

This reward function is useful for:
1. **Style transfer**: Generate digits in specific colors
2. **Conditional generation**: Color as a condition
3. **Bias studies**: How color affects generation
4. **Multi-objective RL**: Balancing multiple goals

---

## Example Output

With proper tuning, you should see:
- Generated digits that are yellow-colored
- Digits that are still recognizable
- Good classification accuracy maintained
- Visual quality preserved

The exact balance depends on your weight settings!
