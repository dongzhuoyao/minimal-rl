# Interesting & Advanced MNIST Reward Functions

This document describes creative and advanced reward functions for MNIST training with FlowGRPO. These go beyond basic classification and quality metrics to provide unique optimization signals.

## Overview

These reward functions explore different aspects of digit generation:
- **Distribution matching** (adversarial, contrastive)
- **Structural properties** (geometric, symmetry)
- **Robustness** (adversarial robustness, ensemble)
- **Feature learning** (feature-based, gradient-based)
- **Reconstruction** (autoencoder)

---

## 1. Adversarial/Discriminator Reward 🎯

**Class**: `MNISTAdversarialReward`

**What it does**: Uses a GAN discriminator trained to distinguish real vs fake MNIST digits. Higher reward = more "realistic" according to the discriminator.

**Key Idea**: Instead of optimizing for a specific property, optimize for "realism" as judged by a discriminator.

**Pros**:
- Encourages realistic digit appearance
- Captures distribution-level quality
- Good for avoiding mode collapse
- Can discover natural digit properties

**Cons**:
- Requires training a discriminator first
- Can be unstable during training (mode collapse)
- May not ensure correctness
- Needs careful GAN training

**Setup**:
```python
# First, train a discriminator on real MNIST
# Then use it for rewards
from rewards.mnist_rewards import MNISTAdversarialReward

reward_fn = MNISTAdversarialReward(device='cuda')
# Load your trained discriminator weights
reward_fn.discriminator.load_state_dict(torch.load('discriminator.pth'))
```

**When to use**: When you want distribution-level realism and have a trained discriminator.

**Research connection**: Similar to GAN training, but used as a reward signal for RL.

---

## 2. Autoencoder Reconstruction Reward 🔄

**Class**: `MNISTAutoencoderReward`

**What it does**: Measures how well an autoencoder can reconstruct the generated digit. Higher reconstruction quality = better digit structure.

**Key Idea**: If an autoencoder can reconstruct it well, the digit has good structure.

**Pros**:
- Encourages well-structured digits
- Captures semantic features
- Good for ensuring digit coherence
- Learns meaningful representations

**Cons**:
- Requires training an autoencoder first
- May not directly optimize for correctness
- Can be biased by autoencoder's training data

**Setup**:
```python
from rewards.mnist_rewards import MNISTAutoencoderReward

reward_fn = MNISTAutoencoderReward(device='cuda')
# Load your trained autoencoder weights
reward_fn.autoencoder.load_state_dict(torch.load('autoencoder.pth'))
```

**When to use**: When you want to ensure digits have coherent structure and have a trained autoencoder.

**Research connection**: Similar to VAE reconstruction loss, but used as reward.

---

## 3. Feature-Based Reward 🧠

**Class**: `MNISTFeatureReward`

**What it does**: Extracts intermediate features from a pre-trained classifier and rewards based on feature quality/magnitude.

**Key Idea**: Intermediate features capture semantic information better than final predictions.

**Pros**:
- Captures high-level semantic features
- More robust than final classification
- Encourages meaningful representations
- Can guide learning at different abstraction levels

**Cons**:
- Requires understanding classifier architecture
- More complex computation
- May need reference features

**Setup**:
```python
from rewards.mnist_rewards import MNISTClassifierReward, MNISTFeatureReward

classifier = MNISTClassifierReward(device='cuda')
reward_fn = MNISTFeatureReward(classifier, layer_name='fc1')
```

**When to use**: When you want to optimize for semantic features rather than just final classification.

**Research connection**: Similar to feature matching in GANs or knowledge distillation.

---

## 4. Geometric Reward 📐

**Class**: `MNISTGeometricReward`

**What it does**: Measures geometric properties like centroid position, aspect ratio, and fill ratio.

**Key Idea**: Real MNIST digits have specific geometric properties (centered, reasonable size, etc.).

**Pros**:
- Fast computation (no neural networks)
- Encourages proper digit structure
- No external models needed
- Interpretable metrics

**Cons**:
- May not capture all quality aspects
- Digit-specific (works better for some digits)
- Can be too rigid

**Setup**:
```python
from rewards.mnist_rewards import MNISTGeometricReward

reward_fn = MNISTGeometricReward(device='cuda')
```

**When to use**: As a lightweight regularizer to ensure digits are properly positioned and sized.

**Metrics measured**:
- Centroid position (should be near center)
- Aspect ratio (should be roughly square)
- Fill ratio (reasonable amount of foreground pixels)

---

## 5. Symmetry Reward 🔀

**Class**: `MNISTSymmetryReward`

**What it does**: Rewards digits that should be symmetric (0, 1, 3, 8) for having proper symmetry.

**Key Idea**: Some digits are naturally symmetric, so we should reward that property.

**Pros**:
- Encourages proper digit structure
- Fast computation
- Digit-specific optimization
- Can improve visual quality

**Cons**:
- Only works for symmetric digits
- May conflict with other rewards for asymmetric digits
- Need to know which digits are symmetric

**Setup**:
```python
from rewards.mnist_rewards import MNISTSymmetryReward

reward_fn = MNISTSymmetryReward(device='cuda')
```

**When to use**: When generating symmetric digits (0, 1, 3, 8) and you want to ensure proper symmetry.

**Symmetric digits**:
- Vertically symmetric: 0, 1, 3, 8
- Horizontally symmetric: 0, 8

---

## 6. Gradient-Based Reward ⬆️

**Class**: `MNISTGradientReward`

**What it does**: Uses gradients from classifier to measure how much the image affects the target class probability.

**Key Idea**: Strong gradient signal = image is moving in the right direction for the target class.

**Pros**:
- Provides direct optimization signal
- Encourages correct class generation
- Can guide generation process
- Interpretable (gradient direction)

**Cons**:
- Requires gradient computation (more expensive)
- May be unstable
- Can lead to adversarial examples

**Setup**:
```python
from rewards.mnist_rewards import MNISTClassifierReward, MNISTGradientReward

classifier = MNISTClassifierReward(device='cuda')
reward_fn = MNISTGradientReward(classifier)
```

**When to use**: When you want direct gradient-based guidance, similar to adversarial training.

**Research connection**: Similar to gradient-based adversarial attacks, but used positively.

---

## 7. Ensemble Reward 🎲

**Class**: `MNISTEnsembleReward`

**What it does**: Combines predictions from multiple classifiers for more robust rewards.

**Key Idea**: Multiple opinions are better than one - reduces overfitting to a single model.

**Pros**:
- More robust than single classifier
- Reduces overfitting to one model
- Better generalization
- Can capture different aspects

**Cons**:
- More computationally expensive
- Requires multiple trained models
- More complex setup

**Setup**:
```python
from rewards.mnist_rewards import MNISTEnsembleReward

reward_fn = MNISTEnsembleReward(num_classifiers=3, device='cuda')
# Load different trained models for each classifier
for i, classifier in enumerate(reward_fn.classifiers):
    classifier.classifier.load_state_dict(torch.load(f'classifier_{i}.pth'))
```

**When to use**: When you have multiple trained classifiers and want robust, ensemble-based rewards.

**Research connection**: Standard ensemble methods applied to reward computation.

---

## 8. Contrastive Reward 🔍

**Class**: `MNISTContrastiveReward`

**What it does**: Uses contrastive learning to measure similarity to reference MNIST samples.

**Key Idea**: Compare generated digits to real MNIST digits in feature space.

**Pros**:
- Encourages realistic digit appearance
- Uses learned representations
- Good for distribution matching
- Can discover natural clusters

**Cons**:
- Requires reference samples
- More complex setup
- Need to maintain reference features

**Setup**:
```python
from rewards.mnist_rewards import MNISTClassifierReward, MNISTContrastiveReward

classifier = MNISTClassifierReward(device='cuda')
reward_fn = MNISTContrastiveReward(classifier, num_references=100)

# Load reference MNIST samples and extract features
# reward_fn.reference_features = extract_features_from_real_mnist()
```

**When to use**: When you want to match the distribution of real MNIST digits.

**Research connection**: Contrastive learning (SimCLR, MoCo) applied to reward computation.

---

## 9. Adversarial Robustness Reward 🛡️

**Class**: `MNISTAdversarialRobustnessReward`

**What it does**: Rewards digits that are robust to small perturbations (adversarial robustness).

**Key Idea**: Good digits should be stable under small changes.

**Pros**:
- Encourages robust features
- Reduces sensitivity to noise
- Better generalization
- More reliable outputs

**Cons**:
- More computationally expensive (needs forward passes with perturbations)
- May reduce diversity
- Can be conservative

**Setup**:
```python
from rewards.mnist_rewards import MNISTClassifierReward, MNISTAdversarialRobustnessReward

classifier = MNISTClassifierReward(device='cuda')
reward_fn = MNISTAdversarialRobustnessReward(classifier, epsilon=0.1)
```

**When to use**: When you want robust, stable digit generation that's not sensitive to small changes.

**Research connection**: Adversarial robustness research, but used as a reward signal.

---

## Combining Multiple Rewards

You can combine these interesting rewards with basic ones:

```python
from rewards.mnist_rewards import (
    CombinedMNISTReward,
    MNISTGeometricReward,
    MNISTSymmetryReward,
    MNISTClassifierReward,
    MNISTConfidenceReward,
    MNISTQualityReward,
)

# Create individual rewards
classifier = MNISTClassifierReward(device='cuda')
geometric = MNISTGeometricReward(device='cuda')
symmetry = MNISTSymmetryReward(device='cuda')
confidence = MNISTConfidenceReward(classifier)
quality = MNISTQualityReward(device='cuda')

# Custom combined reward
class CustomMNISTReward:
    def __init__(self):
        self.rewards = {
            'classification': classifier,
            'confidence': confidence,
            'quality': quality,
            'geometric': geometric,
            'symmetry': symmetry,
        }
        self.weights = {
            'classification': 0.4,
            'confidence': 0.15,
            'quality': 0.2,
            'geometric': 0.15,
            'symmetry': 0.1,
        }
    
    def __call__(self, images, prompts, **kwargs):
        total_rewards = torch.zeros(images.shape[0], device=images.device)
        for name, reward_fn in self.rewards.items():
            weight = self.weights[name]
            rewards = reward_fn(images, prompts, **kwargs)
            total_rewards += weight * rewards
        return total_rewards
```

---

## Recommended Combinations

### 1. "Structural" Configuration
Focus on geometric and structural properties:
- Geometric: 0.3
- Symmetry: 0.2
- Classification: 0.3
- Quality: 0.2

### 2. "Robust" Configuration
Focus on robustness and stability:
- Classification: 0.4
- Adversarial Robustness: 0.3
- Ensemble: 0.2
- Quality: 0.1

### 3. "Realistic" Configuration
Focus on distribution matching:
- Adversarial: 0.4
- Contrastive: 0.3
- Classification: 0.2
- Quality: 0.1

### 4. "Feature-Rich" Configuration
Focus on semantic features:
- Feature-based: 0.3
- Classification: 0.3
- Autoencoder: 0.2
- Quality: 0.2

---

## Implementation Tips

1. **Start simple**: Begin with basic rewards, then add interesting ones
2. **Monitor components**: Log individual reward components to understand what's driving improvements
3. **Tune weights carefully**: Interesting rewards can have strong effects - tune gradually
4. **Consider computational cost**: Some rewards (adversarial, ensemble) are more expensive
5. **Pre-train components**: Many rewards need pre-trained models (discriminator, autoencoder, etc.)
6. **Combine thoughtfully**: Not all rewards work well together - test combinations

---

## Research Connections

These reward functions connect to various research areas:

- **GANs**: Adversarial reward
- **VAEs**: Autoencoder reward
- **Contrastive Learning**: Contrastive reward
- **Adversarial Robustness**: Robustness reward
- **Ensemble Methods**: Ensemble reward
- **Feature Matching**: Feature-based reward
- **Gradient Methods**: Gradient-based reward
- **Geometric Deep Learning**: Geometric reward

---

## Future Directions

Potential extensions:

1. **Learned Rewards**: Train a reward model end-to-end
2. **Temporal Rewards**: For sequential generation
3. **Multi-Scale Rewards**: Combine rewards at different resolutions
4. **Conditional Rewards**: Different rewards for different digit classes
5. **Uncertainty-Based Rewards**: Use model uncertainty as signal
6. **Style Transfer Rewards**: Match style of reference digits
7. **Causal Rewards**: Measure causal relationships in digit structure

---

## References

- Flow-GRPO: [Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
- GANs: Generative Adversarial Networks
- VAEs: Variational Autoencoders
- Contrastive Learning: SimCLR, MoCo
- Adversarial Robustness: Adversarial Training
