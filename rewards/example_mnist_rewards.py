"""
Example: Using MNIST Reward Functions with FlowGRPO

This script demonstrates how to set up and use different reward functions
for training flow matching models on MNIST with GRPO.
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rewards.mnist_rewards import (
    MNISTClassifierReward,
    MNISTConfidenceReward,
    MNISTQualityReward,
    MNISTPerceptualReward,
    MNISTDiversityReward,
    CombinedMNISTReward,
    get_recommended_reward_config,
    # Interesting rewards
    MNISTGeometricReward,
    MNISTSymmetryReward,
    MNISTFeatureReward,
    MNISTGradientReward,
    MNISTAdversarialRobustnessReward,
    MNISTEnsembleReward,
    MNISTContrastiveReward,
    MNISTAdversarialReward,
    MNISTAutoencoderReward,
    # Color rewards
    MNISTYellowReward,
    MNISTColorReward,
    MNISTYellowDigitReward,
)


def example_single_reward():
    """Example: Using a single reward function."""
    print("=" * 60)
    print("Example 1: Single Reward Function (Classification)")
    print("=" * 60)
    
    # Create reward function
    reward_fn = MNISTClassifierReward(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulate generated images (batch_size=4, 1 channel, 28x28)
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    images = torch.sigmoid(images)  # Normalize to [0, 1]
    
    # Prompts
    prompts = ["digit 0", "digit 1", "digit 2", "digit 3"]
    
    # Compute rewards
    rewards = reward_fn(images, prompts)
    
    print(f"Images shape: {images.shape}")
    print(f"Prompts: {prompts}")
    print(f"Rewards: {rewards}")
    print(f"Mean reward: {rewards.mean().item():.4f}")
    print()


def example_combined_reward():
    """Example: Using combined multi-objective reward."""
    print("=" * 60)
    print("Example 2: Combined Reward Function (Balanced)")
    print("=" * 60)
    
    # Get recommended balanced configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    reward_fn = get_recommended_reward_config('balanced', device=device)
    
    # Simulate generated images
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    images = torch.sigmoid(images)
    
    prompts = ["digit 5", "digit 5", "digit 7", "digit 7"]  # Same prompts for grouping
    
    # Compute rewards
    rewards = reward_fn(images, prompts)
    
    print(f"Images shape: {images.shape}")
    print(f"Prompts: {prompts}")
    print(f"Rewards: {rewards}")
    print(f"Mean reward: {rewards.mean().item():.4f}")
    print()


def example_custom_weights():
    """Example: Custom reward weights."""
    print("=" * 60)
    print("Example 3: Custom Reward Weights")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create custom reward function
    reward_fn = CombinedMNISTReward(
        device=device,
        weights={
            'classification': 0.6,  # Higher weight on accuracy
            'confidence': 0.15,
            'quality': 0.2,
            'perceptual': 0.05,
            'diversity': 0.0,
        },
        use_classifier=True,
        use_confidence=True,
        use_quality=True,
        use_perceptual=False,
        use_diversity=False,
    )
    
    # Simulate generated images
    batch_size = 8
    images = torch.randn(batch_size, 1, 28, 28)
    images = torch.sigmoid(images)
    
    prompts = ["digit 3"] * batch_size  # Same prompt for all
    
    # Compute rewards
    rewards = reward_fn(images, prompts)
    
    print(f"Custom weights: classification=0.6, confidence=0.15, quality=0.2")
    print(f"Rewards: {rewards}")
    print(f"Mean reward: {rewards.mean().item():.4f}")
    print(f"Std reward: {rewards.std().item():.4f}")
    print()


def example_all_configurations():
    """Example: Compare different reward configurations."""
    print("=" * 60)
    print("Example 4: Comparing Different Configurations")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generate test images
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    images = torch.sigmoid(images)
    prompts = ["digit 5", "digit 5", "digit 8", "digit 8"]
    
    configs = ['accuracy', 'balanced', 'quality', 'diverse']
    
    print(f"Test images: {images.shape}")
    print(f"Prompts: {prompts}")
    print()
    
    for config_name in configs:
        reward_fn = get_recommended_reward_config(config_name, device=device)
        rewards = reward_fn(images, prompts)
        
        print(f"{config_name:12s}: mean={rewards.mean().item():.4f}, "
              f"std={rewards.std().item():.4f}, "
              f"min={rewards.min().item():.4f}, "
              f"max={rewards.max().item():.4f}")
    print()


def example_interesting_rewards():
    """Example: Using interesting reward functions."""
    print("=" * 60)
    print("Example 5: Interesting Reward Functions")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Simulate generated images
    batch_size = 4
    images = torch.randn(batch_size, 1, 28, 28)
    images = torch.sigmoid(images)
    prompts = ["digit 0", "digit 8", "digit 3", "digit 1"]  # Some symmetric digits
    
    print("Testing interesting reward functions:\n")
    
    # 1. Geometric Reward
    print("1. Geometric Reward (centroid, aspect ratio, fill ratio)")
    geometric = MNISTGeometricReward(device=device)
    rewards = geometric(images, prompts)
    print(f"   Rewards: {rewards}\n")
    
    # 2. Symmetry Reward
    print("2. Symmetry Reward (for symmetric digits: 0, 1, 3, 8)")
    symmetry = MNISTSymmetryReward(device=device)
    rewards = symmetry(images, prompts)
    print(f"   Rewards: {rewards}\n")
    
    # 3. Feature-based Reward
    print("3. Feature-based Reward (using classifier features)")
    classifier = MNISTClassifierReward(device=device)
    feature_reward = MNISTFeatureReward(classifier)
    rewards = feature_reward(images, prompts)
    print(f"   Rewards: {rewards}\n")
    
    # 4. Combined interesting rewards
    print("4. Combined: Geometric + Symmetry + Classification")
    classifier_reward = MNISTClassifierReward(device=device)
    confidence_reward = MNISTConfidenceReward(classifier_reward)
    
    class InterestingCombined:
        def __init__(self):
            self.classifier = classifier_reward
            self.confidence = confidence_reward
            self.geometric = geometric
            self.symmetry = symmetry
        
        def __call__(self, images, prompts, **kwargs):
            cls_rewards = self.classifier(images, prompts, **kwargs)
            conf_rewards = self.confidence(images, prompts, **kwargs)
            geo_rewards = self.geometric(images, prompts, **kwargs)
            sym_rewards = self.symmetry(images, prompts, **kwargs)
            
            # Weighted combination
            total = (0.4 * cls_rewards + 
                    0.2 * conf_rewards + 
                    0.2 * geo_rewards + 
                    0.2 * sym_rewards)
            return total
    
    combined = InterestingCombined()
    rewards = combined(images, prompts)
    print(f"   Combined rewards: {rewards}\n")


def example_integration_with_trainer():
    """Example: How to integrate with FlowGRPOTrainer."""
    print("=" * 60)
    print("Example 6: Integration with FlowGRPOTrainer")
    print("=" * 60)
    
    print("""
# In your training script (e.g., train_mnist.py):

from rewards.mnist_rewards import get_recommended_reward_config
from training.trainer import FlowGRPOTrainer

# 1. Create reward function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reward_fn = get_recommended_reward_config('balanced', device=device)

# 2. Create trainer with MNIST reward
trainer = FlowGRPOTrainer(
    model=model,
    prompt_encoder=prompt_encoder,
    reward_fn=reward_fn,  # Use MNIST reward here
    train_dataset=train_dataset,  # MNIST dataset with prompts like "digit 0", "digit 1", etc.
    test_dataset=test_dataset,
    config={
        'batch_size': 32,
        'num_samples_per_prompt': 4,  # Generate 4 samples per prompt for GRPO grouping
        'learning_rate': 1e-4,
        'device': device,
        # ... other config ...
    },
)

# 3. Train
trainer.train(num_epochs=100)

# The reward function will be called automatically during training:
# - For each batch, generate images from prompts
# - Compute rewards for each image
# - GRPO groups samples by prompt_id and computes group-relative advantages
# - Policy is updated using these advantages
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MNIST Reward Functions Examples")
    print("=" * 60 + "\n")
    
    # Run examples
    example_single_reward()
    example_combined_reward()
    example_custom_weights()
    example_all_configurations()
    example_interesting_rewards()
    example_yellow_reward()
    example_integration_with_trainer()
    
    print("\n" + "=" * 60)
    print("For more details, see: markdown/MNIST_REWARDS.md")
    print("=" * 60 + "\n")
