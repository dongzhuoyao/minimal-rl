"""
Visualize MNIST reward functions and examples.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from rewards.mnist_rewards import MNISTLargeDigitReward
from dataset import MNISTDataset

def visualize_large_reward():
    """Visualize the 'large' reward function with real MNIST data examples."""
    device = "cpu"
    reward_fn = MNISTLargeDigitReward(device=device, threshold=0.1)
    
    # Load real MNIST dataset
    print("Loading MNIST dataset...")
    dataset = MNISTDataset("dataset", split="test", download=True)
    
    # Collect images and compute rewards
    print("Computing rewards for MNIST images...")
    H, W = 28, 28
    all_rewards = []
    all_images = []
    all_labels = []
    
    # Sample a subset of images to find diverse examples
    sample_size = min(1000, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size]
    
    for idx in indices:
        sample = dataset[idx]
        # Get image and denormalize from MNIST normalization
        image_flat = sample["image"]  # [784] - normalized
        label_raw = sample["label"]
        
        # Convert label to int if it's a tensor
        if isinstance(label_raw, torch.Tensor):
            label = int(label_raw.item())
        else:
            label = int(label_raw)
        
        # Denormalize: (x - 0.1307) / 0.3081 -> x
        # So: x = image * 0.3081 + 0.1307
        image_denorm = image_flat * 0.3081 + 0.1307
        image_2d = image_denorm.view(1, H, W)  # [1, 28, 28]
        
        # Clamp to [0, 1] and ensure it's in the right format
        image_2d = torch.clamp(image_2d, 0, 1)
        
        # Compute reward
        prompt = f"digit {label}"
        reward = reward_fn(image_2d, [prompt]).item()
        
        all_rewards.append(reward)
        all_images.append(image_2d)
        all_labels.append(label)
    
    # Find examples with high and low rewards
    all_rewards_tensor = torch.tensor(all_rewards)
    
    # Sort by reward
    sorted_indices = torch.argsort(all_rewards_tensor, descending=True)
    
    # Select diverse examples
    examples = []
    
    # High reward examples (top 3)
    for i in range(3):
        idx = sorted_indices[i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]  # Shape: [1, 28, 28]
        label = all_labels[idx]
        # Calculate active ratio from the 2D image [28, 28]
        active_ratio = (image[0] > 0.1).sum().item() / (H * W)
        examples.append((
            f"High Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image,
            "HIGH",
            label,
            reward_val,
            active_ratio
        ))
    
    # Medium reward examples (middle)
    mid_start = len(sorted_indices) // 2 - 1
    for i in range(2):
        idx = sorted_indices[mid_start + i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]  # Shape: [1, 28, 28]
        label = all_labels[idx]
        active_ratio = (image[0] > 0.1).sum().item() / (H * W)
        examples.append((
            f"Medium Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image,
            "MEDIUM",
            label,
            reward_val,
            active_ratio
        ))
    
    # Low reward examples (bottom 3)
    for i in range(3):
        idx = sorted_indices[-(i+1)].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]  # Shape: [1, 28, 28]
        label = all_labels[idx]
        active_ratio = (image[0] > 0.1).sum().item() / (H * W)
        examples.append((
            f"Low Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image,
            "LOW",
            label,
            reward_val,
            active_ratio
        ))
    
    # Extract images and compute rewards (for verification)
    all_images_tensor = torch.cat([ex[1] for ex in examples], dim=0)
    prompts = [f"digit {ex[3]}" for ex in examples]
    rewards = reward_fn(all_images_tensor, prompts)
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (name, img, category, label, reward_val, active_ratio) in enumerate(examples):
        ax = axes[idx]
        
        # Display image - img is [1, 28, 28], so we take img[0] to get [28, 28]
        img_np = img[0].numpy()  # Shape: [28, 28]
        im = ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        
        # Color code based on reward
        if reward_val > 0.5:
            color = 'green'
        elif reward_val > 0.2:
            color = 'orange'
        else:
            color = 'red'
        
        # Title with reward info
        title = f"{name}\nActive Pixels: {active_ratio:.1%}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel(f'Label: {label}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border with reward-based color
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('MNIST Large Digit Reward Visualization (Real Data)\n(Higher reward = More active pixels)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "large_reward_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved large reward visualization to {output_dir / 'large_reward_visualization.png'}")
    
    # Print summary
    print("\n" + "="*60)
    print("LARGE REWARD SUMMARY (Real MNIST Data)")
    print("="*60)
    print("\nReward Formula:")
    print("  1. Count pixels above threshold (default: 0.1)")
    print("  2. Compute fraction: active_pixels / total_pixels")
    print("  3. Reward = fraction (clamped to [0, 1])")
    print("\nExamples from MNIST test set:")
    print(f"  {'Example':<30s} | {'Reward':<8s} | {'Active Pixels':<15s} | {'Label'}")
    print("  " + "-" * 70)
    for name, _, category, label, reward_val, active_ratio in examples:
        print(f"  {name.split(chr(10))[0]:<30s} | {reward_val:<8.3f} | {active_ratio:<15.1%} | {label}")
    print("="*60 + "\n")
    
    return examples, rewards


if __name__ == "__main__":
    print("Visualizing MNIST reward functions...")
    
    # Visualize large reward
    print("\nCreating large reward visualization...")
    visualize_large_reward()
    
    print("\nDone! Check outputs/ for visualizations.")
