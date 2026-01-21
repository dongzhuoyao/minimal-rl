"""
Visualize MNIST reward functions and examples.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import MNISTDataset
from rewards.mnist_rewards import (
    MNISTBrightDigitReward,
    MNISTCenteredDigitReward,
    MNISTHighContrastReward,
    MNISTLargeDigitReward,
    MNISTSparseDigitReward,
    MNISTTinyDigitReward,
)

# MNIST constants
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
IMG_SIZE = 28


def denormalize_mnist(image: torch.Tensor) -> torch.Tensor:
    """Denormalize MNIST image from normalized form to [0, 1] range."""
    return torch.clamp(image * MNIST_STD + MNIST_MEAN, 0, 1)


def load_mnist_samples(num_samples: int = 1000):
    """Load and preprocess MNIST samples."""
    dataset = MNISTDataset("dataset", split="test", download=True)
    sample_size = min(num_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size]

    all_images = []
    all_labels = []

    for idx in indices:
        sample = dataset[idx]
        image_flat = sample["image"]
        label = int(sample["label"].item()) if isinstance(sample["label"], torch.Tensor) else int(sample["label"])

        image_2d = denormalize_mnist(image_flat).view(1, IMG_SIZE, IMG_SIZE)
        all_images.append(image_2d)
        all_labels.append(label)

    return all_images, all_labels


def compute_rewards_for_dataset(reward_fn, images, labels):
    """Compute rewards for a list of images."""
    rewards = []
    for img, label in zip(images, labels):
        prompt = f"digit {label}"
        reward = reward_fn(img, [prompt]).item()
        rewards.append(reward)
    return rewards

def visualize_large_reward():
    """Visualize the 'large' reward function with real MNIST data examples."""
    device = "cpu"
    reward_fn = MNISTLargeDigitReward(device=device, threshold=0.1)

    print("Loading MNIST dataset...")
    all_images, all_labels = load_mnist_samples(1000)

    print("Computing rewards for MNIST images...")
    all_rewards = compute_rewards_for_dataset(reward_fn, all_images, all_labels)
    
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


def visualize_centered_reward():
    """Visualize the 'centered' reward function with real MNIST data examples."""
    device = "cpu"
    reward_fn = MNISTCenteredDigitReward(device=device, center_radius=0.3)
    
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
    
    # High reward examples (top 3) - well-centered digits
    for i in range(3):
        idx = sorted_indices[i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]  # Shape: [1, 28, 28]
        label = all_labels[idx]
        # Calculate center vs edge ratio
        center_y, center_x = H // 2, W // 2
        y_coords = torch.arange(H, dtype=torch.float32) - center_y
        x_coords = torch.arange(W, dtype=torch.float32) - center_x
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(X**2 + Y**2)
        max_dist = torch.sqrt(torch.tensor(H**2 + W**2, dtype=torch.float32)) / 2
        center_mask = (distances < max_dist * 0.3).float()
        edge_mask = 1.0 - center_mask
        
        center_density = (image[0] * center_mask).sum().item() / (center_mask.sum().item() + 1e-8)
        edge_density = (image[0] * edge_mask).sum().item() / (edge_mask.sum().item() + 1e-8)
        center_ratio = center_density / (edge_density + center_density + 1e-8)
        
        examples.append((
            f"High Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image,
            "HIGH",
            label,
            reward_val,
            center_ratio
        ))
    
    # Medium reward examples (middle)
    mid_start = len(sorted_indices) // 2 - 1
    for i in range(2):
        idx = sorted_indices[mid_start + i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]  # Shape: [1, 28, 28]
        label = all_labels[idx]
        
        # Calculate center vs edge ratio
        center_y, center_x = H // 2, W // 2
        y_coords = torch.arange(H, dtype=torch.float32) - center_y
        x_coords = torch.arange(W, dtype=torch.float32) - center_x
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(X**2 + Y**2)
        max_dist = torch.sqrt(torch.tensor(H**2 + W**2, dtype=torch.float32)) / 2
        center_mask = (distances < max_dist * 0.3).float()
        edge_mask = 1.0 - center_mask
        
        center_density = (image[0] * center_mask).sum().item() / (center_mask.sum().item() + 1e-8)
        edge_density = (image[0] * edge_mask).sum().item() / (edge_mask.sum().item() + 1e-8)
        center_ratio = center_density / (edge_density + center_density + 1e-8)
        
        examples.append((
            f"Medium Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image,
            "MEDIUM",
            label,
            reward_val,
            center_ratio
        ))
    
    # Low reward examples (bottom 3) - off-center digits
    for i in range(3):
        idx = sorted_indices[-(i+1)].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]  # Shape: [1, 28, 28]
        label = all_labels[idx]
        
        # Calculate center vs edge ratio
        center_y, center_x = H // 2, W // 2
        y_coords = torch.arange(H, dtype=torch.float32) - center_y
        x_coords = torch.arange(W, dtype=torch.float32) - center_x
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(X**2 + Y**2)
        max_dist = torch.sqrt(torch.tensor(H**2 + W**2, dtype=torch.float32)) / 2
        center_mask = (distances < max_dist * 0.3).float()
        edge_mask = 1.0 - center_mask
        
        center_density = (image[0] * center_mask).sum().item() / (center_mask.sum().item() + 1e-8)
        edge_density = (image[0] * edge_mask).sum().item() / (edge_mask.sum().item() + 1e-8)
        center_ratio = center_density / (edge_density + center_density + 1e-8)
        
        examples.append((
            f"Low Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image,
            "LOW",
            label,
            reward_val,
            center_ratio
        ))
    
    # Extract images and compute rewards (for verification)
    all_images_tensor = torch.cat([ex[1] for ex in examples], dim=0)
    prompts = [f"digit {ex[3]}" for ex in examples]
    rewards = reward_fn(all_images_tensor, prompts)
    
    # Create visualization with center region overlay
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Create center mask for visualization
    center_y, center_x = H // 2, W // 2
    y_coords = torch.arange(H, dtype=torch.float32) - center_y
    x_coords = torch.arange(W, dtype=torch.float32) - center_x
    Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
    distances = torch.sqrt(X**2 + Y**2)
    max_dist = torch.sqrt(torch.tensor(H**2 + W**2, dtype=torch.float32)) / 2
    center_mask_viz = (distances < max_dist * 0.3).float().numpy()
    
    for idx, (name, img, category, label, reward_val, center_ratio) in enumerate(examples):
        ax = axes[idx]
        
        # Display image - img is [1, 28, 28], so we take img[0] to get [28, 28]
        img_np = img[0].numpy()  # Shape: [28, 28]
        im = ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        
        # Overlay center region mask (semi-transparent)
        center_overlay = np.ma.masked_where(center_mask_viz < 0.5, center_mask_viz)
        ax.imshow(center_overlay, cmap='Reds', alpha=0.3, vmin=0, vmax=1)
        
        # Color code based on reward
        if reward_val > 0.6:
            color = 'green'
        elif reward_val > 0.4:
            color = 'orange'
        else:
            color = 'red'
        
        # Title with reward info
        title = f"{name}\nCenter Ratio: {center_ratio:.2f}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel(f'Label: {label}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border with reward-based color
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('MNIST Centered Digit Reward Visualization (Real Data)\n(Higher reward = More centered, red overlay shows center region)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "centered_reward_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved centered reward visualization to {output_dir / 'centered_reward_visualization.png'}")
    
    # Print summary
    print("\n" + "="*60)
    print("CENTERED REWARD SUMMARY (Real MNIST Data)")
    print("="*60)
    print("\nReward Formula:")
    print("  1. Define center region (radius = 30% of image diagonal)")
    print("  2. Compute center_density = mean pixel value in center")
    print("  3. Compute edge_density = mean pixel value in edges")
    print("  4. Reward = center_density / (edge_density + center_density)")
    print("\nExamples from MNIST test set:")
    print(f"  {'Example':<30s} | {'Reward':<8s} | {'Center Ratio':<15s} | {'Label'}")
    print("  " + "-" * 70)
    for name, _, category, label, reward_val, center_ratio in examples:
        print(f"  {name.split(chr(10))[0]:<30s} | {reward_val:<8.3f} | {center_ratio:<15.2f} | {label}")
    print("="*60 + "\n")
    
    return examples, rewards


def visualize_bright_reward():
    """Visualize the 'bright' reward function with real MNIST data examples."""
    device = "cpu"
    reward_fn = MNISTBrightDigitReward(device=device)
    
    print("Loading MNIST dataset...")
    all_images, all_labels = load_mnist_samples(1000)
    
    print("Computing rewards for MNIST images...")
    all_rewards = []
    for image, label in zip(all_images, all_labels):
        prompt = f"digit {label}"
        reward = reward_fn(image, [prompt]).item()
        all_rewards.append(reward)
    
    all_rewards_tensor = torch.tensor(all_rewards)
    sorted_indices = torch.argsort(all_rewards_tensor, descending=True)
    
    examples = []
    H, W = 28, 28
    
    # High reward (bright)
    for i in range(3):
        idx = sorted_indices[i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        brightness = image[0].mean().item()
        examples.append((
            f"High Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "HIGH", label, reward_val, brightness
        ))
    
    # Medium reward
    mid_start = len(sorted_indices) // 2 - 1
    for i in range(2):
        idx = sorted_indices[mid_start + i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        brightness = image[0].mean().item()
        examples.append((
            f"Medium Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "MEDIUM", label, reward_val, brightness
        ))
    
    # Low reward (dark)
    for i in range(3):
        idx = sorted_indices[-(i+1)].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        brightness = image[0].mean().item()
        examples.append((
            f"Low Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "LOW", label, reward_val, brightness
        ))
    
    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (name, img, category, label, reward_val, brightness) in enumerate(examples):
        ax = axes[idx]
        img_np = img[0].numpy()
        im = ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        
        color = 'green' if reward_val > 0.5 else 'orange' if reward_val > 0.3 else 'red'
        title = f"{name}\nBrightness: {brightness:.3f}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel(f'Label: {label}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('MNIST Bright Digit Reward Visualization (Real Data)\n(Higher reward = Brighter images)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "bright_reward_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved bright reward visualization to {output_dir / 'bright_reward_visualization.png'}")
    
    print("\n" + "="*60)
    print("BRIGHT REWARD SUMMARY")
    print("="*60)
    print("\nReward Formula: reward = mean_brightness (clamped to [0, 1])")
    print("\nExamples:")
    print(f"  {'Example':<30s} | {'Reward':<8s} | {'Brightness':<12s} | {'Label'}")
    print("  " + "-" * 70)
    for name, _, _, label, reward_val, brightness in examples:
        print(f"  {name.split(chr(10))[0]:<30s} | {reward_val:<8.3f} | {brightness:<12.3f} | {label}")
    print("="*60 + "\n")
    
    return examples


def visualize_sparse_reward():
    """Visualize the 'sparse' reward function with real MNIST data examples."""
    device = "cpu"
    reward_fn = MNISTSparseDigitReward(device=device, sparsity_target=0.1, threshold=0.1)
    
    print("Loading MNIST dataset...")
    all_images, all_labels = load_mnist_samples(1000)
    
    print("Computing rewards for MNIST images...")
    all_rewards = []
    for image, label in zip(all_images, all_labels):
        prompt = f"digit {label}"
        reward = reward_fn(image, [prompt]).item()
        all_rewards.append(reward)
    
    all_rewards_tensor = torch.tensor(all_rewards)
    sorted_indices = torch.argsort(all_rewards_tensor, descending=True)
    
    examples = []
    H, W = 28, 28
    
    # High reward (sparse/thin)
    for i in range(3):
        idx = sorted_indices[i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        active_ratio = (image[0] > 0.1).sum().item() / (H * W)
        sparsity = 1.0 - active_ratio
        examples.append((
            f"High Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "HIGH", label, reward_val, active_ratio, sparsity
        ))
    
    # Medium reward
    mid_start = len(sorted_indices) // 2 - 1
    for i in range(2):
        idx = sorted_indices[mid_start + i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        active_ratio = (image[0] > 0.1).sum().item() / (H * W)
        sparsity = 1.0 - active_ratio
        examples.append((
            f"Medium Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "MEDIUM", label, reward_val, active_ratio, sparsity
        ))
    
    # Low reward (dense/thick)
    for i in range(3):
        idx = sorted_indices[-(i+1)].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        active_ratio = (image[0] > 0.1).sum().item() / (H * W)
        sparsity = 1.0 - active_ratio
        examples.append((
            f"Low Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "LOW", label, reward_val, active_ratio, sparsity
        ))
    
    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (name, img, category, label, reward_val, active_ratio, sparsity) in enumerate(examples):
        ax = axes[idx]
        img_np = img[0].numpy()
        im = ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        
        color = 'green' if reward_val > 0.7 else 'orange' if reward_val > 0.4 else 'red'
        title = f"{name}\nActive: {active_ratio:.1%}, Sparsity: {sparsity:.1%}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel(f'Label: {label}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('MNIST Sparse Digit Reward Visualization (Real Data)\n(Higher reward = Sparse/thin digits, closer to target)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "sparse_reward_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved sparse reward visualization to {output_dir / 'sparse_reward_visualization.png'}")
    
    print("\n" + "="*60)
    print("SPARSE REWARD SUMMARY")
    print("="*60)
    print("\nReward Formula:")
    print("  1. Compute sparsity = 1 - (active_pixels / total_pixels)")
    print("  2. Reward = 1 - |sparsity - target| / target")
    print("  3. Higher reward = closer to target sparsity (default: 0.1)")
    print("\nExamples:")
    print(f"  {'Example':<30s} | {'Reward':<8s} | {'Active':<10s} | {'Sparsity':<10s} | {'Label'}")
    print("  " + "-" * 80)
    for name, _, _, label, reward_val, active_ratio, sparsity in examples:
        print(f"  {name.split(chr(10))[0]:<30s} | {reward_val:<8.3f} | {active_ratio:<10.1%} | {sparsity:<10.1%} | {label}")
    print("="*60 + "\n")
    
    return examples


def visualize_contrast_reward():
    """Visualize the 'contrast' reward function with real MNIST data examples."""
    device = "cpu"
    reward_fn = MNISTHighContrastReward(device=device)
    
    print("Loading MNIST dataset...")
    all_images, all_labels = load_mnist_samples(1000)
    
    print("Computing rewards for MNIST images...")
    all_rewards = []
    for image, label in zip(all_images, all_labels):
        prompt = f"digit {label}"
        reward = reward_fn(image, [prompt]).item()
        all_rewards.append(reward)
    
    all_rewards_tensor = torch.tensor(all_rewards)
    sorted_indices = torch.argsort(all_rewards_tensor, descending=True)
    
    examples = []
    H, W = 28, 28
    
    # High reward (high contrast)
    for i in range(3):
        idx = sorted_indices[i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        contrast = image[0].std().item()
        examples.append((
            f"High Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "HIGH", label, reward_val, contrast
        ))
    
    # Medium reward
    mid_start = len(sorted_indices) // 2 - 1
    for i in range(2):
        idx = sorted_indices[mid_start + i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        contrast = image[0].std().item()
        examples.append((
            f"Medium Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "MEDIUM", label, reward_val, contrast
        ))
    
    # Low reward (low contrast)
    for i in range(3):
        idx = sorted_indices[-(i+1)].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        contrast = image[0].std().item()
        examples.append((
            f"Low Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "LOW", label, reward_val, contrast
        ))
    
    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (name, img, category, label, reward_val, contrast) in enumerate(examples):
        ax = axes[idx]
        img_np = img[0].numpy()
        im = ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        
        color = 'green' if reward_val > 0.5 else 'orange' if reward_val > 0.3 else 'red'
        title = f"{name}\nContrast (std): {contrast:.3f}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel(f'Label: {label}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('MNIST High Contrast Reward Visualization (Real Data)\n(Higher reward = Higher contrast/sharpness)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "contrast_reward_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved contrast reward visualization to {output_dir / 'contrast_reward_visualization.png'}")
    
    print("\n" + "="*60)
    print("CONTRAST REWARD SUMMARY")
    print("="*60)
    print("\nReward Formula: reward = std(pixels) / 0.5 (clamped to [0, 1])")
    print("\nExamples:")
    print(f"  {'Example':<30s} | {'Reward':<8s} | {'Contrast (std)':<15s} | {'Label'}")
    print("  " + "-" * 70)
    for name, _, _, label, reward_val, contrast in examples:
        print(f"  {name.split(chr(10))[0]:<30s} | {reward_val:<8.3f} | {contrast:<15.3f} | {label}")
    print("="*60 + "\n")
    
    return examples


def visualize_tiny_reward():
    """Visualize the 'tiny' reward function with real MNIST data examples."""
    device = "cpu"
    reward_fn = MNISTTinyDigitReward(device=device, threshold=0.1)
    
    print("Loading MNIST dataset...")
    all_images, all_labels = load_mnist_samples(1000)
    
    print("Computing rewards for MNIST images...")
    all_rewards = []
    for image, label in zip(all_images, all_labels):
        prompt = f"digit {label}"
        reward = reward_fn(image, [prompt]).item()
        all_rewards.append(reward)
    
    all_rewards_tensor = torch.tensor(all_rewards)
    sorted_indices = torch.argsort(all_rewards_tensor, descending=True)
    
    examples = []
    H, W = 28, 28
    
    # High reward (tiny/small bounding box)
    for i in range(3):
        idx = sorted_indices[i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        # Calculate bounding box
        active_mask = (image[0] > 0.1).float()
        active_y, active_x = torch.where(active_mask > 0)
        if len(active_y) > 0:
            bbox_area = ((active_y.max() - active_y.min() + 1) * (active_x.max() - active_x.min() + 1)).item()
            bbox_ratio = bbox_area / (H * W)
        else:
            bbox_ratio = 0.0
        examples.append((
            f"High Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "HIGH", label, reward_val, bbox_ratio
        ))
    
    # Medium reward
    mid_start = len(sorted_indices) // 2 - 1
    for i in range(2):
        idx = sorted_indices[mid_start + i].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        active_mask = (image[0] > 0.1).float()
        active_y, active_x = torch.where(active_mask > 0)
        if len(active_y) > 0:
            bbox_area = ((active_y.max() - active_y.min() + 1) * (active_x.max() - active_x.min() + 1)).item()
            bbox_ratio = bbox_area / (H * W)
        else:
            bbox_ratio = 0.0
        examples.append((
            f"Medium Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "MEDIUM", label, reward_val, bbox_ratio
        ))
    
    # Low reward (large bounding box)
    for i in range(3):
        idx = sorted_indices[-(i+1)].item()
        reward_val = all_rewards[idx]
        image = all_images[idx]
        label = all_labels[idx]
        active_mask = (image[0] > 0.1).float()
        active_y, active_x = torch.where(active_mask > 0)
        if len(active_y) > 0:
            bbox_area = ((active_y.max() - active_y.min() + 1) * (active_x.max() - active_x.min() + 1)).item()
            bbox_ratio = bbox_area / (H * W)
        else:
            bbox_ratio = 0.0
        examples.append((
            f"Low Reward #{i+1}\n(Digit {label}, Reward: {reward_val:.3f})",
            image, "LOW", label, reward_val, bbox_ratio
        ))
    
    # Visualize with bounding box overlay
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (name, img, category, label, reward_val, bbox_ratio) in enumerate(examples):
        ax = axes[idx]
        img_np = img[0].numpy()
        im = ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        
        # Draw bounding box
        active_mask = (img[0] > 0.1).float()
        active_y, active_x = torch.where(active_mask > 0)
        if len(active_y) > 0:
            min_y, max_y = active_y.min().item(), active_y.max().item()
            min_x, max_x = active_x.min().item(), active_x.max().item()
            # Draw rectangle
            from matplotlib.patches import Rectangle
            rect = Rectangle((min_x-0.5, min_y-0.5), max_x-min_x+1, max_y-min_y+1, 
                           linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        color = 'green' if reward_val > 0.7 else 'orange' if reward_val > 0.4 else 'red'
        title = f"{name}\nBBox Ratio: {bbox_ratio:.1%}"
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel(f'Label: {label}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('MNIST Tiny Digit Reward Visualization (Real Data)\n(Higher reward = Smaller bounding box, red box shows bbox)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "tiny_reward_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved tiny reward visualization to {output_dir / 'tiny_reward_visualization.png'}")
    
    print("\n" + "="*60)
    print("TINY REWARD SUMMARY")
    print("="*60)
    print("\nReward Formula:")
    print("  1. Find bounding box of active pixels")
    print("  2. Compute bbox_ratio = bbox_area / image_area")
    print("  3. Reward = 1 - bbox_ratio (smaller bbox = higher reward)")
    print("\nExamples:")
    print(f"  {'Example':<30s} | {'Reward':<8s} | {'BBox Ratio':<12s} | {'Label'}")
    print("  " + "-" * 70)
    for name, _, _, label, reward_val, bbox_ratio in examples:
        print(f"  {name.split(chr(10))[0]:<30s} | {reward_val:<8.3f} | {bbox_ratio:<12.1%} | {label}")
    print("="*60 + "\n")
    
    return examples


if __name__ == "__main__":
    print("Visualizing all MNIST reward functions...")
    
    # Visualize all rewards
    print("\n" + "="*60)
    print("1. Large Reward")
    print("="*60)
    visualize_large_reward()
    
    print("\n" + "="*60)
    print("2. Centered Reward")
    print("="*60)
    visualize_centered_reward()
    
    print("\n" + "="*60)
    print("3. Bright Reward")
    print("="*60)
    visualize_bright_reward()
    
    print("\n" + "="*60)
    print("4. Sparse Reward")
    print("="*60)
    visualize_sparse_reward()
    
    print("\n" + "="*60)
    print("5. Contrast Reward")
    print("="*60)
    visualize_contrast_reward()
    
    print("\n" + "="*60)
    print("6. Tiny Reward")
    print("="*60)
    visualize_tiny_reward()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - outputs/large_reward_visualization.png")
    print("  - outputs/centered_reward_visualization.png")
    print("  - outputs/bright_reward_visualization.png")
    print("  - outputs/sparse_reward_visualization.png")
    print("  - outputs/contrast_reward_visualization.png")
    print("  - outputs/tiny_reward_visualization.png")
    print("="*60 + "\n")
