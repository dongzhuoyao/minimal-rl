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

def visualize_large_reward():
    """Visualize the 'large' reward function with high and low reward examples."""
    device = "cpu"
    reward_fn = MNISTLargeDigitReward(device=device, threshold=0.1)
    
    # Create example images (28x28 MNIST size)
    H, W = 28, 28
    
    # High reward examples: Large, bold digits with many active pixels
    examples = []
    
    # 1. Very large digit (fills most of image) - HIGH REWARD
    large_digit = torch.zeros(1, 1, H, W)
    # Create a large filled circle/digit shape
    center_y, center_x = H // 2, W // 2
    radius = 12
    y_coords = torch.arange(H, dtype=torch.float32).unsqueeze(1) - center_y
    x_coords = torch.arange(W, dtype=torch.float32).unsqueeze(0) - center_x
    distances = torch.sqrt(y_coords**2 + x_coords**2)
    large_digit[0, 0] = (distances < radius).float() * 0.9
    examples.append(("Very Large Digit\n(Fills ~60% of image)", large_digit, "HIGH"))
    
    # 2. Bold thick digit - HIGH REWARD
    bold_digit = torch.zeros(1, 1, H, W)
    # Create a thick vertical bar
    bold_digit[0, 0, 4:24, 10:18] = 0.9
    examples.append(("Bold Thick Digit\n(Thick strokes)", bold_digit, "HIGH"))
    
    # 3. Medium-sized digit - MEDIUM REWARD
    medium_digit = torch.zeros(1, 1, H, W)
    radius = 8
    distances = torch.sqrt(y_coords**2 + x_coords**2)
    medium_digit[0, 0] = (distances < radius).float() * 0.9
    examples.append(("Medium Digit\n(Fills ~30% of image)", medium_digit, "MEDIUM"))
    
    # 4. Small sparse digit - LOW REWARD
    small_digit = torch.zeros(1, 1, H, W)
    radius = 5
    distances = torch.sqrt(y_coords**2 + x_coords**2)
    small_digit[0, 0] = (distances < radius).float() * 0.9
    examples.append(("Small Digit\n(Fills ~10% of image)", small_digit, "LOW"))
    
    # 5. Very thin digit - LOW REWARD
    thin_digit = torch.zeros(1, 1, H, W)
    # Create a thin vertical line
    thin_digit[0, 0, 6:22, 13:15] = 0.9
    examples.append(("Thin Digit\n(Very sparse)", thin_digit, "LOW"))
    
    # 6. Almost empty - VERY LOW REWARD
    empty_digit = torch.zeros(1, 1, H, W)
    # Just a few pixels
    empty_digit[0, 0, 14, 14] = 0.9
    empty_digit[0, 0, 13, 14] = 0.9
    empty_digit[0, 0, 15, 14] = 0.9
    examples.append(("Almost Empty\n(Very few pixels)", empty_digit, "VERY LOW"))
    
    # Compute rewards for all examples
    all_images = torch.cat([ex[1] for ex in examples], dim=0)
    prompts = ["digit 0"] * len(examples)
    rewards = reward_fn(all_images, prompts)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, ((name, img, category), reward_val) in enumerate(zip(examples, rewards)):
        ax = axes[idx]
        
        # Display image
        img_np = img[0, 0].numpy()
        im = ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        
        # Calculate active pixel ratio for display
        active_ratio = (img_np > 0.1).sum() / (H * W)
        
        # Color code based on reward
        if reward_val > 0.5:
            color = 'green'
        elif reward_val > 0.2:
            color = 'orange'
        else:
            color = 'red'
        
        # Title with reward info
        title = f"{name}\nReward: {reward_val:.3f} | Active Pixels: {active_ratio:.1%}"
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.set_xlabel('Width (pixels)', fontsize=9)
        ax.set_ylabel('Height (pixels)', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border with reward-based color
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('MNIST Large Digit Reward Visualization\n(Higher reward = More active pixels)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "large_reward_visualization.png", dpi=150, bbox_inches='tight')
    print(f"Saved large reward visualization to {output_dir / 'large_reward_visualization.png'}")
    
    # Print summary
    print("\n" + "="*60)
    print("LARGE REWARD SUMMARY")
    print("="*60)
    print("\nReward Formula:")
    print("  1. Count pixels above threshold (default: 0.1)")
    print("  2. Compute fraction: active_pixels / total_pixels")
    print("  3. Reward = fraction (clamped to [0, 1])")
    print("\nExamples:")
    for (name, _, category), reward_val in zip(examples, rewards):
        active_ratio = (examples[examples.index((name, _, category))][1][0, 0] > 0.1).sum().item() / (H * W)
        print(f"  {name:30s} | Reward: {reward_val:.3f} | Active: {active_ratio:.1%}")
    print("="*60 + "\n")
    
    return examples, rewards


if __name__ == "__main__":
    print("Visualizing MNIST reward functions...")
    
    # Visualize large reward
    print("\nCreating large reward visualization...")
    visualize_large_reward()
    
    print("\nDone! Check outputs/ for visualizations.")
