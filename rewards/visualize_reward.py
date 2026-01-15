"""
Visualize the shape reward function and target patterns.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from rewards.simple_reward import compute_shape_reward

def visualize_target_patterns():
    """Visualize the target patterns for each shape type."""
    signal_dim = 64
    device = "cpu"
    
    # Create x coordinates
    x = torch.linspace(0, 1, signal_dim, device=device)
    
    # Create target patterns
    shapes = ["circle", "square", "triangle"]
    targets = {}
    
    for shape_type in shapes:
        if shape_type == "circle":
            target = 1.0 - 4 * (x - 0.5) ** 2
            target = torch.clamp(target, 0, 1)
        elif shape_type == "square":
            target = ((x > 0.33) & (x < 0.67)).float()
        elif shape_type == "triangle":
            target = torch.where(x < 0.5, 2 * x, 2 * (1 - x))
        targets[shape_type] = target
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, shape_type in enumerate(shapes):
        target = targets[shape_type]
        axes[idx].plot(x.numpy(), target.numpy(), 'b-', linewidth=3, label='Target')
        axes[idx].set_title(f'{shape_type.capitalize()} Pattern', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Position (normalized)', fontsize=12)
        axes[idx].set_ylabel('Value', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-0.1, 1.1)
        axes[idx].legend()
    
    plt.suptitle('Target Patterns for Shape Rewards', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "target_patterns.png", dpi=150, bbox_inches='tight')
    print(f"Saved target patterns to {output_dir / 'target_patterns.png'}")
    
    return targets

def visualize_reward_computation():
    """Show how rewards are computed for different signals."""
    signal_dim = 64
    device = "cpu"
    
    # Create target patterns
    x = torch.linspace(0, 1, signal_dim, device=device)
    
    circle_target = torch.clamp(1.0 - 4 * (x - 0.5) ** 2, 0, 1)
    square_target = ((x > 0.33) & (x < 0.67)).float()
    triangle_target = torch.where(x < 0.5, 2 * x, 2 * (1 - x))
    
    # Create example signals
    # 1. Perfect match
    perfect_circle = circle_target.clone()
    
    # 2. Close match (with some noise)
    noisy_circle = circle_target + 0.1 * torch.randn(signal_dim)
    noisy_circle = (noisy_circle - noisy_circle.min()) / (noisy_circle.max() - noisy_circle.min() + 1e-8)
    
    # 3. Wrong shape (square instead of circle)
    wrong_shape = square_target.clone()
    
    # 4. Random signal
    random_signal = torch.rand(signal_dim)
    
    signals = {
        "Perfect Circle": perfect_circle,
        "Noisy Circle": noisy_circle,
        "Wrong Shape (Square)": wrong_shape,
        "Random Signal": random_signal,
    }
    
    # Compute rewards
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, signal) in enumerate(signals.items()):
        # Normalize signal
        signal_norm = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        
        # Compute rewards for each shape
        circle_reward = compute_shape_reward(signal_norm.unsqueeze(0), "circle").item()
        square_reward = compute_shape_reward(signal_norm.unsqueeze(0), "square").item()
        triangle_reward = compute_shape_reward(signal_norm.unsqueeze(0), "triangle").item()
        
        # Plot signal
        axes[idx].plot(x.numpy(), signal_norm.numpy(), 'r-', linewidth=2, label='Signal', alpha=0.7)
        axes[idx].plot(x.numpy(), circle_target.numpy(), 'b--', linewidth=1.5, label='Circle target', alpha=0.5)
        axes[idx].plot(x.numpy(), square_target.numpy(), 'g--', linewidth=1.5, label='Square target', alpha=0.5)
        axes[idx].plot(x.numpy(), triangle_target.numpy(), 'm--', linewidth=1.5, label='Triangle target', alpha=0.5)
        
        axes[idx].set_title(f'{name}\nRewards: Circle={circle_reward:.3f}, Square={square_reward:.3f}, Triangle={triangle_reward:.3f}', 
                           fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Position (normalized)', fontsize=10)
        axes[idx].set_ylabel('Value', fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)
        axes[idx].set_ylim(-0.1, 1.1)
    
    plt.suptitle('Reward Computation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "reward_examples.png", dpi=150, bbox_inches='tight')
    print(f"Saved reward examples to {output_dir / 'reward_examples.png'}")

def show_reward_formula():
    """Show the mathematical formula for reward computation."""
    print("\n" + "="*60)
    print("SHAPE REWARD COMPUTATION")
    print("="*60)
    print("\n1. Normalize signal to [0, 1]:")
    print("   signal_norm = (signal - min(signal)) / (max(signal) - min(signal))")
    print("\n2. Create target pattern based on shape type:")
    print("   Circle:   target = clamp(1.0 - 4 * (x - 0.5)², 0, 1)")
    print("   Square:   target = 1 if 0.33 < x < 0.67, else 0")
    print("   Triangle: target = 2*x if x < 0.5, else 2*(1-x)")
    print("\n3. Compute Mean Squared Error (MSE):")
    print("   MSE = mean((signal_norm - target)²)")
    print("\n4. Convert MSE to reward:")
    print("   reward = 1.0 - MSE")
    print("\n   → Higher reward (closer to 1.0) = better match")
    print("   → Lower reward (closer to 0.0) = worse match")
    print("="*60 + "\n")

if __name__ == "__main__":
    print("Visualizing shape reward function...")
    
    # Show formula
    show_reward_formula()
    
    # Visualize target patterns
    print("Creating target patterns visualization...")
    visualize_target_patterns()
    
    # Visualize reward computation
    print("Creating reward computation examples...")
    visualize_reward_computation()
    
    print("\nDone! Check outputs/ for visualizations.")
