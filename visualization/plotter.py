"""
Visualization utilities for training progress.
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch


class TrainingPlotter:
    """Plot training progress and generated samples."""
    
    def __init__(self, output_dir="outputs"):
        """
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # History for plotting
        self.history = {
            "train_loss": [],
            "train_reward": [],
            "test_reward": [],
            "epochs": [],
        }
    
    def update(self, epoch, train_stats, eval_stats, signals, prompts):
        """
        Update plots with new training data.
        
        Args:
            epoch: Current epoch
            train_stats: Training statistics dict
            eval_stats: Evaluation statistics dict
            signals: Generated signals to visualize [num_samples, signal_dim]
            prompts: Corresponding prompts
        """
        # Update history
        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(train_stats["loss"])
        self.history["train_reward"].append(train_stats["reward"])
        self.history["test_reward"].append(eval_stats["mean_reward"])
        
        # Plot training curves
        self._plot_training_curves()
        
        # Plot samples
        self._plot_samples(signals, prompts, epoch, eval_stats)
    
    def _plot_training_curves(self):
        """Plot training loss and reward curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = self.history["epochs"]
        
        # Loss curve
        axes[0].plot(epochs, self.history["train_loss"], "b-", label="Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True)
        
        # Reward curves
        axes[1].plot(epochs, self.history["train_reward"], "g-", label="Train Reward")
        axes[1].plot(epochs, self.history["test_reward"], "r-", label="Test Reward")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Reward")
        axes[1].set_title("Rewards")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=150)
        plt.close()
    
    def _plot_samples(self, signals, prompts, epoch, eval_stats=None):
        """Plot generated signal samples (MNIST images)."""
        if isinstance(signals, torch.Tensor):
            signals = signals.cpu().numpy()
        
        # Handle both 1D (flattened) and 2D (image) signals
        if signals.ndim == 2:
            # Check if it's flattened MNIST (784) or 2D images
            if signals.shape[1] == 784:
                # Reshape from [batch, 784] to [batch, 28, 28]
                signals = signals.reshape(-1, 28, 28)
            elif signals.shape[1] != 28:
                # Assume it's 1D signal, plot as line graph
                num_samples = min(len(signals), 8)
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                axes = axes.flatten()
                
                rewards_list = eval_stats.get("rewards", []) if eval_stats else []
                
                for i in range(num_samples):
                    signal = signals[i]
                    prompt = prompts[i] if i < len(prompts) else f"Sample {i}"
                    
                    axes[i].plot(signal, "b-", linewidth=2)
                    if i < len(rewards_list):
                        axes[i].set_title(f"{prompt}\nReward: {rewards_list[i]:.3f}")
                    else:
                        axes[i].set_title(prompt)
                    axes[i].set_xlabel("Position")
                    axes[i].set_ylabel("Value")
                    axes[i].grid(True, alpha=0.3)
                
                # Hide unused subplots
                for i in range(num_samples, len(axes)):
                    axes[i].axis("off")
                
                plt.suptitle(f"Generated Samples - Epoch {epoch}", fontsize=16)
                plt.tight_layout()
                plt.savefig(self.output_dir / f"samples_epoch_{epoch}.png", dpi=150)
                plt.close()
                return
        
        # Plot as 2D images (MNIST)
        num_samples = min(len(signals), 8)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        rewards_list = eval_stats.get("rewards", []) if eval_stats else []
        
        for i in range(num_samples):
            image = signals[i]
            prompt = prompts[i] if i < len(prompts) else f"Sample {i}"
            
            # Denormalize if needed (MNIST normalization: mean=0.1307, std=0.3081)
            # For visualization, we'll use sigmoid to map to [0, 1]
            if image.min() < 0 or image.max() > 1:
                image = 1 / (1 + np.exp(-image))  # Sigmoid to [0, 1]
            
            axes[i].imshow(image, cmap="gray", vmin=0, vmax=1)
            if i < len(rewards_list):
                axes[i].set_title(f"Label: {prompt}\nReward: {rewards_list[i]:.3f}")
            else:
                axes[i].set_title(f"Label: {prompt}")
            axes[i].axis("off")
        
        # Hide unused subplots
        for i in range(num_samples, len(axes)):
            axes[i].axis("off")
        
        plt.suptitle(f"Generated MNIST Samples - Epoch {epoch}", fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"samples_epoch_{epoch}.png", dpi=150)
        plt.close()
