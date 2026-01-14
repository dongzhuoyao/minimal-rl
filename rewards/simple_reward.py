"""
Simple reward function for MNIST digit generation.

This reward function evaluates how well a generated image matches the target digit.
Uses a simple classifier-based reward.
"""
import torch
import torch.nn as nn
import numpy as np


class SimpleDigitClassifier(nn.Module):
    """Simple CNN classifier for MNIST digits."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape from [batch, 784] to [batch, 1, 28, 28]
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_digit_reward(signal, target_label):
    """
    Compute reward based on how well generated image matches target digit.
    
    Args:
        signal: Generated image [batch_size, 784] (flattened 28x28)
        target_label: Target digit label [batch_size] (0-9)
    
    Returns:
        reward: Scalar reward [batch_size]
    """
    batch_size = signal.shape[0]
    device = signal.device
    
    # Normalize signal to reasonable range (MNIST normalization)
    # Denormalize from model output to [0, 1] range
    signal_normalized = torch.sigmoid(signal)  # Map to [0, 1]
    
    # Create a simple classifier (or use pre-trained)
    # For simplicity, we'll use a reward based on:
    # 1. Image quality (pixel values in reasonable range)
    # 2. Basic structure (non-zero pixels, reasonable distribution)
    
    # Reward 1: Image quality - penalize extreme values
    pixel_range_penalty = torch.mean((signal_normalized - 0.5) ** 2, dim=-1)
    quality_reward = 1.0 - pixel_range_penalty * 2.0  # Scale to reasonable range
    
    # Reward 2: Structure - reward images with reasonable variance
    pixel_std = torch.std(signal_normalized, dim=-1)
    structure_reward = 1.0 - torch.abs(pixel_std - 0.2)  # Target std around 0.2
    
    # Reward 3: Basic digit-like structure (center should have more activity)
    # Reshape to 28x28
    images = signal_normalized.view(batch_size, 28, 28)
    center_region = images[:, 10:18, 10:18]  # Center 8x8 region
    center_activity = center_region.mean(dim=(1, 2))
    edge_region = torch.cat([
        images[:, :5, :].mean(dim=(1, 2)),
        images[:, -5:, :].mean(dim=(1, 2)),
        images[:, :, :5].mean(dim=(1, 2)),
        images[:, :, -5:].mean(dim=(1, 2))
    ], dim=0).view(4, batch_size).mean(dim=0)
    
    # Reward higher center activity relative to edges
    structure_reward2 = center_activity - edge_region * 0.5
    
    # Combined reward
    total_reward = 0.3 * quality_reward + 0.3 * structure_reward + 0.4 * structure_reward2
    
    # Clamp to [0, 1] range
    total_reward = torch.clamp(total_reward, 0, 1)
    
    return total_reward


def compute_reward(signal, labels):
    """
    Compute reward for generated MNIST images.
    
    Args:
        signal: Generated images [batch_size, 784]
        labels: Target digit labels [batch_size] (as integers or strings)
    
    Returns:
        reward: Tensor of rewards [batch_size]
    """
    batch_size = signal.shape[0]
    device = signal.device
    
    # Convert labels to integers if they're strings
    if isinstance(labels, (list, tuple)):
        if isinstance(labels[0], str):
            labels = torch.tensor([int(l) for l in labels], device=device)
        else:
            labels = torch.tensor(labels, device=device)
    elif isinstance(labels, torch.Tensor):
        if labels.dtype == torch.long or labels.dtype == torch.int:
            pass  # Already integers
        else:
            labels = labels.long()
    else:
        labels = torch.tensor([int(labels)], device=device)
    
    # Ensure labels are in valid range
    labels = labels.clamp(0, 9)
    
    return compute_digit_reward(signal, labels)


class SimpleReward:
    """Reward function wrapper for easy use in training."""
    
    def __init__(self):
        pass
    
    def __call__(self, signals, labels, **kwargs):
        """
        Compute rewards for a batch of signals.
        
        Args:
            signals: Generated signals [batch_size, 784] or list of trajectories
            labels: Target digit labels [batch_size] (as integers or strings)
        
        Returns:
            rewards: Tensor of rewards [batch_size]
        """
        if isinstance(signals, list):
            signals = signals[-1]  # Use final signal
        
        return compute_reward(signals, labels)
