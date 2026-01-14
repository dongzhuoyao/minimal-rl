"""
Simple reward function for tutorial.

This reward function evaluates how well a generated 1D signal matches
the prompt (e.g., "red circle" should generate a circular pattern).
"""
import torch
import numpy as np


def compute_shape_reward(signal, shape_type):
    """
    Compute reward based on how well signal matches shape type.
    
    Args:
        signal: Generated 1D signal [batch_size, signal_dim]
        shape_type: String indicating shape ("circle", "square", "triangle")
    
    Returns:
        reward: Scalar reward [batch_size]
    """
    batch_size, signal_dim = signal.shape
    device = signal.device
    
    # Normalize signal to [0, 1]
    signal_norm = (signal - signal.min(dim=-1, keepdim=True)[0]) / (
        signal.max(dim=-1, keepdim=True)[0] - signal.min(dim=-1, keepdim=True)[0] + 1e-8
    )
    
    # Create target patterns
    x = torch.linspace(0, 1, signal_dim, device=device)
    
    if shape_type == "circle":
        # Circular pattern: smooth peak in the middle
        target = 1.0 - 4 * (x - 0.5) ** 2
        target = torch.clamp(target, 0, 1)
    elif shape_type == "square":
        # Square pattern: high in middle third
        target = ((x > 0.33) & (x < 0.67)).float()
    elif shape_type == "triangle":
        # Triangle pattern: linear increase then decrease
        target = torch.where(x < 0.5, 2 * x, 2 * (1 - x))
    else:
        # Default: uniform
        target = torch.ones_like(x) * 0.5
    
    # Compute similarity (negative MSE)
    mse = torch.mean((signal_norm - target.unsqueeze(0)) ** 2, dim=-1)
    reward = 1.0 - mse  # Higher reward for lower error
    
    return reward


def compute_color_reward(signal, color):
    """
    Compute reward based on color (simplified - uses signal statistics).
    
    Args:
        signal: Generated 1D signal [batch_size, signal_dim]
        color: String indicating color ("red", "blue", "green", etc.)
    
    Returns:
        reward: Scalar reward [batch_size]
    """
    # Map color to target mean value
    color_map = {
        "red": 0.8,
        "blue": 0.4,
        "green": 0.6,
        "yellow": 0.7,
        "purple": 0.5,
        "orange": 0.75,
        "pink": 0.65,
        "cyan": 0.45,
        "magenta": 0.55,
    }
    
    target_mean = color_map.get(color.lower(), 0.5)
    
    # Compute mean of signal
    signal_mean = signal.mean(dim=-1)
    
    # Normalize to [0, 1] for comparison
    signal_mean_norm = torch.sigmoid(signal_mean)
    
    # Reward based on how close mean is to target
    reward = 1.0 - torch.abs(signal_mean_norm - target_mean)
    
    return reward


def parse_prompt(prompt):
    """
    Parse prompt into shape and color.
    
    Args:
        prompt: String like "a red circle"
    
    Returns:
        shape: Shape type string
        color: Color string
    """
    prompt_lower = prompt.lower()
    
    # Extract shape
    shapes = ["circle", "square", "triangle"]
    shape = "circle"  # default
    for s in shapes:
        if s in prompt_lower:
            shape = s
            break
    
    # Extract color
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan", "magenta"]
    color = "red"  # default
    for c in colors:
        if c in prompt_lower:
            color = c
            break
    
    return shape, color


def compute_reward(signal, prompt):
    """
    Compute total reward for a generated signal given a prompt.
    
    Args:
        signal: Generated 1D signal [batch_size, signal_dim]
        prompt: Prompt string or list of strings
    
    Returns:
        reward: Scalar reward [batch_size]
    """
    if isinstance(prompt, str):
        prompt = [prompt]
    
    batch_size = signal.shape[0]
    rewards = torch.zeros(batch_size, device=signal.device)
    
    for i, p in enumerate(prompt):
        shape, color = parse_prompt(p)
        
        shape_reward = compute_shape_reward(signal[i:i+1], shape)
        color_reward = compute_color_reward(signal[i:i+1], color)
        
        # Combined reward (weighted sum)
        total_reward = 0.7 * shape_reward + 0.3 * color_reward
        rewards[i] = total_reward.item()
    
    return rewards


class SimpleReward:
    """Reward function wrapper for easy use in training."""
    
    def __init__(self):
        pass
    
    def __call__(self, signals, prompts, **kwargs):
        """
        Compute rewards for a batch of signals.
        
        Args:
            signals: Generated signals [batch_size, signal_dim]
            prompts: List of prompt strings
        
        Returns:
            rewards: Tensor of rewards [batch_size]
        """
        if isinstance(signals, list):
            signals = signals[-1]  # Use final signal
        
        return compute_reward(signals, prompts)
