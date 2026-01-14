"""
A simplified 1D flow matching model for tutorial purposes.

This model generates simple 1D signals (like shapes) that can be easily visualized.
Instead of generating images, we generate 1D arrays representing shapes.
"""
import torch
import torch.nn as nn
import numpy as np


class SimpleFlowModel(nn.Module):
    """
    A simple flow matching model that generates 1D signals.
    
    The model takes a prompt embedding and generates a 1D signal
    representing a shape (circle, square, triangle) with a color.
    """
    
    def __init__(self, signal_dim=64, prompt_dim=32, hidden_dim=128):
        """
        Args:
            signal_dim: Dimension of the output signal (length of 1D array)
            prompt_dim: Dimension of prompt embedding
            hidden_dim: Hidden dimension of the network
        """
        super().__init__()
        self.signal_dim = signal_dim
        self.prompt_dim = prompt_dim
        
        # Simple MLP that maps (noise, time, prompt) -> velocity
        self.network = nn.Sequential(
            nn.Linear(signal_dim + 1 + prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, signal_dim),
        )
    
    def forward(self, x, t, prompt_embed):
        """
        Compute velocity field v_theta(x_t, t, prompt).
        
        Args:
            x: Current state [batch_size, signal_dim]
            t: Time step [batch_size, 1] in [0, 1]
            prompt_embed: Prompt embedding [batch_size, prompt_dim]
        
        Returns:
            velocity: Velocity field [batch_size, signal_dim]
        """
        # Concatenate inputs
        inputs = torch.cat([x, t, prompt_embed], dim=-1)
        return self.network(inputs)
    
    def sample(self, prompt_embed, num_steps=20, device="cpu"):
        """
        Sample a trajectory using Euler method.
        
        Args:
            prompt_embed: Prompt embedding [batch_size, prompt_dim]
            num_steps: Number of integration steps
            device: Device to run on
        
        Returns:
            trajectory: List of states during sampling [num_steps+1, batch_size, signal_dim]
            log_probs: Log probabilities for each step [num_steps, batch_size]
        """
        batch_size = prompt_embed.shape[0]
        
        # Start from noise
        x = torch.randn(batch_size, self.signal_dim, device=device)
        
        trajectory = [x.clone()]
        log_probs = []
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(batch_size, 1, device=device) * (i * dt)
            
            # Compute velocity
            v = self.forward(x, t, prompt_embed)
            
            # Euler step: x_{t+1} = x_t + dt * v_t
            x = x + dt * v
            
            # Compute log probability (simplified - assumes Gaussian)
            # In real flow matching, this is more complex
            log_prob = -0.5 * torch.sum((v - x) ** 2, dim=-1)
            log_probs.append(log_prob)
            
            trajectory.append(x.clone())
        
        return trajectory, torch.stack(log_probs, dim=0).transpose(0, 1)


class PromptEncoder(nn.Module):
    """Simple prompt encoder that maps digit labels to embeddings."""
    
    def __init__(self, vocab_size=10, embed_dim=32):
        """
        Args:
            vocab_size: Number of digits (0-9)
            embed_dim: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
    
    def forward(self, prompt_ids):
        """
        Encode digit labels to embeddings.
        
        Args:
            prompt_ids: Digit labels [batch_size] or [batch_size, 1]
        
        Returns:
            embeddings: [batch_size, embed_dim]
        """
        # Handle both [batch_size] and [batch_size, 1] shapes
        if prompt_ids.dim() > 1:
            prompt_ids = prompt_ids.squeeze(-1)
        # Ensure prompt_ids are in valid range [0, vocab_size-1]
        prompt_ids = prompt_ids.clamp(0, self.embedding.num_embeddings - 1)
        return self.embedding(prompt_ids)  # [batch_size, embed_dim]


def create_toy_model(signal_dim=784, prompt_dim=32, hidden_dim=256, vocab_size=10):
    """Create a toy flow model and prompt encoder."""
    model = SimpleFlowModel(signal_dim, prompt_dim, hidden_dim)
    encoder = PromptEncoder(vocab_size, prompt_dim)
    return model, encoder
