"""
Evaluation utilities for FlowGRPO tutorial.
"""
import torch
import numpy as np
from collections import defaultdict


class Evaluator:
    """Evaluate model performance."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def compute_metrics(self, signals, prompts, rewards):
        """
        Compute evaluation metrics.
        
        Args:
            signals: Generated signals [batch_size, signal_dim]
            prompts: List of prompt strings
            rewards: Reward values [batch_size]
        
        Returns:
            metrics: Dict of metric names to values
        """
        metrics = {}
        
        # Average reward
        metrics["mean_reward"] = float(rewards.mean().item() if isinstance(rewards, torch.Tensor) else np.mean(rewards))
        metrics["std_reward"] = float(rewards.std().item() if isinstance(rewards, torch.Tensor) else np.std(rewards))
        
        # Reward by prompt type
        prompt_types = defaultdict(list)
        for i, prompt in enumerate(prompts):
            if "circle" in prompt.lower():
                prompt_types["circle"].append(rewards[i].item() if isinstance(rewards, torch.Tensor) else rewards[i])
            elif "square" in prompt.lower():
                prompt_types["square"].append(rewards[i].item() if isinstance(rewards, torch.Tensor) else rewards[i])
            elif "triangle" in prompt.lower():
                prompt_types["triangle"].append(rewards[i].item() if isinstance(rewards, torch.Tensor) else rewards[i])
        
        for shape, shape_rewards in prompt_types.items():
            metrics[f"reward_{shape}"] = float(np.mean(shape_rewards))
        
        # Signal statistics
        if isinstance(signals, torch.Tensor):
            signals_np = signals.cpu().numpy()
        else:
            signals_np = signals
        
        metrics["signal_mean"] = float(np.mean(signals_np))
        metrics["signal_std"] = float(np.std(signals_np))
        metrics["signal_range"] = float(np.max(signals_np) - np.min(signals_np))
        
        return metrics
    
    def update(self, metrics):
        """Update running metrics."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
    
    def get_summary(self):
        """Get summary statistics."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = float(np.mean(values))
                summary[f"{key}_std"] = float(np.std(values))
        return summary
