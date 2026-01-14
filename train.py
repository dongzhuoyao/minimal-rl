"""
Main training script for FlowGRPO tutorial.

Usage:
    python train.py                          # Use default config
    python train.py training=fast            # Use fast training config
    python train.py training=gpu            # Use GPU config
    python train.py model=large              # Use large model
    python train.py training.num_epochs=100 # Override specific parameter
    python train.py training.batch_size=8 training.lr=0.0005  # Override multiple
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from collections import defaultdict
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import MNISTDataset
from models.toy_flow_model import create_toy_model
from rewards.simple_reward import SimpleReward
from visualization.plotter import TrainingPlotter


# ============================================================================
# GRPO Functions
# ============================================================================

def compute_group_advantages(rewards, prompt_ids, clip_range=1e-4):
    """
    Compute group-relative advantages for GRPO.
    
    In GRPO, advantages are computed relative to the mean reward within
    each prompt group, rather than a global baseline. This reduces
    variance and improves training stability.
    
    Args:
        rewards: Tensor of rewards [batch_size]
        prompt_ids: Tensor of prompt IDs [batch_size]
        clip_range: Clipping range for advantages
    
    Returns:
        advantages: Tensor of advantages [batch_size]
        group_stats: Dict with group statistics
    """
    # Group rewards by prompt_id
    groups = defaultdict(list)
    for i, pid in enumerate(prompt_ids.cpu().numpy()):
        groups[pid].append(i)
    
    # Compute group means
    group_means = {}
    for pid, indices in groups.items():
        group_rewards = rewards[indices]
        group_means[pid] = group_rewards.mean().item()
    
    # Compute advantages relative to group mean
    advantages = torch.zeros_like(rewards)
    for pid, indices in groups.items():
        group_rewards = rewards[indices]
        group_mean = group_means[pid]
        advantages[indices] = group_rewards - group_mean
    
    # Clip advantages
    advantages = torch.clamp(advantages, -5.0, 5.0)
    
    # Compute statistics
    group_stats = {
        "num_groups": len(groups),
        "group_means": group_means,
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
    }
    
    return advantages, group_stats


def compute_grpo_loss(
    log_probs_new,
    log_probs_old,
    advantages,
    clip_range=1e-4,
    beta=0.0,
    log_probs_ref=None,
):
    """
    Compute GRPO policy loss.
    
    Args:
        log_probs_new: New policy log probabilities [batch_size, num_steps]
        log_probs_old: Old policy log probabilities [batch_size, num_steps]
        advantages: Advantages [batch_size, num_steps]
        clip_range: Clipping range for importance ratio
        beta: KL penalty coefficient (0 = no KL penalty)
        log_probs_ref: Reference policy log probs for KL penalty [batch_size, num_steps]
    
    Returns:
        loss: Policy loss scalar
        info: Dict with loss statistics
    """
    # Compute importance ratio
    ratio = torch.exp(log_probs_new - log_probs_old)
    
    # Clipped policy loss
    unclipped_loss = -advantages * ratio
    clipped_loss = -advantages * torch.clamp(
        ratio,
        1.0 - clip_range,
        1.0 + clip_range,
    )
    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
    
    # KL penalty (if beta > 0)
    kl_loss = torch.tensor(0.0, device=policy_loss.device)
    if beta > 0 and log_probs_ref is not None:
        kl = log_probs_new - log_probs_ref
        kl_loss = beta * torch.mean(kl)
    
    total_loss = policy_loss + kl_loss
    
    # Statistics
    info = {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
    }
    
    return total_loss, info


# ============================================================================
# FlowGRPOTrainer Class
# ============================================================================

class FlowGRPOTrainer:
    """Trainer for FlowGRPO."""
    
    def __init__(
        self,
        model,
        prompt_encoder,
        reward_fn,
        train_dataset,
        test_dataset,
        config,
    ):
        """
        Args:
            model: Flow matching model
            prompt_encoder: Prompt encoder
            reward_fn: Reward function
            train_dataset: Training dataset
            test_dataset: Test dataset
            config: Training configuration dict
        """
        self.model = model
        self.prompt_encoder = prompt_encoder
        self.reward_fn = reward_fn
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(prompt_encoder.parameters()),
            lr=config.get("lr", 1e-3),
        )
        
        # Setup device
        self.device = torch.device(config.get("device", "cpu"))
        self.model.to(self.device)
        self.prompt_encoder.to(self.device)
        
        # Setup visualization
        self.plotter = TrainingPlotter(config.get("output_dir", "outputs"))
        
        # Training state
        self.epoch = 0
        self.step = 0
    
    def encode_prompts(self, labels):
        """Encode digit labels to embeddings."""
        # Convert labels to tensor if needed
        if isinstance(labels, (list, tuple)):
            if isinstance(labels[0], str):
                # Convert string labels to integers
                labels = torch.tensor([int(l) for l in labels], device=self.device, dtype=torch.long)
            else:
                labels = torch.tensor(labels, device=self.device, dtype=torch.long)
        elif isinstance(labels, torch.Tensor):
            labels = labels.to(self.device).long()
        else:
            labels = torch.tensor([int(labels)], device=self.device, dtype=torch.long)
        
        # Ensure labels are in valid range [0, 9]
        labels = labels.clamp(0, 9)
        
        return self.prompt_encoder(labels)
    
    def sample_trajectories(self, labels, num_samples_per_prompt=4):
        """
        Sample trajectories from the current policy.
        
        Args:
            labels: Tensor or list of digit labels (0-9)
            num_samples_per_prompt: Number of samples per prompt
        
        Returns:
            samples: Dict containing trajectories, log_probs, etc.
        """
        # Convert labels to tensor if needed
        # Labels from DataLoader batch are already tensors
        if isinstance(labels, torch.Tensor):
            # Convert tensor to list of integers
            labels = labels.cpu().tolist()
        
        if isinstance(labels, (list, tuple)):
            # Ensure all elements are integers
            labels = [int(l) for l in labels]
        else:
            labels = [int(labels)]
        
        # Expand labels
        expanded_labels = []
        prompt_ids = []
        for i, label in enumerate(labels):
            for _ in range(num_samples_per_prompt):
                expanded_labels.append(label)
                prompt_ids.append(i)
        
        expanded_labels = torch.tensor(expanded_labels, device=self.device, dtype=torch.long)
        
        # Encode labels
        prompt_embeds = self.encode_prompts(expanded_labels)
        
        # Sample trajectories
        self.model.eval()
        with torch.no_grad():
            trajectory, log_probs = self.model.sample(
                prompt_embeds,
                num_steps=self.config.get("num_steps", 20),
                device=self.device,
            )
        
        # Get final signals
        final_signals = trajectory[-1]
        
        # Compute rewards
        rewards = self.reward_fn(final_signals, expanded_labels)
        
        return {
            "trajectory": trajectory,
            "log_probs": log_probs,
            "rewards": rewards,
            "labels": expanded_labels,
            "prompt_ids": torch.tensor(prompt_ids, device=self.device),
            "final_signals": final_signals,
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        self.prompt_encoder.train()
        
        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True,
        )
        
        epoch_losses = []
        epoch_rewards = []
        
        for batch in tqdm(dataloader, desc=f"Epoch {self.epoch}"):
            labels = batch["label"]
            
            # Sample trajectories
            samples = self.sample_trajectories(
                labels,
                num_samples_per_prompt=self.config.get("num_samples_per_prompt", 4),
            )
            
            # Compute advantages
            advantages, group_stats = compute_group_advantages(
                samples["rewards"],
                samples["prompt_ids"],
                clip_range=self.config.get("clip_range", 1e-4),
            )
            
            # Expand advantages to match log_probs shape
            # log_probs: [batch_size, num_steps]
            # advantages: [batch_size] -> expand to [batch_size, num_steps]
            num_steps = samples["log_probs"].shape[1]
            advantages_expanded = advantages.unsqueeze(1).expand(-1, num_steps)
            
            # Compute loss for each timestep
            total_loss = 0
            loss_info = {}
            
            for step_idx in range(num_steps):
                # Get log probs for this step
                log_probs_old = samples["log_probs"][:, step_idx].detach()
                advantages_step = advantages_expanded[:, step_idx]
                
                # Recompute log probs with current policy
                prompt_embeds = self.encode_prompts(samples["labels"])
                
                # Get state at this step
                x = samples["trajectory"][step_idx]
                t = torch.ones(len(samples["labels"]), 1, device=self.device) * (
                    step_idx / num_steps
                )
                
                # Compute velocity and log prob
                v = self.model(x, t, prompt_embeds)
                log_prob_new = -0.5 * torch.sum((v - x) ** 2, dim=-1)
                
                # Compute loss
                loss, info = compute_grpo_loss(
                    log_prob_new.unsqueeze(1),
                    log_probs_old.unsqueeze(1),
                    advantages_step.unsqueeze(1),
                    clip_range=self.config.get("clip_range", 1e-4),
                    beta=self.config.get("beta", 0.0),
                )
                
                total_loss += loss
                
                # Accumulate info
                for k, v in info.items():
                    if k not in loss_info:
                        loss_info[k] = []
                    loss_info[k].append(v)
            
            # Average loss over steps
            total_loss = total_loss / num_steps
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.prompt_encoder.parameters()),
                self.config.get("max_grad_norm", 1.0),
            )
            self.optimizer.step()
            
            # Logging
            epoch_losses.append(total_loss.item())
            epoch_rewards.append(samples["rewards"].mean().item())
            
            self.step += 1
        
        # Average statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_reward = sum(epoch_rewards) / len(epoch_rewards)
        
        return {
            "loss": avg_loss,
            "reward": avg_reward,
            "loss_info": {k: sum(v) / len(v) for k, v in loss_info.items()},
        }
    
    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        self.prompt_encoder.eval()
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.get("test_batch_size", 8),
            shuffle=False,
        )
        
        all_rewards = []
        all_signals = []
        all_prompts = []
        
        with torch.no_grad():
            for batch in test_loader:
                labels = batch["label"]
                
                # Sample single trajectory per label
                prompt_embeds = self.encode_prompts(labels)
                trajectory, _ = self.model.sample(
                    prompt_embeds,
                    num_steps=self.config.get("eval_num_steps", 20),
                    device=self.device,
                )
                
                final_signals = trajectory[-1]
                rewards = self.reward_fn(final_signals, labels)
                
                all_rewards.extend(rewards.cpu().numpy())
                all_signals.append(final_signals.cpu())
                all_prompts.extend(labels.cpu().tolist())
        
        return {
            "rewards": all_rewards,
            "signals": torch.cat(all_signals, dim=0),
            "prompts": all_prompts,
            "mean_reward": sum(all_rewards) / len(all_rewards),
        }
    
    def train(self, num_epochs):
        """Main training loop."""
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_stats = self.train_epoch()
            
            # Evaluate
            if (epoch + 1) % self.config.get("eval_freq", 5) == 0:
                eval_stats = self.evaluate()
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_stats['loss']:.4f}")
                print(f"Train Reward: {train_stats['reward']:.4f}")
                print(f"Test Reward: {eval_stats['mean_reward']:.4f}")
                
                # Visualize
                # Reshape signals from [batch, 784] to [batch, 28, 28] for visualization
                signals_reshaped = eval_stats["signals"][:8].view(-1, 28, 28)
                self.plotter.update(
                    epoch,
                    train_stats,
                    eval_stats,
                    signals_reshaped,  # Show first 8 samples as 28x28 images
                    eval_stats["prompts"][:8],
                )
            else:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_stats['loss']:.4f}")
                print(f"Train Reward: {train_stats['reward']:.4f}")


# ============================================================================
# Main Training Function
# ============================================================================


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration."""
    
    # Print configuration
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Set random seed
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    
    # Determine device
    device = cfg.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load MNIST datasets
    print("Loading MNIST datasets...")
    dataset_dir = Path(cfg.dataset.dataset_dir)
    train_dataset = MNISTDataset(dataset_dir, split="train", download=True)
    test_dataset = MNISTDataset(dataset_dir, split="test", download=True)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Creating model...")
    model, prompt_encoder = create_toy_model(
        signal_dim=cfg.model.signal_dim,
        prompt_dim=cfg.model.prompt_dim,
        hidden_dim=cfg.model.hidden_dim,
        vocab_size=cfg.model.vocab_size,
    )
    
    # Create reward function
    reward_fn = SimpleReward()
    
    # Training config dictionary for trainer
    trainer_config = {
        "batch_size": cfg.training.batch_size,
        "num_samples_per_prompt": cfg.training.num_samples_per_prompt,
        "num_steps": cfg.training.num_steps,
        "eval_num_steps": cfg.training.eval_num_steps,
        "lr": cfg.training.lr,
        "clip_range": cfg.training.clip_range,
        "beta": cfg.training.beta,
        "device": device,
        "output_dir": cfg.paths.output_dir,
        "eval_freq": cfg.training.eval_freq,
        "max_grad_norm": cfg.training.max_grad_norm,
    }
    
    # Create trainer
    trainer = FlowGRPOTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=trainer_config,
    )
    
    # Train
    print("Starting training...")
    trainer.train(num_epochs=cfg.training.num_epochs)
    
    print(f"\nTraining complete! Check outputs in {cfg.paths.output_dir}")


if __name__ == "__main__":
    main()
