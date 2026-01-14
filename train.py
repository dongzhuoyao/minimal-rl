"""
FloWGRPO (Flow-based Group Relative Policy Optimization) Training Script.

This script implements GRPO training for flow matching models, following the original
implementation pattern. It loads a pretrained checkpoint from train0.py and fine-tunes
using reward-based optimization.

Usage:
    python train.py                          # Use default config
    python train.py training.max_steps=5000  # Override specific parameter
"""
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import os
import wandb
import numpy as np
from PIL import Image
from collections import defaultdict
import copy

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import MNISTDataset
from model import create_mnist_model
from rewards.mnist_rewards import CombinedMNISTReward, get_recommended_reward_config


class PerPromptStatTracker:
    """
    Tracks statistics per prompt/label for computing advantages in GRPO.
    Based on original_impl/flow_grpo/stat_tracking.py
    """
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    def update(self, prompts, rewards, type='grpo'):
        """
        Compute advantages from rewards using per-prompt statistics.
        
        Args:
            prompts: List of prompt strings (or labels) [batch_size]
            rewards: Array of rewards [batch_size]
            type: Type of advantage computation ('grpo', 'rwr', 'sft', 'dpo')
        
        Returns:
            advantages: Array of advantages [batch_size]
        """
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(str(prompt)))
        
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            
            if type == 'grpo':
                advantages[prompts == prompt] = (prompt_rewards - mean) / std
            elif type == 'rwr':
                advantages[prompts == prompt] = prompt_rewards
            elif type == 'sft':
                advantages[prompts == prompt] = (torch.tensor(prompt_rewards) == torch.max(torch.tensor(prompt_rewards))).float().numpy()
            elif type == 'dpo':
                prompt_advantages = torch.tensor(prompt_rewards)
                max_idx = torch.argmax(prompt_advantages)
                min_idx = torch.argmin(prompt_advantages)
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(prompt_advantages).float()
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[prompts == prompt] = result.numpy()
        
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts
    
    def clear(self):
        self.stats = {}


def sample_with_logprob(model, labels, num_steps, device, return_trajectory=False, enable_grad=False):
    """
    Sample trajectories with log probabilities for GRPO training.
    
    Args:
        model: Flow matching model
        labels: [B] class labels (0-9)
        num_steps: Number of integration steps
        device: Device to run on
        return_trajectory: If True, return full trajectory
        enable_grad: If True, enable gradients (for training phase)
    
    Returns:
        final_images: [B, signal_dim] final generated images
        trajectory: List of [B, signal_dim] tensors (if return_trajectory=True)
        log_probs: [B, num_steps] log probabilities for each step
        timesteps: [B, num_steps] time values for each step
    """
    if not enable_grad:
        model.eval()
    B = labels.shape[0]
    signal_dim = model.signal_dim
    
    # Initialize with noise
    x = torch.randn(B, signal_dim, device=device)
    
    trajectory = [x.clone()] if return_trajectory else None
    log_probs = []
    timesteps = []
    
    # Euler integration
    dt = 1.0 / num_steps
    t = torch.zeros(B, 1, device=device)
    
    context = torch.enable_grad() if enable_grad else torch.no_grad()
    with context:
        for step in range(num_steps):
            # Compute velocity
            v = model(x, t, labels)
            
            # Compute log probability for this step
            # For flow matching: log p ≈ -0.5 * ||v||^2 * dt
            log_prob = -0.5 * torch.sum(v ** 2, dim=-1) * dt
            log_probs.append(log_prob)
            timesteps.append(t.clone())
            
            # Update x
            x_new = x + dt * v
            
            # Update x
            x = x_new
            
            # Update time
            t = t + dt
            
            # Store trajectory if needed
            if return_trajectory:
                trajectory.append(x.clone())
    
    log_probs = torch.stack(log_probs, dim=1)  # [B, num_steps]
    timesteps = torch.cat(timesteps, dim=1)  # [B, num_steps]
    
    # Reshape final images to [B, 1, 28, 28] for reward computation
    final_images = x.view(B, 1, 28, 28)
    
    if return_trajectory:
        return final_images, trajectory, log_probs, timesteps
    else:
        return final_images, log_probs, timesteps


def compute_log_prob_at_timestep(model, x_t, t, labels, x_next, dt):
    """
    Compute log probability of transitioning from x_t to x_next at timestep t.
    
    Args:
        model: Flow matching model
        x_t: [B, signal_dim] current state
        t: [B, 1] current time
        labels: [B] class labels
        x_next: [B, signal_dim] next state (from trajectory)
        dt: Time step size
    
    Returns:
        log_prob: [B] log probabilities
    """
    # Predict velocity
    v_pred = model(x_t, t, labels)
    
    # Compute log probability: -0.5 * ||v_pred||^2 * dt
    log_prob = -0.5 * torch.sum(v_pred ** 2, dim=-1) * dt
    
    return log_prob


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    """Main FloWGRPO training function with Hydra configuration."""
    
    # Print configuration
    print("=" * 60)
    print("FloWGRPO Training Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Initialize wandb
    if cfg.wandb.enabled:
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = Path(hydra_cfg.run.dir)
        wandb_dir = hydra_output_dir / "wandb"
        os.environ["WANDB_DIR"] = str(wandb_dir)
        wandb_dir.mkdir(parents=True, exist_ok=True)
        
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name if cfg.wandb.run_name else cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.get("tags", []),
            dir=str(wandb_dir),
        )
        print(f"Wandb initialized: {wandb.run.url}")
    
    # Set random seed
    if hasattr(cfg, 'seed'):
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        np.random.seed(cfg.seed)
    
    # Determine device
    device = cfg.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    device = torch.device(device)
    
    # Create output directory
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MNIST dataset
    print("Loading MNIST datasets...")
    dataset_dir = Path(cfg.dataset.dataset_dir)
    train_dataset = MNISTDataset(dataset_dir, split="train", download=True)
    print(f"Train samples: {len(train_dataset)}")
    
    # Create model
    print("Creating flow matching model...")
    model = create_mnist_model(
        signal_dim=cfg.model.signal_dim,
        time_emb_dim=cfg.model.get("time_emb_dim", 40),
        vocab_size=cfg.model.vocab_size,
    )
    model.to(device)
    
    # Load pretrained checkpoint
    checkpoint_path = cfg.pretrained_checkpoint
    if checkpoint_path is None:
        # Default to checkpoint from train0.py
        checkpoint_path = "outputs/flow_matching/best_model.pt"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading pretrained checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch.")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )
    
    # Initialize reward function
    print("Initializing reward function...")
    reward_config = cfg.get("reward", {}).get("config", "balanced")
    reward_fn = get_recommended_reward_config(reward_config, device=device)
    
    # Initialize stat tracker for advantages
    stat_tracker = PerPromptStatTracker(global_std=cfg.training.get("global_std", False))
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.get("num_workers", 4),
    )
    
    # Training loop
    print("Starting FloWGRPO training...")
    train_iter = iter(train_loader)
    step = 0
    epoch = 0
    
    pbar = tqdm(total=cfg.training.max_steps, desc="Training")
    
    while step < cfg.training.max_steps:
        # ========== SAMPLING PHASE ==========
        model.eval()
        samples = []
        
        # Sample multiple times per batch
        num_samples_per_prompt = cfg.training.num_samples_per_prompt
        batch_size = cfg.training.batch_size
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            epoch += 1
        
        labels = batch["label"].to(device)  # [batch_size]
        prompts = batch["prompt"]  # List of strings
        
        # Expand labels and prompts for multiple samples per prompt
        expanded_labels = labels.repeat_interleave(num_samples_per_prompt)  # [batch_size * num_samples_per_prompt]
        expanded_prompts = [p for p in prompts for _ in range(num_samples_per_prompt)]
        
        # Sample trajectories
        with torch.no_grad():
            final_images, log_probs_old, timesteps = sample_with_logprob(
                model,
                expanded_labels,
                num_steps=cfg.training.num_steps,
                device=device,
                return_trajectory=False,
            )
        
        # Compute rewards
        # Convert images to format expected by reward function: [B, 1, 28, 28] -> [B, 1, 28, 28]
        # Reward function expects images in [0, 1] range
        images_for_reward = torch.clamp(final_images, 0.0, 1.0)
        
        # Create prompt strings for reward function
        reward_prompts = [f"digit {label.item()}" for label in expanded_labels]
        
        rewards = reward_fn(images_for_reward, reward_prompts)  # [batch_size * num_samples_per_prompt]
        rewards = rewards.cpu().numpy()
        
        # Compute advantages using stat tracker
        advantages = stat_tracker.update(expanded_prompts, rewards, type='grpo')
        advantages = torch.tensor(advantages, device=device, dtype=torch.float32)
        
        # Reshape for training: [batch_size * num_samples_per_prompt, num_steps] -> [batch_size, num_samples_per_prompt, num_steps]
        log_probs_old = log_probs_old.view(batch_size, num_samples_per_prompt, cfg.training.num_steps)
        advantages = advantages.view(batch_size, num_samples_per_prompt)
        rewards = rewards.reshape(batch_size, num_samples_per_prompt)
        
        # Store samples for training (need to store trajectories for proper training)
        # Re-sample with trajectories stored (no grad for sampling phase)
        with torch.no_grad():
            _, trajectories_full, log_probs_old_full, timesteps_full = sample_with_logprob(
                model,
                expanded_labels,
                num_steps=cfg.training.num_steps,
                device=device,
                return_trajectory=True,
                enable_grad=False,
            )
        
        # Store samples with trajectories
        for i in range(batch_size):
            for j in range(num_samples_per_prompt):
                sample_idx = i * num_samples_per_prompt + j
                samples.append({
                    "label": expanded_labels[sample_idx],
                    "prompt": expanded_prompts[sample_idx],
                    "log_probs_old": log_probs_old_full[sample_idx],  # [num_steps]
                    "trajectory": [traj[sample_idx] for traj in trajectories_full],  # List of [signal_dim]
                    "timesteps": timesteps_full[sample_idx],  # [num_steps]
                    "advantages": advantages[i, j],  # scalar
                    "reward": rewards[i, j],  # scalar
                    "image": final_images[sample_idx],  # [1, 28, 28]
                })
        
        # ========== TRAINING PHASE ==========
        model.train()
        
        # Shuffle samples
        np.random.shuffle(samples)
        
        # Train on samples
        num_inner_epochs = cfg.training.get("num_inner_epochs", 1)
        for inner_epoch in range(num_inner_epochs):
            # Re-shuffle for each inner epoch
            if inner_epoch > 0:
                np.random.shuffle(samples)
            
            # Process samples in mini-batches
            train_batch_size = cfg.training.get("train_batch_size", batch_size)
            for batch_start in range(0, len(samples), train_batch_size):
                batch_samples = samples[batch_start:batch_start + train_batch_size]
                
                if len(batch_samples) == 0:
                    continue
                
                # Extract batch data
                batch_labels = torch.stack([s["label"] for s in batch_samples])
                batch_log_probs_old = torch.stack([s["log_probs_old"] for s in batch_samples])  # [B, num_steps]
                batch_advantages = torch.stack([s["advantages"] for s in batch_samples])  # [B]
                
                # For each timestep, compute new log prob and GRPO loss
                num_train_timesteps = cfg.training.num_steps
                dt = 1.0 / num_train_timesteps
                
                total_loss = 0.0
                info = defaultdict(list)
                
                # Extract trajectories from samples
                trajectories_list = []
                timesteps_list = []
                for s in batch_samples:
                    trajectories_list.append(s["trajectory"])
                    timesteps_list.append(s["timesteps"])
                
                # Convert to tensors: [B, num_steps+1, signal_dim]
                max_len = max(len(traj) for traj in trajectories_list)
                signal_dim = trajectories_list[0][0].shape[0]
                trajectories_tensor = torch.zeros(len(batch_samples), max_len, signal_dim, device=device)
                for i, traj in enumerate(trajectories_list):
                    for j, state in enumerate(traj):
                        trajectories_tensor[i, j] = state
                
                timesteps_tensor = torch.stack(timesteps_list, dim=0)  # [B, num_steps]
                
                # Train on each timestep
                for j in range(num_train_timesteps):
                    # Get states at timestep j
                    x_t = trajectories_tensor[:, j]  # [B, signal_dim]
                    x_next = trajectories_tensor[:, j + 1]  # [B, signal_dim]
                    t = timesteps_tensor[:, j:j+1]  # [B, 1]
                    
                    # Compute new log probability
                    log_prob_new = compute_log_prob_at_timestep(
                        model, x_t, t, batch_labels, x_next, dt
                    )  # [B]
                    
                    # Get old log probability for this timestep
                    log_prob_old = batch_log_probs_old[:, j]  # [B]
                    
                    # Expand advantages to match timesteps (same advantage for all timesteps)
                    advantages_expanded = batch_advantages  # [B]
                    
                    # Clip advantages
                    adv_clip_max = cfg.training.get("adv_clip_max", 10.0)
                    advantages_clipped = torch.clamp(
                        advantages_expanded,
                        -adv_clip_max,
                        adv_clip_max,
                    )
                    
                    # Compute ratio
                    ratio = torch.exp(log_prob_new - log_prob_old)  # [B]
                    
                    # GRPO loss: clipped PPO-style loss
                    clip_range = cfg.training.clip_range
                    unclipped_loss = -advantages_clipped * ratio
                    clipped_loss = -advantages_clipped * torch.clamp(
                        ratio,
                        1.0 - clip_range,
                        1.0 + clip_range,
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    
                    # Optional KL penalty (if beta > 0)
                    beta = cfg.training.get("beta", 0.0)
                    if beta > 0:
                        # Approximate KL divergence: 0.5 * (log_prob_new - log_prob_old)^2
                        kl_loss = 0.5 * torch.mean((log_prob_new - log_prob_old) ** 2)
                        loss = policy_loss + beta * kl_loss
                    else:
                        loss = policy_loss
                    
                    total_loss += loss
                    
                    # Track metrics
                    info["approx_kl"].append(0.5 * torch.mean((log_prob_new - log_prob_old) ** 2).item())
                    info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > clip_range).float()).item())
                    info["policy_loss"].append(policy_loss.item())
                    if beta > 0:
                        info["kl_loss"].append(kl_loss.item())
                
                # Average loss across timesteps
                total_loss = total_loss / num_train_timesteps
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if cfg.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.training.max_grad_norm,
                    )
                
                optimizer.step()
                
                step += 1
                pbar.update(1)
                
                # Logging
                if cfg.wandb.enabled and step % cfg.wandb.log_freq == 0:
                    log_dict = {
                        "train/loss": total_loss.item(),
                        "train/policy_loss": np.mean(info["policy_loss"]),
                        "train/approx_kl": np.mean(info["approx_kl"]),
                        "train/clipfrac": np.mean(info["clipfrac"]),
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "train/mean_reward": np.mean([s["reward"] for s in batch_samples]),
                        "train/mean_advantage": advantages_clipped.mean().item(),
                        "step": step,
                    }
                    if beta > 0:
                        log_dict["train/kl_loss"] = np.mean(info["kl_loss"])
                    
                    wandb.log(log_dict, step=step)
                
                # Evaluation and checkpointing
                if step % cfg.training.eval_freq == 0:
                    # Sample some images for visualization
                    if cfg.wandb.enabled:
                        model.eval()
                        with torch.no_grad():
                            test_labels = torch.randint(0, cfg.model.vocab_size, (8,), device=device)
                            test_images, _, _ = sample_with_logprob(
                                model,
                                test_labels,
                                num_steps=cfg.training.eval_num_steps,
                                device=device,
                            )
                            test_images = torch.clamp(test_images, 0.0, 1.0)
                            
                            # Convert to wandb images
                            wandb_images = []
                            for i in range(len(test_images)):
                                img = test_images[i, 0].cpu().numpy()
                                img = (img * 255).astype(np.uint8)
                                pil_img = Image.fromarray(img, mode='L')
                                wandb_img = wandb.Image(pil_img, caption=f"Label: {test_labels[i].item()}")
                                wandb_images.append(wandb_img)
                            
                            wandb.log({"eval/sampled_images": wandb_images}, step=step)
                        model.train()
                    
                    # Save checkpoint
                    checkpoint = {
                        'step': step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(checkpoint, output_dir / f'checkpoint_step_{step}.pt')
                    print(f"\nSaved checkpoint at step {step}")
                
                if step >= cfg.training.max_steps:
                    break
            
            if step >= cfg.training.max_steps:
                break
        
        if step >= cfg.training.max_steps:
            break
    
    pbar.close()
    
    # Save final model
    final_checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(final_checkpoint, output_dir / 'final_model.pt')
    print(f"\nTraining complete! Final model saved to {output_dir / 'final_model.pt'}")
    
    # Finish wandb run
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
