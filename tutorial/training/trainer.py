"""
Main training loop for FlowGRPO tutorial.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

from tutorial.models.toy_flow_model import create_toy_model
from tutorial.rewards.simple_reward import SimpleReward
from tutorial.training.grpo import compute_group_advantages, compute_grpo_loss
from tutorial.visualization.plotter import TrainingPlotter


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
            lr=config.get("learning_rate", 1e-3),
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
    
    def encode_prompts(self, prompts):
        """Encode text prompts to embeddings."""
        # Simple tokenization: map words to IDs
        # In practice, you'd use a proper tokenizer
        vocab = {
            "a": 0, "red": 1, "blue": 2, "green": 3, "yellow": 4,
            "circle": 5, "square": 6, "triangle": 7,
            "small": 8, "large": 9, "purple": 10, "orange": 11,
            "pink": 12, "cyan": 13, "magenta": 14,
        }
        
        batch_size = len(prompts)
        max_len = 10
        
        prompt_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=self.device)
        
        for i, prompt in enumerate(prompts):
            words = prompt.lower().split()
            for j, word in enumerate(words[:max_len]):
                prompt_ids[i, j] = vocab.get(word, 0)
        
        return self.prompt_encoder(prompt_ids)
    
    def sample_trajectories(self, prompts, num_samples_per_prompt=4):
        """
        Sample trajectories from the current policy.
        
        Args:
            prompts: List of prompt strings
            num_samples_per_prompt: Number of samples per prompt
        
        Returns:
            samples: Dict containing trajectories, log_probs, etc.
        """
        # Expand prompts
        expanded_prompts = []
        prompt_ids = []
        for i, prompt in enumerate(prompts):
            for _ in range(num_samples_per_prompt):
                expanded_prompts.append(prompt)
                prompt_ids.append(i)
        
        # Encode prompts
        prompt_embeds = self.encode_prompts(expanded_prompts)
        
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
        rewards = self.reward_fn(final_signals, expanded_prompts)
        
        return {
            "trajectory": trajectory,
            "log_probs": log_probs,
            "rewards": rewards,
            "prompts": expanded_prompts,
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
            prompts = batch["prompt"]
            
            # Sample trajectories
            samples = self.sample_trajectories(
                prompts,
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
                prompt_embeds = self.encode_prompts(samples["prompts"])
                
                # Get state at this step
                x = samples["trajectory"][step_idx]
                t = torch.ones(len(samples["prompts"]), 1, device=self.device) * (
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
                prompts = batch["prompt"]
                
                # Sample single trajectory per prompt
                prompt_embeds = self.encode_prompts(prompts)
                trajectory, _ = self.model.sample(
                    prompt_embeds,
                    num_steps=self.config.get("eval_num_steps", 20),
                    device=self.device,
                )
                
                final_signals = trajectory[-1]
                rewards = self.reward_fn(final_signals, prompts)
                
                all_rewards.extend(rewards.cpu().numpy())
                all_signals.append(final_signals.cpu())
                all_prompts.extend(prompts)
        
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
                self.plotter.update(
                    epoch,
                    train_stats,
                    eval_stats,
                    eval_stats["signals"][:8],  # Show first 8 samples
                    eval_stats["prompts"][:8],
                )
            else:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f"Train Loss: {train_stats['loss']:.4f}")
                print(f"Train Reward: {train_stats['reward']:.4f}")
