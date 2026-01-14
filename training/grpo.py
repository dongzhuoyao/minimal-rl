"""
Group Relative Policy Optimization (GRPO) implementation.

GRPO is similar to PPO but uses group-based advantages computed relative
to the mean reward within each prompt group.
"""
import torch
import numpy as np
from collections import defaultdict


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
