"""
Simple example demonstrating FlowGRPO usage.

This script shows how to:
1. Load the dataset
2. Create a model
3. Sample trajectories
4. Compute rewards
5. Visualize results
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tutorial.dataset.dataset import PromptDataset
from tutorial.dataset.generate_dataset import generate_dataset
from tutorial.models.toy_flow_model import create_toy_model
from tutorial.rewards.simple_reward import SimpleReward
import matplotlib.pyplot as plt


def main():
    # Generate dataset if needed
    dataset_dir = Path("tutorial/dataset")
    if not (dataset_dir / "train.txt").exists():
        print("Generating dataset...")
        generate_dataset()
    
    # Load dataset
    dataset = PromptDataset(dataset_dir, split="train")
    print(f"Loaded {len(dataset)} prompts")
    
    # Create model
    model, prompt_encoder = create_toy_model()
    print("Created model")
    
    # Create reward function
    reward_fn = SimpleReward()
    
    # Get a few prompts
    prompts = [dataset[i]["prompt"] for i in range(4)]
    print(f"\nPrompts: {prompts}")
    
    # Simple tokenization (same as in trainer)
    vocab = {
        "a": 0, "red": 1, "blue": 2, "green": 3, "yellow": 4,
        "circle": 5, "square": 6, "triangle": 7,
        "small": 8, "large": 9, "purple": 10, "orange": 11,
        "pink": 12, "cyan": 13, "magenta": 14,
    }
    
    batch_size = len(prompts)
    max_len = 10
    prompt_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    
    for i, prompt in enumerate(prompts):
        words = prompt.lower().split()
        for j, word in enumerate(words[:max_len]):
            prompt_ids[i, j] = vocab.get(word, 0)
    
    # Encode prompts
    prompt_embeds = prompt_encoder(prompt_ids)
    
    # Sample trajectories
    print("\nSampling trajectories...")
    model.eval()
    with torch.no_grad():
        trajectory, log_probs = model.sample(prompt_embeds, num_steps=20)
    
    final_signals = trajectory[-1]
    print(f"Generated signals shape: {final_signals.shape}")
    
    # Compute rewards
    rewards = reward_fn(final_signals, prompts)
    print(f"\nRewards: {rewards}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i in range(len(prompts)):
        signal = final_signals[i].cpu().numpy()
        axes[i].plot(signal, "b-", linewidth=2)
        axes[i].set_title(f"{prompts[i]}\nReward: {rewards[i]:.3f}")
        axes[i].set_xlabel("Position")
        axes[i].set_ylabel("Value")
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle("Example Generated Signals", fontsize=16)
    plt.tight_layout()
    
    output_dir = Path("tutorial/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "example_signals.png", dpi=150)
    print(f"\nSaved visualization to {output_dir / 'example_signals.png'}")


if __name__ == "__main__":
    main()
