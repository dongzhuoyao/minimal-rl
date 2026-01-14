"""
Main training script for FlowGRPO tutorial.

Usage:
    python train.py
"""
import argparse
from pathlib import Path
import torch

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataset import PromptDataset
from dataset.generate_dataset import generate_dataset
from models.toy_flow_model import create_toy_model
from rewards.simple_reward import SimpleReward
from training.trainer import FlowGRPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train FlowGRPO on toy dataset")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_samples_per_prompt", type=int, default=4, help="Samples per prompt")
    parser.add_argument("--num_steps", type=int, default=20, help="Number of flow steps")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--clip_range", type=float, default=1e-4, help="GRPO clip range")
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty coefficient")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--eval_freq", type=int, default=5, help="Evaluation frequency")
    
    args = parser.parse_args()
    
    # Generate dataset if needed
    dataset_dir = Path(args.dataset_dir)
    if not (dataset_dir / "train.txt").exists():
        print("Generating dataset...")
        generate_dataset()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = PromptDataset(dataset_dir, split="train")
    test_dataset = PromptDataset(dataset_dir, split="test")
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("Creating model...")
    model, prompt_encoder = create_toy_model(
        signal_dim=64,
        prompt_dim=32,
        hidden_dim=128,
        vocab_size=20,
    )
    
    # Create reward function
    reward_fn = SimpleReward()
    
    # Training config
    config = {
        "batch_size": args.batch_size,
        "num_samples_per_prompt": args.num_samples_per_prompt,
        "num_steps": args.num_steps,
        "eval_num_steps": args.num_steps,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        "beta": args.beta,
        "device": args.device,
        "output_dir": args.output_dir,
        "eval_freq": args.eval_freq,
        "max_grad_norm": 1.0,
    }
    
    # Create trainer
    trainer = FlowGRPOTrainer(
        model=model,
        prompt_encoder=prompt_encoder,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        config=config,
    )
    
    # Train
    print("Starting training...")
    trainer.train(num_epochs=args.num_epochs)
    
    print(f"\nTraining complete! Check outputs in {args.output_dir}")


if __name__ == "__main__":
    main()
