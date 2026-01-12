"""
Standalone training script for Google Colab.

Copy this entire file into a Colab cell and run it, or use the notebook version.
"""
import os
import sys
from pathlib import Path
import torch

# Setup paths
BASE_DIR = Path.cwd()
sys.path.insert(0, str(BASE_DIR))

# Create directories
os.makedirs("tutorial/dataset", exist_ok=True)
os.makedirs("tutorial/outputs", exist_ok=True)

# Import after path setup
from tutorial.dataset.dataset import PromptDataset
from tutorial.dataset.generate_dataset import generate_dataset
from tutorial.models.toy_flow_model import create_toy_model
from tutorial.rewards.simple_reward import SimpleReward
from tutorial.training.trainer import FlowGRPOTrainer

def main():
    """Main training function."""
    # Setup paths
    dataset_dir = Path("tutorial/dataset")
    output_dir = Path("tutorial/outputs")
    
    # Generate dataset if needed
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
    # Use CUDA if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    config = {
        "batch_size": 4,
        "num_samples_per_prompt": 4,
        "num_steps": 20,
        "eval_num_steps": 20,
        "learning_rate": 1e-3,
        "clip_range": 1e-4,
        "beta": 0.0,
        "device": device,
        "output_dir": str(output_dir),
        "eval_freq": 5,
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
    trainer.train(num_epochs=50)
    
    print(f"\nTraining complete! Check outputs in {output_dir}")
    
    # List output files
    print("\nOutput files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
