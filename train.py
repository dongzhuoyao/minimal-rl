"""
Main training script for FlowGRPO tutorial.

Usage:
    python train.py                          # Use default config
    python train.py training=fast            # Use fast training config
    python train.py training=gpu            # Use GPU config
    python train.py model=large              # Use large model
    python train.py training.num_epochs=100 # Override specific parameter
    python train.py training.batch_size=8 training.learning_rate=0.0005  # Override multiple
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.dataset import PromptDataset
from dataset.generate_dataset import generate_dataset
from models.toy_flow_model import create_toy_model
from rewards.simple_reward import SimpleReward
from training.trainer import FlowGRPOTrainer


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
    
    # Generate dataset if needed
    dataset_dir = Path(cfg.dataset.dataset_dir)
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
        "learning_rate": cfg.training.learning_rate,
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
