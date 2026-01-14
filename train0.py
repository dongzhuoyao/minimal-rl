"""
Flow Matching training script.

Flow matching is a supervised learning approach where the model learns to predict
the velocity field that transforms noise to data.

Usage:
    python train0.py                          # Use default config
    python train0.py training.num_steps=10000  # Override specific parameter
    python train0.py training.lr=0.0005      # Override learning rate
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import MNISTDataset
from models.toy_flow_model import create_toy_model


@hydra.main(version_base=None, config_path=".", config_name="config0")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration."""
    
    # Print configuration
    print("=" * 60)
    print("Flow Matching Training Configuration:")
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
    device = torch.device(device)
    
    # Create output directory
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    model.to(device)
    prompt_encoder.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(prompt_encoder.parameters()),
        lr=cfg.training.lr,
    )
    
    # Training loop
    print("Starting training...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    
    # Create iterator that cycles through the data loader
    train_iter = iter(train_loader)
    
    best_test_loss = float('inf')
    step = 0
    running_train_loss = 0.0
    loss_count = 0
    
    # Training loop - step-based
    pbar = tqdm(total=cfg.training.num_steps, desc="Training")
    
    while step < cfg.training.num_steps:
        # Training phase
        model.train()
        prompt_encoder.train()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            # Reset iterator when exhausted
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        images = batch["image"].to(device)  # [batch_size, 784]
        labels = batch["label"].to(device)  # [batch_size]
        
        # Encode prompts
        prompt_embeds = prompt_encoder(labels)  # [batch_size, prompt_dim]
        
        # Flow matching loss computation
        batch_size = images.shape[0]
        
        # Sample random time t uniformly from [0, 1]
        t = torch.rand(batch_size, 1, device=device)  # [batch_size, 1]
        
        # Sample noise x_1 ~ N(0, I)
        noise = torch.randn_like(images)  # [batch_size, 784]
        
        # Interpolate: x_t = (1-t) * x_0 + t * x_1
        # where x_0 = data (images) and x_1 = noise
        x_t = (1 - t) * images + t * noise  # [batch_size, 784]
        
        # Target velocity: v_target = x_1 - x_0 = noise - images
        v_target = noise - images  # [batch_size, 784]
        
        # Predict velocity: v_pred = model(x_t, t, prompt_embed)
        v_pred = model(x_t, t, prompt_embeds)  # [batch_size, 784]
        
        # Flow matching loss: L = ||v_pred - v_target||^2
        loss = nn.functional.mse_loss(v_pred, v_target, reduction='mean')
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(prompt_encoder.parameters()),
                cfg.training.max_grad_norm,
            )
        
        optimizer.step()
        
        # Update running statistics
        running_train_loss += loss.item()
        loss_count += 1
        step += 1
        pbar.update(1)
        
        # Evaluation phase
        if step % cfg.training.eval_freq == 0:
            avg_train_loss = running_train_loss / loss_count
            running_train_loss = 0.0
            loss_count = 0
            
            model.eval()
            prompt_encoder.eval()
            test_losses = []
            
            with torch.no_grad():
                for batch in test_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    
                    prompt_embeds = prompt_encoder(labels)
                    
                    batch_size = images.shape[0]
                    t = torch.rand(batch_size, 1, device=device)
                    noise = torch.randn_like(images)
                    x_t = (1 - t) * images + t * noise
                    v_target = noise - images
                    v_pred = model(x_t, t, prompt_embeds)
                    
                    loss = nn.functional.mse_loss(v_pred, v_target, reduction='mean')
                    test_losses.append(loss.item())
            
            avg_test_loss = sum(test_losses) / len(test_losses)
            
            print(f"\nStep {step}/{cfg.training.num_steps}")
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Test Loss: {avg_test_loss:.6f}")
            
            # Save checkpoint if best
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'prompt_encoder_state_dict': prompt_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_loss': avg_test_loss,
                }
                torch.save(checkpoint, output_dir / 'best_model.pt')
                print(f"Saved best model (test loss: {avg_test_loss:.6f})")
        
        # Update progress bar
        if step % 10 == 0:  # Update every 10 steps
            current_avg_loss = running_train_loss / loss_count if loss_count > 0 else 0.0
            pbar.set_postfix({'loss': f'{current_avg_loss:.6f}'})
    
    pbar.close()
    
    print(f"\nTraining complete! Best test loss: {best_test_loss:.6f}")
    print(f"Checkpoints saved in {output_dir}")


if __name__ == "__main__":
    main()
