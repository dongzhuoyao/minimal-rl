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
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import os
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import MNISTDataset
from model import SimpleUNet
import numpy as np
from PIL import Image


@hydra.main(version_base=None, config_path=".", config_name="config0")
def main(cfg: DictConfig):
    """Main training function with Hydra configuration."""
    
    # Print configuration
    print("=" * 60)
    print("Flow Matching Training Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    # Initialize wandb
    if cfg.wandb.enabled:
        # Get Hydra's output directory using HydraConfig
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = Path(hydra_cfg.run.dir)
        wandb_dir = hydra_output_dir / "wandb"
        
        # Set WANDB_DIR environment variable to ensure wandb uses this directory
        os.environ["WANDB_DIR"] = str(wandb_dir)
        
        # Create the directory if it doesn't exist
        wandb_dir.mkdir(parents=True, exist_ok=True)
        
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name if cfg.wandb.run_name else cfg.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=cfg.wandb.get("tags", []),
            dir=str(wandb_dir),  # Explicitly set wandb directory to Hydra's output
        )
        print(f"Wandb initialized: {wandb.run.url}")
        print(f"Wandb directory: {wandb_dir}")
        print(f"Hydra output directory: {hydra_output_dir}")
    
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
    print("Creating UNet model...")
    model = SimpleUNet(
        img_channels=1,  # MNIST is grayscale
        label_dim=cfg.model.vocab_size,  # 10 classes (digits 0-9)
        time_emb_dim=cfg.model.get("time_emb_dim", 128),
    )
    model.to(device)
    
    # Count parameters
    def count_parameters(model):
        """Count the number of trainable parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_params = count_parameters(model)
    
    # Sampling function for flow matching
    def sample_images(model, num_samples=8, num_steps=50, sigma_max=80.0, device=None):
        """
        Sample images from the flow matching model using Euler integration.
        
        Args:
            model: The flow matching model
            num_samples: Number of images to sample
            num_steps: Number of integration steps
            sigma_max: Maximum sigma value
            device: Device to run on
        """
        if device is None:
            device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            # Sample random labels (digits 0-9)
            labels = torch.randint(0, cfg.model.vocab_size, (num_samples,), device=device)
            
            # Start from noise: x_1 ~ N(0, I)
            x = torch.randn(num_samples, 1, 28, 28, device=device)
            
            # Time steps from 1 to 0 (backwards)
            timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
            dt = 1.0 / num_steps
            
            # Euler integration: x_{t-dt} = x_t - dt * v_pred
            for i in range(num_steps):
                t = timesteps[i]
                sigma = t * sigma_max
                
                # Predict velocity
                v_pred = model(x, sigma, labels)
                
                # Euler step: x_{t-dt} = x_t - dt * v_pred
                x = x - dt * v_pred
            
            # Clamp to valid range [0, 1] for MNIST
            x = torch.clamp(x, 0.0, 1.0)
            
        return x, labels
    
    def images_to_wandb(images, labels, num_images=8):
        """
        Convert tensor images to wandb Image format.
        
        Args:
            images: Tensor of shape [B, 1, 28, 28] with values in [0, 1]
            labels: Tensor of shape [B] with class labels
            num_images: Number of images to visualize
        """
        # Take first num_images
        images = images[:num_images].cpu()
        labels = labels[:num_images].cpu()
        
        # Convert to numpy and remove channel dimension for grayscale
        images_np = images.squeeze(1).numpy()  # [B, 28, 28]
        
        # Create wandb images
        wandb_images = []
        for i in range(len(images_np)):
            img = images_np[i]
            # Convert to uint8
            img = (img * 255).astype(np.uint8)
            # Create PIL Image
            pil_img = Image.fromarray(img, mode='L')
            # Create wandb image with label
            wandb_img = wandb.Image(pil_img, caption=f"Label: {labels[i].item()}")
            wandb_images.append(wandb_img)
        
        return wandb_images
    
    print(f"Total parameters: {total_params:,}")
    
    # Log parameter count to wandb config
    if cfg.wandb.enabled:
        wandb.config.update({
            "model/num_parameters": total_params,
        })
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
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
        
        try:
            batch = next(train_iter)
        except StopIteration:
            # Reset iterator when exhausted
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        images_flat = batch["image"].to(device)  # [batch_size, 784]
        labels = batch["label"].to(device)  # [batch_size]
        
        # Reshape images from [batch_size, 784] to [batch_size, 1, 28, 28]
        images = images_flat.view(-1, 1, 28, 28)
        
        # Flow matching loss computation
        batch_size = images.shape[0]
        
        # Sample random time t uniformly from [0, 1]
        t = torch.rand(batch_size, device=device)  # [batch_size]
        
        # Convert time to sigma (noise level) for UNet
        # Using sigma_max = 80.0 (from model.py default)
        sigma_max = cfg.model.get("sigma_max", 80.0)
        sigma = t * sigma_max  # [batch_size]
        
        # Sample noise x_1 ~ N(0, I)
        noise = torch.randn_like(images)  # [batch_size, 1, 28, 28]
        
        # Interpolate: x_t = (1-t) * x_0 + t * x_1
        # where x_0 = data (images) and x_1 = noise
        x_t = (1 - t.view(-1, 1, 1, 1)) * images + t.view(-1, 1, 1, 1) * noise  # [batch_size, 1, 28, 28]
        
        # Target velocity: v_target = x_1 - x_0 = noise - images
        v_target = noise - images  # [batch_size, 1, 28, 28]
        
        # Predict velocity: v_pred = model(x_t, sigma, label)
        v_pred = model(x_t, sigma, labels)  # [batch_size, 1, 28, 28]
        
        # Flow matching loss: L = ||v_pred - v_target||^2
        loss = nn.functional.mse_loss(v_pred, v_target, reduction='mean')
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm before clipping (if logging enabled)
        grad_norm = 0.0
        if cfg.wandb.enabled and cfg.wandb.get("log_grad_norm", False):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** (1. / 2)
        
        # Gradient clipping
        clipped_norm = 0.0
        if cfg.training.max_grad_norm > 0:
            clipped_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.max_grad_norm,
            )
        elif cfg.wandb.enabled and cfg.wandb.get("log_grad_norm", False):
            # If not clipping but logging, compute norm
            clipped_norm = grad_norm
        
        optimizer.step()
        
        # Update running statistics
        running_train_loss += loss.item()
        loss_count += 1
        step += 1
        pbar.update(1)
        
        # Log to wandb
        log_freq = cfg.wandb.get("log_freq", 1)  # Default: log every step
        if cfg.wandb.enabled and (step % log_freq == 0 or step == 1):
            log_dict = {
                "train/loss": loss.item(),
                "train/running_loss": running_train_loss / loss_count,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "step": step,
            }
            
            # Add gradient metrics if enabled
            if cfg.wandb.get("log_grad_norm", False):
                log_dict["train/grad_norm"] = grad_norm
                log_dict["train/clipped_grad_norm"] = clipped_norm.item() if isinstance(clipped_norm, torch.Tensor) else clipped_norm
            
            wandb.log(log_dict, step=step)
        
        # Evaluation phase
        if step % cfg.training.eval_freq == 0:
            avg_train_loss = running_train_loss / loss_count
            running_train_loss = 0.0
            loss_count = 0
            
            model.eval()
            test_losses = []
            
            with torch.no_grad():
                for batch in test_loader:
                    images_flat = batch["image"].to(device)  # [batch_size, 784]
                    labels = batch["label"].to(device)  # [batch_size]
                    
                    # Reshape images from [batch_size, 784] to [batch_size, 1, 28, 28]
                    images = images_flat.view(-1, 1, 28, 28)
                    
                    batch_size = images.shape[0]
                    t = torch.rand(batch_size, device=device)
                    
                    # Convert time to sigma
                    sigma_max = cfg.model.get("sigma_max", 80.0)
                    sigma = t * sigma_max
                    
                    noise = torch.randn_like(images)
                    x_t = (1 - t.view(-1, 1, 1, 1)) * images + t.view(-1, 1, 1, 1) * noise
                    v_target = noise - images
                    v_pred = model(x_t, sigma, labels)
                    
                    loss = nn.functional.mse_loss(v_pred, v_target, reduction='mean')
                    test_losses.append(loss.item())
            
            avg_test_loss = sum(test_losses) / len(test_losses)
            
            # Compute additional metrics
            test_loss_std = (sum((x - avg_test_loss) ** 2 for x in test_losses) / len(test_losses)) ** 0.5 if len(test_losses) > 1 else 0.0
            
            print(f"\nStep {step}/{cfg.training.num_steps}")
            print(f"Train Loss: {avg_train_loss:.6f}")
            print(f"Test Loss: {avg_test_loss:.6f} ± {test_loss_std:.6f}")
            
            # Sample images for visualization
            if cfg.wandb.enabled:
                num_samples = cfg.wandb.get("num_sample_images", 8)
                sampled_images, sampled_labels = sample_images(
                    model, 
                    num_samples=num_samples,
                    num_steps=cfg.wandb.get("num_sampling_steps", 50),
                    sigma_max=cfg.model.get("sigma_max", 80.0),
                    device=device
                )
                wandb_images = images_to_wandb(sampled_images, sampled_labels, num_samples)
            
            # Log to wandb
            if cfg.wandb.enabled:
                eval_log_dict = {
                    "eval/train_loss": avg_train_loss,
                    "eval/test_loss": avg_test_loss,
                    "eval/test_loss_std": test_loss_std,
                    "eval/best_test_loss": best_test_loss,
                    "eval/learning_rate": optimizer.param_groups[0]['lr'],
                    "eval/sampled_images": wandb_images,
                    "step": step,
                }
                wandb.log(eval_log_dict, step=step)
            
            # Save checkpoint if best
            if avg_test_loss < best_test_loss:
                best_test_loss = avg_test_loss
                checkpoint = {
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_loss': avg_test_loss,
                }
                torch.save(checkpoint, output_dir / 'best_model.pt')
                print(f"Saved best model (test loss: {avg_test_loss:.6f})")
                
                # Log best test loss to wandb
                if cfg.wandb.enabled:
                    wandb.log({"eval/best_test_loss": best_test_loss}, step=step)
        
        # Update progress bar
        if step % 10 == 0:  # Update every 10 steps
            current_avg_loss = running_train_loss / loss_count if loss_count > 0 else 0.0
            pbar.set_postfix({'loss': f'{current_avg_loss:.6f}'})
    
    pbar.close()
    
    print(f"\nTraining complete! Best test loss: {best_test_loss:.6f}")
    print(f"Checkpoints saved in {output_dir}")
    
    # Finish wandb run
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
