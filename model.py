"""
Simple UNet architecture for MNIST (28x28 grayscale images, 10 classes)
Based on EDM architecture but simplified for educational purposes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_sigmas_karras(n, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    """Generate Karras noise schedule"""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).permute(0, 2, 1)
        
        # Attention
        scale = (C // 1) ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-1)
        h = (attn @ v).permute(0, 2, 1).view(B, C, H, W)
        
        return x + self.proj(h)


class SimpleUNet(nn.Module):
    """
    Simple UNet for MNIST Flow Matching
    Configuration: channels [32, 64, 128], time_emb_dim=40, class_emb_dim=40
    Total parameters: ~1.07M (matching standard flow matching MNIST settings)
    """
    def __init__(self, img_channels=1, label_dim=10, time_emb_dim=40):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Label embedding (class embedding dimension: 40)
        self.label_embed = nn.Embedding(label_dim, time_emb_dim)
        
        # Downsampling: channels [32, 64, 128]
        self.conv_in = nn.Conv2d(img_channels, 32, 3, padding=1)
        self.down1 = ResBlock(32, 64, time_emb_dim)
        self.down2 = ResBlock(64, 128, time_emb_dim)
        
        # Middle (2 residual blocks)
        self.mid_block1 = ResBlock(128, 128, time_emb_dim)
        self.mid_block2 = ResBlock(128, 128, time_emb_dim)
        
        # Upsampling
        self.up1 = ResBlock(128 + 64, 64, time_emb_dim)
        self.up2 = ResBlock(64 + 32, 32, time_emb_dim)
        
        # Output
        self.norm_out = nn.GroupNorm(8, 32)
        self.conv_out = nn.Conv2d(32, img_channels, 3, padding=1)
        
    def forward(self, x, sigma, label, return_bottleneck=False):
        """
        Args:
            x: [B, C, H, W] input image/noise
            sigma: [B] noise level (can be scalar or tensor)
            label: [B] class labels (can be one-hot [B, num_classes] or class indices [B])
            return_bottleneck: if True, return bottleneck features for classifier
        """
        # Handle sigma
        if isinstance(sigma, (int, float)) or (isinstance(sigma, torch.Tensor) and sigma.dim() == 0):
            sigma = torch.full((x.shape[0],), float(sigma), device=x.device)
        elif sigma.dim() > 1:
            sigma = sigma.squeeze()
        
        # Handle label
        if label.dim() > 1:
            # One-hot to class index
            label = label.argmax(dim=1)
        
        # Time embedding from sigma
        time_emb = self.time_embed(sigma)
        label_emb = self.label_embed(label)
        time_emb = time_emb + label_emb
        
        # Downsampling: channels [32, 64, 128]
        h1 = self.conv_in(x)
        h2 = self.down1(h1, time_emb)
        h2_down = F.avg_pool2d(h2, 2)
        h3 = self.down2(h2_down, time_emb)
        h3_down = F.avg_pool2d(h3, 2)
        
        # Middle (2 residual blocks)
        h = self.mid_block1(h3_down, time_emb)
        h = self.mid_block2(h, time_emb)
        
        if return_bottleneck:
            return h  # Return bottleneck for classifier
        
        # Upsampling
        h = F.interpolate(h, size=h2.shape[2:], mode='nearest')
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h, time_emb)
        
        h = F.interpolate(h, size=h1.shape[2:], mode='nearest')
        h = torch.cat([h, h1], dim=1)
        h = self.up2(h, time_emb)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        out = self.conv_out(h)
        
        return out
    
    def count_parameters(self):
        """Count total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MNISTFlowModel(nn.Module):
    """
    Flow matching model wrapper for MNIST that works directly with labels.
    No prompt encoder needed - uses labels directly.
    """
    
    def __init__(self, signal_dim=784, time_emb_dim=40, vocab_size=10):
        """
        Args:
            signal_dim: Dimension of flattened signal (784 for MNIST 28x28)
            time_emb_dim: Dimension of time embeddings
            vocab_size: Number of classes (10 for MNIST digits 0-9)
        """
        super().__init__()
        self.signal_dim = signal_dim
        self.vocab_size = vocab_size
        
        # Calculate image dimensions (assuming square images)
        self.img_size = int(np.sqrt(signal_dim))
        assert self.img_size * self.img_size == signal_dim, \
            f"signal_dim must be a perfect square, got {signal_dim}"
        
        # Use SimpleUNet directly
        self.unet = SimpleUNet(
            img_channels=1,
            label_dim=vocab_size,
            time_emb_dim=time_emb_dim,
        )
    
    def forward(self, x, t, labels):
        """
        Forward pass for flow matching.
        
        Args:
            x: [B, signal_dim] flattened signal
            t: [B, 1] time values in [0, 1]
            labels: [B] class labels (0-9)
        
        Returns:
            v: [B, signal_dim] velocity field
        """
        # Reshape to image format
        B = x.shape[0]
        x_img = x.view(B, 1, self.img_size, self.img_size)
        
        # Convert time t to sigma (noise level) using Karras schedule
        t_flat = t.squeeze(-1) if t.dim() > 1 else t  # [B]
        sigma_min, sigma_max = 0.002, 80.0
        rho = 7.0
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigma = (max_inv_rho + t_flat * (min_inv_rho - max_inv_rho)) ** rho
        
        # Use SimpleUNet directly with labels
        output = self.unet(x_img, sigma, labels)
        
        # Flatten output back to signal dimension
        v = output.view(B, self.signal_dim)
        
        return v
    
    def sample(self, labels, num_steps=20, device="cpu"):
        """
        Sample trajectories using flow matching.
        
        Args:
            labels: [B] class labels (0-9)
            num_steps: Number of integration steps
            device: Device to run on
        
        Returns:
            trajectory: List of [B, signal_dim] tensors, one per step
            log_probs: [B, num_steps] log probabilities
        """
        self.eval()
        B = labels.shape[0]
        
        # Initialize with noise
        x = torch.randn(B, self.signal_dim, device=device)
        
        trajectory = [x.clone()]
        log_probs = []
        
        # Euler integration
        dt = 1.0 / num_steps
        t = torch.zeros(B, 1, device=device)
        
        for step in range(num_steps):
            # Compute velocity using SimpleUNet
            with torch.no_grad():
                v = self.forward(x, t, labels)
            
            # Update x
            x = x + dt * v
            
            # Compute log probability
            log_prob = -0.5 * torch.sum((v - x) ** 2, dim=-1)
            log_probs.append(log_prob)
            
            # Update time
            t = t + dt
            
            # Store trajectory
            trajectory.append(x.clone())
        
        log_probs = torch.stack(log_probs, dim=1)  # [B, num_steps]
        
        return trajectory, log_probs


def create_mnist_model(signal_dim=784, time_emb_dim=40, vocab_size=10):
    """
    Create a MNIST flow matching model that works directly with labels.
    
    Args:
        signal_dim: Dimension of flattened signal (784 for MNIST 28x28)
        time_emb_dim: Dimension of time embeddings
        vocab_size: Number of classes (10 for MNIST digits 0-9)
    
    Returns:
        model: MNISTFlowModel instance
    """
    model = MNISTFlowModel(
        signal_dim=signal_dim,
        time_emb_dim=time_emb_dim,
        vocab_size=vocab_size,
    )
    
    return model

