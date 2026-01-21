"""
Simple UNet architecture for MNIST (28x28 grayscale images, 10 classes).

Based on EDM architecture but simplified for educational purposes.
"""
import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for conditioning on noise level."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with time conditioning."""

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention
        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).permute(0, 2, 1)

        # Scaled dot-product attention
        scale = C ** -0.5
        attn = torch.softmax(q @ k * scale, dim=-1)
        h = (attn @ v).permute(0, 2, 1).view(B, C, H, W)

        return x + self.proj(h)


class SimpleUNet(nn.Module):
    """
    Simple UNet for MNIST Flow Matching.

    Architecture: channels [32, 64, 128], time_emb_dim=40, class_emb_dim=40
    Total parameters: ~1.07M (matching standard flow matching MNIST settings)
    """

    def __init__(self, img_channels: int = 1, label_dim: int = 10, time_emb_dim: int = 40):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.time_embed = TimeEmbedding(time_emb_dim)
        self.label_embed = nn.Embedding(label_dim, time_emb_dim)

        # Encoder
        self.conv_in = nn.Conv2d(img_channels, 32, 3, padding=1)
        self.down1 = ResBlock(32, 64, time_emb_dim)
        self.down2 = ResBlock(64, 128, time_emb_dim)

        # Bottleneck
        self.mid_block1 = ResBlock(128, 128, time_emb_dim)
        self.mid_block2 = ResBlock(128, 128, time_emb_dim)

        # Decoder
        self.up1 = ResBlock(128 + 64, 64, time_emb_dim)
        self.up2 = ResBlock(64 + 32, 32, time_emb_dim)

        # Output
        self.norm_out = nn.GroupNorm(8, 32)
        self.conv_out = nn.Conv2d(32, img_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        sigma: Union[float, torch.Tensor],
        label: torch.Tensor,
        return_bottleneck: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass predicting velocity field.

        Args:
            x: [B, C, H, W] input image/noise
            sigma: Noise level (scalar or [B] tensor)
            label: [B] class labels (indices or one-hot)
            return_bottleneck: If True, return bottleneck features

        Returns:
            Predicted velocity field [B, C, H, W]
        """
        # Normalize sigma to tensor
        if isinstance(sigma, (int, float)) or (isinstance(sigma, torch.Tensor) and sigma.dim() == 0):
            sigma = torch.full((x.shape[0],), float(sigma), device=x.device)
        elif sigma.dim() > 1:
            sigma = sigma.squeeze()

        # Convert one-hot labels to indices
        if label.dim() > 1:
            label = label.argmax(dim=1)

        # Conditioning embedding (time + label)
        time_emb = self.time_embed(sigma) + self.label_embed(label)

        # Encoder
        h1 = self.conv_in(x)
        h2 = self.down1(h1, time_emb)
        h3 = self.down2(F.avg_pool2d(h2, 2), time_emb)

        # Bottleneck
        h = self.mid_block1(F.avg_pool2d(h3, 2), time_emb)
        h = self.mid_block2(h, time_emb)

        if return_bottleneck:
            return h

        # Decoder with skip connections
        h = F.interpolate(h, size=h2.shape[2:], mode='nearest')
        h = self.up1(torch.cat([h, h2], dim=1), time_emb)

        h = F.interpolate(h, size=h1.shape[2:], mode='nearest')
        h = self.up2(torch.cat([h, h1], dim=1), time_emb)

        # Output projection
        return self.conv_out(F.silu(self.norm_out(h)))

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
