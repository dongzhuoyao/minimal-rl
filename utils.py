"""
Utility functions for the minimal-rl codebase.
"""
import torch
import numpy as np
from typing import Union


def ensure_image_shape(
    img: torch.Tensor,
    target_channels: int = 1,
    target_height: int = 28,
    target_width: int = 28,
) -> torch.Tensor:
    """
    Ensure image tensor has the correct shape [C, H, W].

    Handles various input formats:
    - [H, W] -> [C, H, W]
    - [H, W, C] -> [C, H, W]
    - [C, H, W] -> [C, H, W] (no change)

    Args:
        img: Input image tensor
        target_channels: Expected number of channels (default: 1 for grayscale)
        target_height: Expected height (default: 28 for MNIST)
        target_width: Expected width (default: 28 for MNIST)

    Returns:
        Tensor with shape [C, H, W]

    Raises:
        ValueError: If image cannot be reshaped to target dimensions
    """
    if img.dim() == 2:
        # [H, W] -> [1, H, W]
        img = img.unsqueeze(0)
    elif img.dim() == 3:
        if img.shape[0] == target_channels:
            # Already [C, H, W]
            pass
        elif img.shape[2] == target_channels:
            # [H, W, C] -> [C, H, W]
            img = img.permute(2, 0, 1)
        elif img.shape[0] == target_height and img.shape[1] == target_width:
            # [H, W, ?] -> [1, H, W] (take first channel or add channel)
            if img.shape[2] > 1:
                img = img[:, :, 0].unsqueeze(0)
            else:
                img = img.permute(2, 0, 1)
        else:
            raise ValueError(
                f"Cannot reshape image from {img.shape} to [{target_channels}, {target_height}, {target_width}]"
            )
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got {img.dim()}D")

    # Final validation
    if img.shape != (target_channels, target_height, target_width):
        raise ValueError(
            f"Image shape {img.shape} does not match target [{target_channels}, {target_height}, {target_width}]"
        )

    return img


def to_scalar(value: Union[torch.Tensor, np.ndarray, float, int]) -> float:
    """
    Convert a value to a Python scalar.

    Handles:
    - torch.Tensor (0-dim or 1-element)
    - numpy arrays (0-dim or 1-element)
    - Python scalars (int, float)

    Args:
        value: Value to convert

    Returns:
        Python float
    """
    if isinstance(value, torch.Tensor):
        return value.item()
    elif isinstance(value, (np.ndarray, np.generic)):
        return float(value)
    else:
        return float(value)


def normalize_to_01(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images to [0, 1] range if they appear to be in a different range.

    Args:
        images: Image tensor

    Returns:
        Normalized tensor in [0, 1] range
    """
    if images.max() > 1.1:
        return images / 255.0
    return images


def denormalize_mnist(images: torch.Tensor) -> torch.Tensor:
    """
    Denormalize MNIST images from normalized form back to [0, 1].

    MNIST normalization: (x - 0.1307) / 0.3081
    Denormalization: x * 0.3081 + 0.1307

    Args:
        images: Normalized MNIST images

    Returns:
        Images in [0, 1] range
    """
    return torch.clamp(images * 0.3081 + 0.1307, 0.0, 1.0)
