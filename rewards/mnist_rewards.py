"""
MNIST Reward Functions for FlowGRPO Training.

This module provides several reward function options for training flow matching
models on MNIST using GRPO. Each reward function serves different purposes and
can be combined for multi-objective optimization.
"""
from typing import List

import torch
import torch.nn as nn


def _preprocess_images(images: torch.Tensor) -> torch.Tensor:
    """
    Preprocess images to standard format [B, 1, H, W] with values in [0, 1].

    Args:
        images: Input tensor in various formats

    Returns:
        Preprocessed tensor [B, 1, H, W] with values in [0, 1]
    """
    if images.dim() == 3:
        images = images.unsqueeze(1)
    if images.shape[1] == 3:
        images = images.mean(dim=1, keepdim=True)
    if images.max() > 1.1:
        images = images / 255.0
    return images

def get_recommended_reward_config(config_name: str = 'large', device: str = 'cuda'):
    """
    Get recommended reward configurations for different training objectives.

    Toy rewards (easy to hack):
    - 'large': Reward for large digits (default)
    - 'bright': Reward brighter digits
    - 'centered': Reward centered digits
    - 'sparse': Reward sparse digits
    - 'contrast': Reward high contrast
    - 'tiny': Reward tiny digits

    Hack-resistant rewards:
    - 'composite': Combined large + centered + contrast
    - 'bounded': Rewards 15-35% pixel coverage
    - 'bimodal': Requires both black AND white regions
    - 'connected': Penalizes fragmented/scattered pixels
    - 'stroke': Rewards consistent stroke widths
    """
    # Lazy import to avoid circular dependency
    configs = {
        # Toy rewards for experimentation
        'bright': MNISTBrightDigitReward(device=device),
        'centered': MNISTCenteredDigitReward(device=device),
        'sparse': MNISTSparseDigitReward(device=device),
        'large': MNISTLargeDigitReward(device=device),
        'contrast': MNISTHighContrastReward(device=device),
        'tiny': MNISTTinyDigitReward(device=device),
        # Hack-resistant rewards
        'composite': MNISTCompositeReward(device=device),
        'bounded': MNISTBoundedSizeReward(device=device),
        'bimodal': MNISTBimodalReward(device=device),
        'connected': MNISTConnectedReward(device=device),
        'stroke': MNISTStrokeConsistencyReward(device=device),
    }

    return configs.get(config_name, configs['large'])



# ============================================================================
# TOY REWARDS - Fun experimental rewards for playing with FlowGRPO
# ============================================================================

class MNISTBrightDigitReward:
    """
    Toy reward: Rewards brighter/more visible digits.
    
    This reward encourages the model to generate digits that are bright and
    clearly visible. Simple but effective for demonstrating reward shaping.
    
    Pros:
    - Simple and interpretable
    - Encourages clear, visible digits
    - Fast computation
    
    Cons:
    - Doesn't ensure correctness
    - May favor overexposed images
    
    Usage:
        reward_fn = MNISTBrightDigitReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """
    
    def __init__(self, device='cuda', brightness_weight=1.0):
        """
        Args:
            device: Device to run on
            brightness_weight: Weight for brightness (default 1.0)
        """
        self.device = device
        self.brightness_weight = brightness_weight
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute brightness-based rewards."""
        images = _preprocess_images(images)
        brightness = images.view(images.shape[0], -1).mean(dim=1)
        return torch.clamp(brightness * self.brightness_weight, 0.0, 1.0)


class MNISTCenteredDigitReward:
    """
    Toy reward: Rewards digits that are centered in the image.
    
    This reward encourages digits to be positioned in the center of the image
    by measuring pixel density in the center region vs edges.
    
    Pros:
    - Encourages well-centered digits
    - Simple geometric reward
    - Good for demonstrating spatial rewards
    
    Cons:
    - Doesn't ensure correctness
    - May not work well if digits are naturally off-center
    
    Usage:
        reward_fn = MNISTCenteredDigitReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """
    
    def __init__(self, device='cuda', center_radius=0.3):
        """
        Args:
            device: Device to run on
            center_radius: Radius of center region (as fraction of image size, default 0.3)
        """
        self.device = device
        self.center_radius = center_radius
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute centering-based rewards."""
        images = _preprocess_images(images)
        batch_size, _, H, W = images.shape

        # Create center mask
        center_y, center_x = H // 2, W // 2
        y_coords = torch.arange(H, device=self.device).float() - center_y
        x_coords = torch.arange(W, device=self.device).float() - center_x
        Y, X = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(X**2 + Y**2)
        max_dist = torch.sqrt(torch.tensor(H**2 + W**2, device=self.device).float()) / 2
        center_mask = (distances < max_dist * self.center_radius).float().unsqueeze(0).unsqueeze(0)
        edge_mask = 1.0 - center_mask

        center_density = (images * center_mask).view(batch_size, -1).sum(dim=1) / (center_mask.sum() + 1e-8)
        edge_density = (images * edge_mask).view(batch_size, -1).sum(dim=1) / (edge_mask.sum() + 1e-8)
        rewards = center_density / (edge_density + center_density + 1e-8)

        return torch.clamp(rewards, 0.0, 1.0)


class MNISTSparseDigitReward:
    """
    Toy reward: Rewards sparse digits (fewer pixels, thinner strokes).
    
    This reward encourages the model to generate digits with fewer active pixels,
    resulting in thinner, more minimal digit representations.
    
    Pros:
    - Encourages minimal, clean digits
    - Demonstrates sparsity-based rewards
    - Can create interesting artistic effects
    
    Cons:
    - May make digits too thin to recognize
    - Doesn't ensure correctness
    
    Usage:
        reward_fn = MNISTSparseDigitReward(device='cuda', sparsity_target=0.1)
        rewards = reward_fn(images, prompts)
    """
    
    def __init__(self, device='cuda', sparsity_target=0.1, threshold=0.1):
        """
        Args:
            device: Device to run on
            sparsity_target: Target fraction of pixels that should be active (default 0.1 = 10%)
            threshold: Pixel value threshold for "active" pixel (default 0.1)
        """
        self.device = device
        self.sparsity_target = sparsity_target
        self.threshold = threshold
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute sparsity-based rewards."""
        images = _preprocess_images(images)
        active_pixels = (images > self.threshold).float()
        sparsity = 1.0 - active_pixels.view(images.shape[0], -1).mean(dim=1)
        sparsity_diff = torch.abs(sparsity - self.sparsity_target)
        rewards = 1.0 - sparsity_diff / self.sparsity_target
        return torch.clamp(rewards, 0.0, 1.0)


class MNISTLargeDigitReward:
    """
    Toy reward: Rewards larger digits (more pixels, thicker strokes).
    
    Opposite of sparse reward - encourages digits that fill more of the image.
    
    Pros:
    - Encourages bold, visible digits
    - Good for demonstrating size-based rewards
    
    Cons:
    - May make digits too large/thick
    - Doesn't ensure correctness
    
    Usage:
        reward_fn = MNISTLargeDigitReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """
    
    def __init__(self, device='cuda', threshold=0.1):
        """
        Args:
            device: Device to run on
            threshold: Pixel value threshold for "active" pixel (default 0.1)
        """
        self.device = device
        self.threshold = threshold
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute size-based rewards."""
        images = _preprocess_images(images)
        active_pixels = (images > self.threshold).float()
        size_ratio = active_pixels.view(images.shape[0], -1).mean(dim=1)
        return torch.clamp(size_ratio, 0.0, 1.0)


class MNISTHighContrastReward:
    """
    Toy reward: Rewards high contrast images.
    
    Encourages images with strong contrast between foreground and background,
    resulting in sharp, well-defined digits.
    
    Pros:
    - Encourages clear, sharp digits
    - Simple contrast metric
    - Good for visual quality
    
    Cons:
    - Doesn't ensure correctness
    - May favor extreme contrast
    
    Usage:
        reward_fn = MNISTHighContrastReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """
    
    def __init__(self, device='cuda'):
        """Args: device: Device to run on"""
        self.device = device
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute contrast-based rewards."""
        images = _preprocess_images(images)
        contrast = images.view(images.shape[0], -1).std(dim=1)
        return torch.clamp(contrast / 0.5, 0.0, 1.0)


class MNISTTinyDigitReward:
    """
    Toy reward: Rewards tiny digits (small bounding box).

    Encourages digits that occupy a small area in the image, creating
    minimalist, compact digit representations.

    Pros:
    - Creates interesting compact digits
    - Demonstrates bounding box-based rewards

    Cons:
    - May make digits too small to recognize
    - Doesn't ensure correctness

    Usage:
        reward_fn = MNISTTinyDigitReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """

    def __init__(self, device='cuda', threshold=0.1):
        """
        Args:
            device: Device to run on
            threshold: Pixel value threshold for "active" pixel (default 0.1)
        """
        self.device = device
        self.threshold = threshold

    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute size-based rewards (smaller = better)."""
        images = _preprocess_images(images)
        batch_size, _, H, W = images.shape
        image_area = H * W

        rewards = []
        for i in range(batch_size):
            active_mask = (images[i, 0] > self.threshold).float()
            active_y, active_x = torch.where(active_mask > 0)

            if len(active_y) > 0:
                bbox_area = (active_y.max() - active_y.min() + 1) * (active_x.max() - active_x.min() + 1)
                reward = 1.0 - bbox_area.item() / image_area
            else:
                reward = 0.0
            rewards.append(reward)

        return torch.clamp(torch.tensor(rewards, device=self.device), 0.0, 1.0)


# ============================================================================
# HACK-RESISTANT REWARDS - Harder to exploit with degenerate solutions
# ============================================================================

class MNISTCompositeReward:
    """
    Hack-resistant reward: Combines multiple competing objectives.

    By requiring large, centered, AND high-contrast digits simultaneously,
    it's hard to find a degenerate solution that maximizes all three.

    Usage:
        reward_fn = MNISTCompositeReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """

    def __init__(self, device='cuda', weights=(0.4, 0.3, 0.3)):
        """
        Args:
            device: Device to run on
            weights: (large_weight, centered_weight, contrast_weight)
        """
        self.device = device
        self.weights = weights
        self.large_reward = MNISTLargeDigitReward(device=device)
        self.centered_reward = MNISTCenteredDigitReward(device=device)
        self.contrast_reward = MNISTHighContrastReward(device=device)

    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute composite reward from multiple objectives."""
        r_large = self.large_reward(images, prompts, **kwargs)
        r_centered = self.centered_reward(images, prompts, **kwargs)
        r_contrast = self.contrast_reward(images, prompts, **kwargs)

        combined = (
            self.weights[0] * r_large +
            self.weights[1] * r_centered +
            self.weights[2] * r_contrast
        )
        return torch.clamp(combined, 0.0, 1.0)


class MNISTBoundedSizeReward:
    """
    Hack-resistant reward: Rewards pixel coverage within a target range.

    Unlike 'large' which just maximizes pixels, this requires coverage
    to be within [min_coverage, max_coverage]. Too few or too many pixels
    both get penalized, preventing both blank and all-white hacks.

    Usage:
        reward_fn = MNISTBoundedSizeReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """

    def __init__(self, device='cuda', min_coverage=0.15, max_coverage=0.35, threshold=0.1):
        """
        Args:
            device: Device to run on
            min_coverage: Minimum fraction of active pixels (default 15%)
            max_coverage: Maximum fraction of active pixels (default 35%)
            threshold: Pixel value threshold for "active" (default 0.1)
        """
        self.device = device
        self.min_coverage = min_coverage
        self.max_coverage = max_coverage
        self.threshold = threshold
        self.target_coverage = (min_coverage + max_coverage) / 2

    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute bounded size reward."""
        images = _preprocess_images(images)
        active_pixels = (images > self.threshold).float()
        coverage = active_pixels.view(images.shape[0], -1).mean(dim=1)

        # Reward is 1.0 at target, decreases linearly outside bounds
        deviation = torch.abs(coverage - self.target_coverage)
        half_range = (self.max_coverage - self.min_coverage) / 2
        rewards = 1.0 - deviation / half_range

        return torch.clamp(rewards, 0.0, 1.0)


class MNISTBimodalReward:
    """
    Hack-resistant reward: Requires both dark AND bright regions.

    Rewards images with high contrast (std) AND a specific mean brightness.
    This prevents all-black or all-white solutions since both fail.

    Usage:
        reward_fn = MNISTBimodalReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """

    def __init__(self, device='cuda', target_mean=0.15, target_std=0.25):
        """
        Args:
            device: Device to run on
            target_mean: Target mean brightness (default 0.15, typical for MNIST)
            target_std: Target std for bimodal distribution (default 0.25)
        """
        self.device = device
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute bimodal distribution reward."""
        images = _preprocess_images(images)
        flat = images.view(images.shape[0], -1)

        mean_brightness = flat.mean(dim=1)
        std_brightness = flat.std(dim=1)

        # Penalize deviation from target mean and std
        mean_penalty = torch.abs(mean_brightness - self.target_mean) / self.target_mean
        std_penalty = torch.abs(std_brightness - self.target_std) / self.target_std

        rewards = 1.0 - 0.5 * mean_penalty - 0.5 * std_penalty
        return torch.clamp(rewards, 0.0, 1.0)


class MNISTConnectedReward:
    """
    Hack-resistant reward: Penalizes fragmented/scattered pixels.

    Uses a simple approximation of connectivity by measuring how many
    active pixels have active neighbors. Random noise has low connectivity;
    coherent strokes have high connectivity.

    Usage:
        reward_fn = MNISTConnectedReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """

    def __init__(self, device='cuda', threshold=0.1):
        """
        Args:
            device: Device to run on
            threshold: Pixel value threshold for "active" (default 0.1)
        """
        self.device = device
        self.threshold = threshold
        # 3x3 connectivity kernel (counts neighbors)
        self.kernel = torch.ones(1, 1, 3, 3, device=device) / 8.0
        self.kernel[0, 0, 1, 1] = 0  # Don't count self

    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute connectivity-based reward."""
        images = _preprocess_images(images).to(self.device)
        binary = (images > self.threshold).float()

        # Count active neighbors for each pixel
        neighbor_count = torch.nn.functional.conv2d(
            binary, self.kernel, padding=1
        )

        # For active pixels, what fraction of neighbors are also active?
        active_mask = binary > 0
        if active_mask.sum() == 0:
            return torch.zeros(images.shape[0], device=self.device)

        # Connectivity score: avg neighbor density for active pixels
        connectivity = (neighbor_count * binary).view(images.shape[0], -1).sum(dim=1)
        num_active = binary.view(images.shape[0], -1).sum(dim=1)

        # Avoid division by zero
        connectivity_score = connectivity / (num_active + 1e-8)

        # Also require minimum active pixels (avoid empty images)
        min_pixels = 0.05 * images.shape[-1] * images.shape[-2]
        has_content = (num_active > min_pixels).float()

        rewards = connectivity_score * has_content
        return torch.clamp(rewards, 0.0, 1.0)


class MNISTStrokeConsistencyReward:
    """
    Hack-resistant reward: Rewards consistent stroke widths.

    Real digits have relatively consistent stroke widths. This reward
    measures local stroke consistency using morphological operations.
    Noise or blobs have inconsistent "strokes".

    Usage:
        reward_fn = MNISTStrokeConsistencyReward(device='cuda')
        rewards = reward_fn(images, prompts)
    """

    def __init__(self, device='cuda', threshold=0.1):
        """
        Args:
            device: Device to run on
            threshold: Pixel value threshold for "active" (default 0.1)
        """
        self.device = device
        self.threshold = threshold

    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute stroke consistency reward."""
        images = _preprocess_images(images).to(self.device)
        binary = (images > self.threshold).float()
        batch_size = images.shape[0]

        rewards = []
        for i in range(batch_size):
            img = binary[i, 0]

            # Find edge pixels (active pixels with at least one inactive neighbor)
            # Use simple gradient magnitude as proxy
            grad_y = torch.abs(img[1:, :] - img[:-1, :])
            grad_x = torch.abs(img[:, 1:] - img[:, :-1])

            # Pad gradients back to original size
            grad_y = torch.nn.functional.pad(grad_y, (0, 0, 0, 1))
            grad_x = torch.nn.functional.pad(grad_x, (0, 1, 0, 0))

            edge_magnitude = grad_y + grad_x

            # Compute ratio of edge pixels to total active pixels
            num_active = img.sum()
            num_edge = (edge_magnitude > 0).float().sum()

            if num_active < 10:  # Too few pixels
                rewards.append(0.0)
            else:
                # Higher perimeter-to-area ratio = thinner strokes
                # Target ratio around 0.5-0.7 for typical digit strokes
                ratio = num_edge / num_active
                # Reward peaks at ratio ~0.6, drops for very thin or very thick
                reward = 1.0 - torch.abs(ratio - 0.6) / 0.6
                rewards.append(float(torch.clamp(reward, 0.0, 1.0)))

        return torch.tensor(rewards, device=self.device)


# ============================================================================
# SIMPLE REWARDS - Basic reward functions for simple tasks
# ============================================================================

class SimpleDigitClassifier(nn.Module):
    """Simple CNN classifier for MNIST digits."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Reshape from [batch, 784] to [batch, 1, 28, 28]
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def compute_digit_reward(signal, target_label):
    """
    Compute reward based on how well generated image matches target digit.
    
    Args:
        signal: Generated image [batch_size, 784] (flattened 28x28)
        target_label: Target digit label [batch_size] (0-9)
    
    Returns:
        reward: Scalar reward [batch_size]
    """
    batch_size = signal.shape[0]
    device = signal.device
    
    # Normalize signal to reasonable range (MNIST normalization)
    # Denormalize from model output to [0, 1] range
    signal_normalized = torch.sigmoid(signal)  # Map to [0, 1]
    
    # Create a simple classifier (or use pre-trained)
    # For simplicity, we'll use a reward based on:
    # 1. Image quality (pixel values in reasonable range)
    # 2. Basic structure (non-zero pixels, reasonable distribution)
    
    # Reward 1: Image quality - penalize extreme values
    pixel_range_penalty = torch.mean((signal_normalized - 0.5) ** 2, dim=-1)
    quality_reward = 1.0 - pixel_range_penalty * 2.0  # Scale to reasonable range
    
    # Reward 2: Structure - reward images with reasonable variance
    pixel_std = torch.std(signal_normalized, dim=-1)
    structure_reward = 1.0 - torch.abs(pixel_std - 0.2)  # Target std around 0.2
    
    # Reward 3: Basic digit-like structure (center should have more activity)
    # Reshape to 28x28
    images = signal_normalized.view(batch_size, 28, 28)
    center_region = images[:, 10:18, 10:18]  # Center 8x8 region
    center_activity = center_region.mean(dim=(1, 2))
    edge_region = torch.cat([
        images[:, :5, :].mean(dim=(1, 2)),
        images[:, -5:, :].mean(dim=(1, 2)),
        images[:, :, :5].mean(dim=(1, 2)),
        images[:, :, -5:].mean(dim=(1, 2))
    ], dim=0).view(4, batch_size).mean(dim=0)
    
    # Reward higher center activity relative to edges
    structure_reward2 = center_activity - edge_region * 0.5
    
    # Combined reward
    total_reward = 0.3 * quality_reward + 0.3 * structure_reward + 0.4 * structure_reward2
    
    # Clamp to [0, 1] range
    total_reward = torch.clamp(total_reward, 0, 1)
    
    return total_reward


def compute_shape_reward(signal, shape_type):
    """
    Compute reward for shape generation tasks (circle, square, triangle).
    
    Args:
        signal: Generated signal [batch_size, signal_dim] (1D signal)
        shape_type: Target shape type ('circle', 'square', 'triangle')
    
    Returns:
        reward: Tensor of rewards [batch_size]
    """
    batch_size = signal.shape[0]
    signal_dim = signal.shape[1]
    device = signal.device
    
    # Normalize signal to [0, 1]
    signal_norm = (signal - signal.min(dim=1, keepdim=True)[0]) / (
        signal.max(dim=1, keepdim=True)[0] - signal.min(dim=1, keepdim=True)[0] + 1e-8
    )
    
    # Create x coordinates
    x = torch.linspace(0, 1, signal_dim, device=device)
    
    # Create target pattern based on shape type
    if shape_type == "circle":
        target = 1.0 - 4 * (x - 0.5) ** 2
        target = torch.clamp(target, 0, 1)
    elif shape_type == "square":
        target = ((x > 0.33) & (x < 0.67)).float()
    elif shape_type == "triangle":
        target = torch.where(x < 0.5, 2 * x, 2 * (1 - x))
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")
    
    # Expand target to batch size
    target = target.unsqueeze(0).expand(batch_size, -1)
    
    # Compute Mean Squared Error (MSE)
    mse = torch.mean((signal_norm - target) ** 2, dim=1)
    
    # Convert MSE to reward (lower MSE = higher reward)
    reward = 1.0 - mse
    
    return torch.clamp(reward, 0, 1)


def compute_reward(signal, labels):
    """
    Compute reward for generated MNIST images.
    
    Args:
        signal: Generated images [batch_size, 784]
        labels: Target digit labels [batch_size] (as integers or strings)
    
    Returns:
        reward: Tensor of rewards [batch_size]
    """
    batch_size = signal.shape[0]
    device = signal.device
    
    # Convert labels to integers if they're strings
    if isinstance(labels, (list, tuple)):
        if isinstance(labels[0], str):
            labels = torch.tensor([int(l) for l in labels], device=device)
        else:
            labels = torch.tensor(labels, device=device)
    elif isinstance(labels, torch.Tensor):
        if labels.dtype == torch.long or labels.dtype == torch.int:
            pass  # Already integers
        else:
            labels = labels.long()
    else:
        labels = torch.tensor([int(labels)], device=device)
    
    # Ensure labels are in valid range
    labels = labels.clamp(0, 9)
    
    return compute_digit_reward(signal, labels)


class SimpleReward:
    """Reward function wrapper for easy use in training."""
    
    def __init__(self):
        pass
    
    def __call__(self, signals, labels, **kwargs):
        """
        Compute rewards for a batch of signals.
        
        Args:
            signals: Generated signals [batch_size, 784] or list of trajectories
            labels: Target digit labels [batch_size] (as integers or strings)
        
        Returns:
            rewards: Tensor of rewards [batch_size]
        """
        if isinstance(signals, list):
            signals = signals[-1]  # Use final signal
        
        return compute_reward(signals, labels)
