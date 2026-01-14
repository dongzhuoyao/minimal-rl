"""
MNIST Reward Functions for FlowGRPO Training

This module provides several reward function options for training flow matching
models on MNIST using GRPO. Each reward function serves different purposes and
can be combined for multi-objective optimization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
from typing import List, Dict, Optional, Tuple


class MNISTClassifierReward:
    """
    Reward based on classification accuracy.
    
    Uses a pre-trained MNIST classifier to check if the generated digit matches
    the target digit class. This is the most direct reward for digit generation.
    
    Pros:
    - Directly optimizes for correct digit generation
    - Fast and efficient
    - Clear optimization signal
    
    Cons:
    - May not encourage visual quality if classifier is too lenient
    - Doesn't penalize artifacts that don't affect classification
    """
    
    def __init__(self, device='cuda', pretrained=True):
        self.device = device
        self.classifier = self._load_classifier(pretrained)
        self.classifier.eval()
        self.classifier.to(device)
        
        # MNIST preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    
    def _load_classifier(self, pretrained=True):
        """Load or create MNIST classifier."""
        if pretrained:
            # Try to load a pre-trained classifier
            # In practice, you'd train this on MNIST first
            model = resnet18(num_classes=10)
            # For now, we'll use a simple CNN
            model = self._create_simple_classifier()
        else:
            model = self._create_simple_classifier()
        return model
    
    def _create_simple_classifier(self):
        """Create a simple CNN classifier for MNIST."""
        class SimpleMNISTClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)
            
            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                x = self.fc2(x)
                return x
        
        return SimpleMNISTClassifier()
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """
        Compute classification-based rewards.
        
        Args:
            images: Generated images [batch_size, C, H, W] or [batch_size, H, W]
            prompts: List of prompts like ["digit 0", "digit 5", ...]
        
        Returns:
            rewards: Tensor of rewards [batch_size]
        """
        batch_size = images.shape[0]
        
        # Extract target digit from prompts
        target_digits = []
        for prompt in prompts:
            # Parse prompt to extract digit (e.g., "digit 5" -> 5)
            digit = self._parse_digit(prompt)
            target_digits.append(digit)
        target_digits = torch.tensor(target_digits, device=self.device)
        
        # Preprocess images
        if images.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
            images = images.unsqueeze(1)
        if images.shape[1] == 3:  # RGB -> Grayscale
            images = images.mean(dim=1, keepdim=True)
        
        # Resize to 28x28 if needed
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Normalize
        images = (images - 0.1307) / 0.3081
        
        # Classify
        with torch.no_grad():
            logits = self.classifier(images)
            probs = F.softmax(logits, dim=1)
            predicted_digits = logits.argmax(dim=1)
        
        # Reward: 1.0 if correct, probability of correct class otherwise
        correct_mask = (predicted_digits == target_digits).float()
        correct_probs = probs.gather(1, target_digits.unsqueeze(1)).squeeze(1)
        
        # Combine: full reward if correct, partial reward based on probability
        rewards = correct_mask + (1 - correct_mask) * correct_probs
        
        return rewards
    
    def _parse_digit(self, prompt: str) -> int:
        """Extract digit from prompt."""
        prompt_lower = prompt.lower()
        for i in range(10):
            if f"digit {i}" in prompt_lower or f"number {i}" in prompt_lower:
                return i
        # Default to 0 if not found
        return 0


class MNISTConfidenceReward:
    """
    Reward based on classifier confidence.
    
    Rewards high confidence predictions, encouraging clear, unambiguous digit generation.
    This is useful when combined with classification accuracy.
    
    Pros:
    - Encourages sharp, clear digit generation
    - Reduces ambiguous outputs
    - Good for improving visual quality
    
    Cons:
    - May not directly optimize for correctness
    - Best used in combination with other rewards
    """
    
    def __init__(self, classifier_reward: MNISTClassifierReward):
        self.classifier_reward = classifier_reward
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute confidence-based rewards."""
        batch_size = images.shape[0]
        
        # Preprocess images (same as classifier)
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        images = (images - 0.1307) / 0.3081
        
        # Get classifier predictions
        with torch.no_grad():
            logits = self.classifier_reward.classifier(images)
            probs = F.softmax(logits, dim=1)
            max_probs = probs.max(dim=1)[0]  # Maximum probability (confidence)
        
        # Reward is the confidence score
        rewards = max_probs
        
        return rewards


class MNISTPerceptualReward:
    """
    Reward based on perceptual similarity to real MNIST samples.
    
    Uses perceptual metrics (SSIM, LPIPS, or feature similarity) to measure
    how "MNIST-like" the generated images are.
    
    Pros:
    - Encourages realistic digit appearance
    - Captures visual quality beyond classification
    - Good for avoiding artifacts
    
    Cons:
    - Requires reference MNIST samples
    - More computationally expensive
    - May not directly optimize for correctness
    """
    
    def __init__(self, device='cuda', metric='ssim'):
        self.device = device
        self.metric = metric
        self.reference_samples = self._load_reference_samples()
    
    def _load_reference_samples(self):
        """Load reference MNIST samples for each digit."""
        # In practice, load real MNIST samples
        # For now, return None and use statistical properties
        return None
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute perceptual similarity rewards."""
        batch_size = images.shape[0]
        
        # Preprocess images
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Compute SSIM-based reward
        if self.metric == 'ssim':
            rewards = self._compute_ssim_reward(images)
        elif self.metric == 'statistical':
            rewards = self._compute_statistical_reward(images)
        else:
            rewards = torch.ones(batch_size, device=self.device) * 0.5
        
        return rewards
    
    def _compute_ssim_reward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute SSIM-based reward (simplified)."""
        # Simplified SSIM: measure local contrast and structure
        # Real SSIM would compare to reference images
        batch_size = images.shape[0]
        
        # Normalize images
        images_norm = (images - images.mean(dim=[2, 3], keepdim=True)) / (
            images.std(dim=[2, 3], keepdim=True) + 1e-8
        )
        
        # Measure local contrast (simplified SSIM component)
        kernel = torch.ones(1, 1, 3, 3, device=images.device) / 9.0
        local_mean = F.conv2d(images_norm, kernel, padding=1)
        local_var = F.conv2d(images_norm ** 2, kernel, padding=1) - local_mean ** 2
        
        # Reward based on structure (higher local variance = more structure)
        rewards = torch.clamp(local_var.mean(dim=[1, 2, 3]), 0, 1)
        
        return rewards
    
    def _compute_statistical_reward(self, images: torch.Tensor) -> torch.Tensor:
        """Compute reward based on statistical similarity to MNIST."""
        batch_size = images.shape[0]
        
        # MNIST statistics: mean ~0.13, std ~0.31
        # Reward images that match these statistics
        img_mean = images.mean(dim=[2, 3])
        img_std = images.std(dim=[2, 3])
        
        mean_reward = 1.0 - torch.abs(img_mean - 0.13) / 0.5
        std_reward = 1.0 - torch.abs(img_std - 0.31) / 0.5
        
        rewards = (mean_reward + std_reward) / 2.0
        rewards = torch.clamp(rewards, 0, 1)
        
        return rewards.mean(dim=1)


class MNISTQualityReward:
    """
    Reward based on image quality metrics (sharpness, contrast, etc.).
    
    Measures visual quality without requiring reference images or classifiers.
    
    Pros:
    - Fast computation
    - Encourages sharp, high-contrast images
    - No external dependencies
    
    Cons:
    - Doesn't ensure correctness
    - May not capture all aspects of quality
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute quality-based rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
        # Compute multiple quality metrics
        sharpness = self._compute_sharpness(images)
        contrast = self._compute_contrast(images)
        brightness = self._compute_brightness(images)
        
        # Combine metrics
        rewards = (sharpness + contrast + brightness) / 3.0
        
        return rewards
    
    def _compute_sharpness(self, images: torch.Tensor) -> torch.Tensor:
        """Compute image sharpness using Laplacian variance."""
        # Convert to grayscale if needed
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
        # Laplacian kernel for edge detection
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
        
        # Apply Laplacian
        edges = F.conv2d(images, laplacian_kernel, padding=1)
        sharpness = edges.var(dim=[2, 3])  # Higher variance = sharper
        
        # Normalize to [0, 1]
        sharpness = torch.clamp(sharpness / 0.1, 0, 1)
        
        return sharpness.squeeze(1)
    
    def _compute_contrast(self, images: torch.Tensor) -> torch.Tensor:
        """Compute local contrast."""
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
        # Local mean
        kernel = torch.ones(1, 1, 5, 5, device=images.device) / 25.0
        local_mean = F.conv2d(images, kernel, padding=2)
        
        # Contrast = deviation from local mean
        contrast = torch.abs(images - local_mean).mean(dim=[2, 3])
        
        # Normalize
        contrast = torch.clamp(contrast * 5.0, 0, 1)
        
        return contrast.squeeze(1)
    
    def _compute_brightness(self, images: torch.Tensor) -> torch.Tensor:
        """Reward appropriate brightness (not too dark, not too bright)."""
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
        mean_brightness = images.mean(dim=[2, 3])
        
        # Reward brightness around 0.5 (MNIST-like)
        brightness_reward = 1.0 - torch.abs(mean_brightness - 0.5) * 2.0
        brightness_reward = torch.clamp(brightness_reward, 0, 1)
        
        return brightness_reward.squeeze(1)


class MNISTDiversityReward:
    """
    Reward for diversity in generated samples.
    
    Encourages the model to generate diverse outputs for the same prompt,
    preventing mode collapse.
    
    Pros:
    - Prevents mode collapse
    - Encourages exploration
    - Good for multi-modal generation
    
    Cons:
    - May conflict with quality rewards
    - Best used with small weight
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """
        Compute diversity rewards.
        
        Note: This should be computed across samples with the same prompt.
        The reward is higher when samples are diverse.
        """
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        
        # Flatten images
        images_flat = images.view(batch_size, -1)
        
        # Compute pairwise distances
        # For efficiency, compute variance within each sample group
        # In practice, you'd group by prompt_id
        
        # Simplified: reward based on variance within batch
        # (In real implementation, group by prompt_id first)
        batch_variance = images_flat.var(dim=0).mean()
        
        # Normalize
        diversity_reward = torch.clamp(batch_variance * 10.0, 0, 1)
        
        # Return same reward for all samples (since it's a group metric)
        rewards = torch.ones(batch_size, device=self.device) * diversity_reward
        
        return rewards


class CombinedMNISTReward:
    """
    Combined multi-objective reward function.
    
    Combines multiple reward signals for balanced optimization.
    This is typically the best approach for GRPO training.
    """
    
    def __init__(
        self,
        device='cuda',
        weights: Optional[Dict[str, float]] = None,
        use_classifier: bool = True,
        use_confidence: bool = True,
        use_quality: bool = True,
        use_perceptual: bool = False,
        use_diversity: bool = False,
    ):
        self.device = device
        
        # Default weights
        default_weights = {
            'classification': 0.5,
            'confidence': 0.2,
            'quality': 0.2,
            'perceptual': 0.1,
            'diversity': 0.0,  # Usually small or zero
        }
        self.weights = weights or default_weights
        
        # Initialize reward functions
        self.rewards = {}
        
        if use_classifier:
            self.rewards['classification'] = MNISTClassifierReward(device=device)
        
        if use_confidence:
            if 'classification' not in self.rewards:
                self.rewards['classification'] = MNISTClassifierReward(device=device)
            self.rewards['confidence'] = MNISTConfidenceReward(self.rewards['classification'])
        
        if use_quality:
            self.rewards['quality'] = MNISTQualityReward(device=device)
        
        if use_perceptual:
            self.rewards['perceptual'] = MNISTPerceptualReward(device=device)
        
        if use_diversity:
            self.rewards['diversity'] = MNISTDiversityReward(device=device)
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute combined rewards."""
        batch_size = images.shape[0]
        total_rewards = torch.zeros(batch_size, device=self.device)
        
        # Compute each reward component
        reward_components = {}
        for name, reward_fn in self.rewards.items():
            if name == 'confidence' and 'classification' in self.rewards:
                # Confidence needs classification results
                component_rewards = reward_fn(images, prompts, **kwargs)
            else:
                component_rewards = reward_fn(images, prompts, **kwargs)
            
            reward_components[name] = component_rewards
            
            # Add weighted component
            weight = self.weights.get(name, 0.0)
            total_rewards += weight * component_rewards
        
        # Normalize to reasonable range
        total_rewards = torch.clamp(total_rewards, 0, 1)
        
        return total_rewards


class MNISTAdversarialReward:
    """
    Adversarial/Discriminator reward using a GAN discriminator.
    
    Uses a discriminator trained to distinguish real vs fake MNIST digits.
    Higher reward = more "realistic" according to discriminator.
    
    Pros:
    - Encourages realistic digit appearance
    - Captures distribution-level quality
    - Good for avoiding mode collapse
    
    Cons:
    - Requires training a discriminator first
    - Can be unstable during training
    - May not ensure correctness
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.discriminator = self._create_discriminator()
        self.discriminator.eval()
        self.discriminator.to(device)
    
    def _create_discriminator(self):
        """Create a simple discriminator network."""
        class Discriminator(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout = nn.Dropout(0.25)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 1)
            
            def forward(self, x):
                x = self.conv1(x)
                x = F.leaky_relu(x, 0.2)
                x = self.conv2(x)
                x = F.leaky_relu(x, 0.2)
                x = F.max_pool2d(x, 2)
                x = self.dropout(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.leaky_relu(x, 0.2)
                x = self.fc2(x)
                return torch.sigmoid(x)
        
        return Discriminator()
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute adversarial rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Get discriminator score (probability of being real)
        with torch.no_grad():
            scores = self.discriminator(images).squeeze(1)
        
        return scores


class MNISTAutoencoderReward:
    """
    Reconstruction-based reward using an autoencoder.
    
    Measures how well an autoencoder can reconstruct the generated digit.
    Higher reconstruction quality = better digit structure.
    
    Pros:
    - Encourages well-structured digits
    - Captures semantic features
    - Good for ensuring digit coherence
    
    Cons:
    - Requires training an autoencoder first
    - May not directly optimize for correctness
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.autoencoder = self._create_autoencoder()
        self.autoencoder.eval()
        self.autoencoder.to(device)
    
    def _create_autoencoder(self):
        """Create a simple autoencoder."""
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                # Encoder
                self.enc1 = nn.Conv2d(1, 16, 3, padding=1)
                self.enc2 = nn.Conv2d(16, 8, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                
                # Decoder
                self.dec1 = nn.ConvTranspose2d(8, 16, 2, stride=2)
                self.dec2 = nn.ConvTranspose2d(16, 1, 2, stride=2)
            
            def forward(self, x):
                # Encode
                x = F.relu(self.enc1(x))
                x = self.pool(x)
                x = F.relu(self.enc2(x))
                x = self.pool(x)
                
                # Decode
                x = F.relu(self.dec1(x))
                x = torch.sigmoid(self.dec2(x))
                return x
        
        return Autoencoder()
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute reconstruction-based rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        images_norm = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        # Reconstruct
        with torch.no_grad():
            reconstructed = self.autoencoder(images_norm)
        
        # Compute reconstruction error (lower error = higher reward)
        mse = F.mse_loss(reconstructed, images_norm, reduction='none').mean(dim=[1, 2, 3])
        rewards = 1.0 - torch.clamp(mse, 0, 1)  # Invert: lower error = higher reward
        
        return rewards


class MNISTFeatureReward:
    """
    Feature-based reward using intermediate features from a pre-trained classifier.
    
    Extracts features from intermediate layers and compares to reference features.
    Encourages semantically meaningful digit representations.
    
    Pros:
    - Captures high-level semantic features
    - More robust than final classification
    - Encourages meaningful representations
    
    Cons:
    - Requires reference feature extraction
    - More complex computation
    """
    
    def __init__(self, classifier_reward: MNISTClassifierReward, layer_name='fc1'):
        self.classifier = classifier_reward.classifier
        self.device = classifier_reward.device
        self.layer_name = layer_name
        self.feature_extractor = self._create_feature_extractor()
    
    def _create_feature_extractor(self):
        """Create feature extractor from classifier."""
        # Hook to extract intermediate features
        features = {}
        
        def get_features(name):
            def hook(model, input, output):
                features[name] = output.detach()
            return hook
        
        # Register hook at desired layer
        if hasattr(self.classifier, 'fc1'):
            self.classifier.fc1.register_forward_hook(get_features('fc1'))
        
        return features
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute feature-based rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        images = (images - 0.1307) / 0.3081
        
        # Extract features
        with torch.no_grad():
            _ = self.classifier(images)
            if 'fc1' in self.feature_extractor:
                features = self.feature_extractor['fc1']
            else:
                # Fallback: use final logits
                features = self.classifier(images)
        
        # Normalize features
        features_norm = F.normalize(features, p=2, dim=1)
        
        # Reward based on feature magnitude (encourages confident features)
        feature_magnitude = torch.norm(features_norm, dim=1)
        rewards = torch.clamp(feature_magnitude, 0, 1)
        
        return rewards


class MNISTGeometricReward:
    """
    Geometric reward measuring digit properties (centroid, aspect ratio, etc.).
    
    Encourages digits with proper geometric properties typical of MNIST.
    
    Pros:
    - Fast computation
    - Encourages proper digit structure
    - No external models needed
    
    Cons:
    - May not capture all quality aspects
    - Digit-specific (works better for some digits)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute geometric rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Normalize
        images_norm = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        # Compute geometric properties
        rewards = []
        for i in range(batch_size):
            img = images_norm[i, 0]
            
            # 1. Centroid should be near center
            y_coords, x_coords = torch.meshgrid(
                torch.arange(28, device=self.device),
                torch.arange(28, device=self.device),
                indexing='ij'
            )
            total_mass = img.sum()
            if total_mass > 1e-6:
                centroid_x = (img * x_coords).sum() / total_mass
                centroid_y = (img * y_coords).sum() / total_mass
                center_x, center_y = 13.5, 13.5  # Center of 28x28
                centroid_error = torch.sqrt((centroid_x - center_x)**2 + (centroid_y - center_y)**2)
                centroid_reward = 1.0 - torch.clamp(centroid_error / 10.0, 0, 1)
            else:
                centroid_reward = 0.0
            
            # 2. Aspect ratio (should be roughly square for most digits)
            # Compute bounding box
            nonzero = torch.nonzero(img > 0.1, as_tuple=False)
            if len(nonzero) > 0:
                min_y, min_x = nonzero.min(dim=0)[0]
                max_y, max_x = nonzero.max(dim=0)[0]
                height = (max_y - min_y).float() + 1
                width = (max_x - min_x).float() + 1
                aspect_ratio = height / (width + 1e-8)
                # Reward aspect ratio close to 1.0 (square)
                aspect_reward = 1.0 - torch.clamp(torch.abs(aspect_ratio - 1.0), 0, 1)
            else:
                aspect_reward = 0.0
            
            # 3. Fill ratio (should have reasonable amount of foreground)
            fill_ratio = (img > 0.1).float().mean()
            # Reward fill ratio between 0.1 and 0.5 (typical for MNIST)
            if 0.1 <= fill_ratio <= 0.5:
                fill_reward = 1.0
            else:
                fill_reward = 1.0 - torch.clamp(torch.abs(fill_ratio - 0.3) / 0.3, 0, 1)
            
            # Combine rewards
            total_reward = (centroid_reward + aspect_reward + fill_reward) / 3.0
            rewards.append(total_reward)
        
        return torch.tensor(rewards, device=self.device)


class MNISTSymmetryReward:
    """
    Symmetry reward for digits that should be symmetric (0, 1, 3, 8).
    
    Encourages vertical or horizontal symmetry for symmetric digits.
    
    Pros:
    - Encourages proper digit structure
    - Fast computation
    - Digit-specific optimization
    
    Cons:
    - Only works for symmetric digits
    - May conflict with other rewards for asymmetric digits
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        # Digits that are vertically symmetric
        self.vertically_symmetric = {0, 1, 3, 8}
        # Digits that are horizontally symmetric (less common)
        self.horizontally_symmetric = {0, 8}
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute symmetry rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        
        # Normalize
        images_norm = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        # Extract target digits
        target_digits = []
        for prompt in prompts:
            digit = self._parse_digit(prompt)
            target_digits.append(digit)
        
        rewards = []
        for i, digit in enumerate(target_digits):
            img = images_norm[i, 0]
            
            if digit in self.vertically_symmetric:
                # Check vertical symmetry (left-right)
                img_flipped = torch.flip(img, dims=[1])
                symmetry = 1.0 - F.mse_loss(img, img_flipped, reduction='mean')
            elif digit in self.horizontally_symmetric:
                # Check horizontal symmetry (top-bottom)
                img_flipped = torch.flip(img, dims=[0])
                symmetry = 1.0 - F.mse_loss(img, img_flipped, reduction='mean')
            else:
                # For asymmetric digits, don't penalize (return neutral reward)
                symmetry = 0.5
            
            rewards.append(torch.clamp(symmetry, 0, 1))
        
        return torch.tensor(rewards, device=self.device)
    
    def _parse_digit(self, prompt: str) -> int:
        """Extract digit from prompt."""
        prompt_lower = prompt.lower()
        for i in range(10):
            if f"digit {i}" in prompt_lower or f"number {i}" in prompt_lower:
                return i
        return 0


class MNISTGradientReward:
    """
    Gradient-based reward using gradients from classifier.
    
    Uses gradients from classifier to guide generation towards correct class.
    Higher reward = stronger gradient signal towards target class.
    
    Pros:
    - Provides direct optimization signal
    - Encourages correct class generation
    - Can be combined with other rewards
    
    Cons:
    - Requires gradient computation (more expensive)
    - May be unstable
    """
    
    def __init__(self, classifier_reward: MNISTClassifierReward):
        self.classifier = classifier_reward.classifier
        self.device = classifier_reward.device
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute gradient-based rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        images_norm = (images - 0.1307) / 0.3081
        
        # Extract target digits
        target_digits = []
        for prompt in prompts:
            digit = self._parse_digit(prompt)
            target_digits.append(digit)
        target_digits = torch.tensor(target_digits, device=self.device)
        
        # Compute gradients
        images_norm.requires_grad_(True)
        logits = self.classifier(images_norm)
        
        # Get logits for target classes
        target_logits = logits.gather(1, target_digits.unsqueeze(1)).squeeze(1)
        
        # Compute gradient magnitude (how much the image affects target class probability)
        gradients = torch.autograd.grad(
            target_logits.sum(),
            images_norm,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Reward based on gradient magnitude (stronger gradient = better signal)
        gradient_magnitude = gradients.norm(dim=[1, 2, 3])
        rewards = torch.clamp(gradient_magnitude / 10.0, 0, 1)  # Normalize
        
        return rewards
    
    def _parse_digit(self, prompt: str) -> int:
        """Extract digit from prompt."""
        prompt_lower = prompt.lower()
        for i in range(10):
            if f"digit {i}" in prompt_lower or f"number {i}" in prompt_lower:
                return i
        return 0


class MNISTEnsembleReward:
    """
    Ensemble reward using multiple classifiers.
    
    Combines predictions from multiple classifiers for more robust rewards.
    
    Pros:
    - More robust than single classifier
    - Reduces overfitting to one model
    - Better generalization
    
    Cons:
    - More computationally expensive
    - Requires multiple trained models
    """
    
    def __init__(self, num_classifiers=3, device='cuda'):
        self.device = device
        self.classifiers = []
        for i in range(num_classifiers):
            classifier = MNISTClassifierReward(device=device, pretrained=False)
            # In practice, load different trained models
            self.classifiers.append(classifier)
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute ensemble rewards."""
        batch_size = images.shape[0]
        
        # Get rewards from each classifier
        all_rewards = []
        for classifier in self.classifiers:
            rewards = classifier(images, prompts, **kwargs)
            all_rewards.append(rewards)
        
        # Average rewards
        all_rewards = torch.stack(all_rewards, dim=0)
        ensemble_rewards = all_rewards.mean(dim=0)
        
        return ensemble_rewards


class MNISTContrastiveReward:
    """
    Contrastive reward comparing to reference MNIST samples.
    
    Uses contrastive learning to measure similarity to real MNIST digits.
    
    Pros:
    - Encourages realistic digit appearance
    - Uses learned representations
    - Good for distribution matching
    
    Cons:
    - Requires reference samples
    - More complex setup
    """
    
    def __init__(self, classifier_reward: MNISTClassifierReward, num_references=10):
        self.classifier = classifier_reward.classifier
        self.device = classifier_reward.device
        self.num_references = num_references
        # In practice, load reference MNIST samples
        self.reference_features = None
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from classifier."""
        # Use intermediate layer features
        # Simplified: use final logits as features
        with torch.no_grad():
            logits = self.classifier(images)
            features = F.normalize(logits, p=2, dim=1)
        return features
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute contrastive rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        images_norm = (images - 0.1307) / 0.3081
        
        # Extract features
        features = self._extract_features(images_norm)
        
        # If no reference features, use random initialization
        if self.reference_features is None:
            # Create random reference features (in practice, use real MNIST)
            self.reference_features = torch.randn(
                self.num_references, features.shape[1], device=self.device
            )
            self.reference_features = F.normalize(self.reference_features, p=2, dim=1)
        
        # Compute similarity to reference features
        similarities = torch.mm(features, self.reference_features.t())  # [batch, num_refs]
        max_similarity = similarities.max(dim=1)[0]  # Best match
        
        # Reward based on similarity
        rewards = (max_similarity + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
        
        return rewards


class MNISTAdversarialRobustnessReward:
    """
    Adversarial robustness reward.
    
    Rewards digits that are robust to small perturbations.
    Encourages stable, well-formed digits.
    
    Pros:
    - Encourages robust features
    - Reduces sensitivity to noise
    - Better generalization
    
    Cons:
    - More computationally expensive
    - May reduce diversity
    """
    
    def __init__(self, classifier_reward: MNISTClassifierReward, epsilon=0.1):
        self.classifier = classifier_reward.classifier
        self.device = classifier_reward.device
        self.epsilon = epsilon
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute robustness rewards."""
        batch_size = images.shape[0]
        
        # Preprocess
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if images.shape[1] == 3:
            images = images.mean(dim=1, keepdim=True)
        if images.shape[-1] != 28:
            images = F.interpolate(images, size=(28, 28), mode='bilinear', align_corners=False)
        images_norm = (images - 0.1307) / 0.3081
        
        # Extract target digits
        target_digits = []
        for prompt in prompts:
            digit = self._parse_digit(prompt)
            target_digits.append(digit)
        target_digits = torch.tensor(target_digits, device=self.device)
        
        # Original prediction
        with torch.no_grad():
            logits_orig = self.classifier(images_norm)
            probs_orig = F.softmax(logits_orig, dim=1)
            target_probs_orig = probs_orig.gather(1, target_digits.unsqueeze(1)).squeeze(1)
        
        # Add small perturbation
        noise = torch.randn_like(images_norm) * self.epsilon
        images_perturbed = torch.clamp(images_norm + noise, 0, 1)
        
        # Perturbed prediction
        with torch.no_grad():
            logits_pert = self.classifier(images_perturbed)
            probs_pert = F.softmax(logits_pert, dim=1)
            target_probs_pert = probs_pert.gather(1, target_digits.unsqueeze(1)).squeeze(1)
        
        # Reward: how stable is the prediction?
        # Higher reward = less change in target probability
        prob_change = torch.abs(target_probs_orig - target_probs_pert)
        robustness = 1.0 - prob_change  # Less change = more robust
        
        return torch.clamp(robustness, 0, 1)
    
    def _parse_digit(self, prompt: str) -> int:
        """Extract digit from prompt."""
        prompt_lower = prompt.lower()
        for i in range(10):
            if f"digit {i}" in prompt_lower or f"number {i}" in prompt_lower:
                return i
        return 0


class MNISTYellowReward:
    """
    Reward for generating yellow digits.
    
    Encourages digits with yellow color. For RGB images, detects yellow directly.
    For grayscale images, interprets "yellow" as bright pixels (since yellow is bright).
    
    Pros:
    - Directly optimizes for yellow color
    - Can be combined with other rewards
    - Works with both RGB and grayscale
    
    Cons:
    - May conflict with digit correctness if not balanced
    - For grayscale, "yellow" is approximated as brightness
    
    Usage:
        reward_fn = MNISTYellowReward(device='cuda', yellow_threshold=0.6)
        rewards = reward_fn(images, prompts)
    """
    
    def __init__(self, device='cuda', yellow_threshold=0.6, brightness_threshold=0.7):
        """
        Args:
            device: Device to run on
            yellow_threshold: Threshold for yellow detection in RGB (0-1)
            brightness_threshold: Threshold for brightness in grayscale (0-1)
        """
        self.device = device
        self.yellow_threshold = yellow_threshold
        self.brightness_threshold = brightness_threshold
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """
        Compute yellow-based rewards.
        
        Args:
            images: Generated images [batch_size, C, H, W] or [batch_size, H, W]
            prompts: List of prompt strings
        
        Returns:
            rewards: Tensor of rewards [batch_size]
        """
        batch_size = images.shape[0]
        
        # Handle different input formats
        if images.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
            images = images.unsqueeze(1)
        
        # Normalize to [0, 1] if needed
        if images.max() > 1.1:
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        rewards = []
        
        for i in range(batch_size):
            img = images[i]
            
            if img.shape[0] == 3:  # RGB image
                reward = self._compute_rgb_yellow_reward(img)
            else:  # Grayscale image
                reward = self._compute_grayscale_yellow_reward(img)
            
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device)
    
    def _compute_rgb_yellow_reward(self, img: torch.Tensor) -> float:
        """
        Compute yellow reward for RGB image.
        
        Yellow in RGB: high R and G, low B
        Typical yellow: R≈1.0, G≈1.0, B≈0.0
        """
        # img shape: [3, H, W]
        R, G, B = img[0], img[1], img[2]
        
        # Yellow detection: high R and G, low B
        # Yellow score: (R + G) / 2 - B
        yellow_score = (R + G) / 2.0 - B
        
        # Threshold: pixels with yellow_score > threshold are considered yellow
        yellow_mask = (yellow_score > self.yellow_threshold).float()
        
        # Reward: fraction of pixels that are yellow
        yellow_ratio = yellow_mask.mean().item()
        
        # Also reward intensity of yellow (how "yellow" the yellow pixels are)
        yellow_intensity = (yellow_score * yellow_mask).sum() / (yellow_mask.sum() + 1e-8)
        
        # Combined reward: ratio * intensity
        reward = yellow_ratio * (yellow_intensity.item() + 0.5)  # Scale intensity
        
        return float(torch.clamp(torch.tensor(reward), 0, 1))
    
    def _compute_grayscale_yellow_reward(self, img: torch.Tensor) -> float:
        """
        Compute yellow reward for grayscale image.
        
        Since yellow is bright in grayscale, we interpret bright pixels as "yellow".
        """
        # img shape: [1, H, W] or [H, W]
        if img.dim() == 3:
            img = img.squeeze(0)
        
        # Bright pixels (high intensity) are interpreted as "yellow"
        bright_mask = (img > self.brightness_threshold).float()
        
        # Reward: fraction of bright pixels
        bright_ratio = bright_mask.mean().item()
        
        # Also reward brightness intensity
        bright_intensity = (img * bright_mask).sum() / (bright_mask.sum() + 1e-8)
        
        # Combined reward
        reward = bright_ratio * (bright_intensity.item() + 0.3)  # Scale intensity
        
        return float(torch.clamp(torch.tensor(reward), 0, 1))


class MNISTColorReward:
    """
    General color reward function that can target any color.
    
    More flexible than yellow-specific reward - can be configured for any color.
    
    Usage:
        # For yellow
        reward_fn = MNISTColorReward(device='cuda', target_color='yellow')
        
        # For red
        reward_fn = MNISTColorReward(device='cuda', target_color='red')
        
        # Custom RGB color
        reward_fn = MNISTColorReward(device='cuda', target_rgb=(1.0, 1.0, 0.0))
    """
    
    def __init__(self, device='cuda', target_color='yellow', target_rgb=None, threshold=0.6):
        """
        Args:
            device: Device to run on
            target_color: Target color name ('yellow', 'red', 'blue', 'green', etc.)
            target_rgb: Custom RGB color as tuple (R, G, B) in [0, 1]
            threshold: Threshold for color detection
        """
        self.device = device
        self.threshold = threshold
        
        # Color definitions (RGB values)
        color_map = {
            'yellow': (1.0, 1.0, 0.0),
            'red': (1.0, 0.0, 0.0),
            'green': (0.0, 1.0, 0.0),
            'blue': (0.0, 0.0, 1.0),
            'orange': (1.0, 0.5, 0.0),
            'purple': (0.5, 0.0, 0.5),
            'cyan': (0.0, 1.0, 1.0),
            'magenta': (1.0, 0.0, 1.0),
            'white': (1.0, 1.0, 1.0),
            'black': (0.0, 0.0, 0.0),
        }
        
        if target_rgb is not None:
            self.target_rgb = torch.tensor(target_rgb, device=device)
        elif target_color.lower() in color_map:
            self.target_rgb = torch.tensor(color_map[target_color.lower()], device=device)
        else:
            # Default to yellow
            self.target_rgb = torch.tensor(color_map['yellow'], device=device)
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute color-based rewards."""
        batch_size = images.shape[0]
        
        # Handle different input formats
        if images.dim() == 3:
            images = images.unsqueeze(1)
        
        # Normalize to [0, 1] if needed
        if images.max() > 1.1:
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        rewards = []
        
        for i in range(batch_size):
            img = images[i]
            
            if img.shape[0] == 3:  # RGB
                reward = self._compute_rgb_color_reward(img)
            else:  # Grayscale
                reward = self._compute_grayscale_color_reward(img)
            
            rewards.append(reward)
        
        return torch.tensor(rewards, device=self.device)
    
    def _compute_rgb_color_reward(self, img: torch.Tensor) -> float:
        """Compute color reward for RGB image."""
        # img shape: [3, H, W]
        R, G, B = img[0], img[1], img[2]
        target_R, target_G, target_B = self.target_rgb[0], self.target_rgb[1], self.target_rgb[2]
        
        # Compute color similarity (inverse distance)
        color_diff = torch.sqrt(
            (R - target_R) ** 2 + 
            (G - target_G) ** 2 + 
            (B - target_B) ** 2
        )
        
        # Pixels similar to target color
        color_mask = (color_diff < self.threshold).float()
        
        # Reward: fraction of pixels matching target color
        color_ratio = color_mask.mean().item()
        
        # Also reward how close the matching pixels are
        similarity = (1.0 - color_diff * color_mask).sum() / (color_mask.sum() + 1e-8)
        
        reward = color_ratio * (similarity.item() + 0.3)
        
        return float(torch.clamp(torch.tensor(reward), 0, 1))
    
    def _compute_grayscale_color_reward(self, img: torch.Tensor) -> float:
        """Compute color reward for grayscale image."""
        # For grayscale, interpret brightness based on target color
        if img.dim() == 3:
            img = img.squeeze(0)
        
        # Bright colors (yellow, white) -> bright pixels
        # Dark colors (black, blue) -> dark pixels
        target_brightness = self.target_rgb.mean().item()
        
        # Reward pixels close to target brightness
        brightness_diff = torch.abs(img - target_brightness)
        brightness_mask = (brightness_diff < self.threshold).float()
        
        brightness_ratio = brightness_mask.mean().item()
        similarity = (1.0 - brightness_diff * brightness_mask).sum() / (brightness_mask.sum() + 1e-8)
        
        reward = brightness_ratio * (similarity.item() + 0.3)
        
        return float(torch.clamp(torch.tensor(reward), 0, 1))


class MNISTYellowDigitReward:
    """
    Combined reward: Yellow color + correct digit.
    
    Rewards digits that are both yellow AND correct.
    This ensures you get yellow digits that are still recognizable.
    
    Usage:
        classifier = MNISTClassifierReward(device='cuda')
        yellow_reward = MNISTYellowReward(device='cuda')
        
        combined = MNISTYellowDigitReward(classifier, yellow_reward, 
                                         digit_weight=0.6, yellow_weight=0.4)
    """
    
    def __init__(self, classifier_reward: MNISTClassifierReward, 
                 yellow_reward: MNISTYellowReward,
                 digit_weight=0.6, yellow_weight=0.4):
        """
        Args:
            classifier_reward: Classification reward function
            yellow_reward: Yellow color reward function
            digit_weight: Weight for digit correctness (default 0.6)
            yellow_weight: Weight for yellow color (default 0.4)
        """
        self.classifier_reward = classifier_reward
        self.yellow_reward = yellow_reward
        self.digit_weight = digit_weight
        self.yellow_weight = yellow_weight
        
        # Normalize weights
        total = digit_weight + yellow_weight
        self.digit_weight /= total
        self.yellow_weight /= total
    
    def __call__(self, images: torch.Tensor, prompts: List[str], **kwargs) -> torch.Tensor:
        """Compute combined yellow + digit rewards."""
        # Get digit correctness rewards
        digit_rewards = self.classifier_reward(images, prompts, **kwargs)
        
        # Get yellow color rewards
        yellow_rewards = self.yellow_reward(images, prompts, **kwargs)
        
        # Combine
        combined_rewards = (self.digit_weight * digit_rewards + 
                          self.yellow_weight * yellow_rewards)
        
        return combined_rewards


# Example usage and recommended configurations

def get_recommended_reward_config(config_name: str = 'balanced', device: str = 'cuda'):
    """
    Get recommended reward configurations for different training objectives.
    
    Configurations:
    - 'accuracy': Focus on classification accuracy
    - 'quality': Focus on visual quality
    - 'balanced': Balance between accuracy and quality (recommended)
    - 'diverse': Include diversity reward
    """
    configs = {
        'accuracy': CombinedMNISTReward(
            device=device,
            weights={
                'classification': 0.7,
                'confidence': 0.3,
                'quality': 0.0,
                'perceptual': 0.0,
                'diversity': 0.0,
            },
            use_classifier=True,
            use_confidence=True,
            use_quality=False,
            use_perceptual=False,
            use_diversity=False,
        ),
        'quality': CombinedMNISTReward(
            device=device,
            weights={
                'classification': 0.3,
                'confidence': 0.2,
                'quality': 0.4,
                'perceptual': 0.1,
                'diversity': 0.0,
            },
            use_classifier=True,
            use_confidence=True,
            use_quality=True,
            use_perceptual=True,
            use_diversity=False,
        ),
        'balanced': CombinedMNISTReward(
            device=device,
            weights={
                'classification': 0.5,
                'confidence': 0.2,
                'quality': 0.25,
                'perceptual': 0.05,
                'diversity': 0.0,
            },
            use_classifier=True,
            use_confidence=True,
            use_quality=True,
            use_perceptual=False,
            use_diversity=False,
        ),
        'diverse': CombinedMNISTReward(
            device=device,
            weights={
                'classification': 0.4,
                'confidence': 0.2,
                'quality': 0.25,
                'perceptual': 0.05,
                'diversity': 0.1,
            },
            use_classifier=True,
            use_confidence=True,
            use_quality=True,
            use_perceptual=False,
            use_diversity=True,
        ),
    }
    
    return configs.get(config_name, configs['balanced'])
