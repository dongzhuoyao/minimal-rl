"""
Rewards module for FlowGRPO training.

This module contains reward functions for training flow matching models:
- mnist_rewards.py: MNIST-specific reward functions
- simple_reward.py: Simple reward functions for toy tasks
- example_mnist_rewards.py: Examples for MNIST rewards
- example_toy_rewards.py: Examples for toy rewards
- visualize_reward.py: Visualization utilities for rewards
"""

from .mnist_rewards import (
    MNISTClassifierReward,
    MNISTConfidenceReward,
    MNISTQualityReward,
    MNISTPerceptualReward,
    MNISTDiversityReward,
    CombinedMNISTReward,
    MNISTDigitZeroReward,
    MNISTYellowReward,
    MNISTColorReward,
    MNISTYellowDigitReward,
    MNISTGeometricReward,
    MNISTAdversarialRobustnessReward,
    # Toy rewards
    MNISTBrightDigitReward,
    MNISTCenteredDigitReward,
    MNISTSparseDigitReward,
    MNISTLargeDigitReward,
    MNISTHighContrastReward,
    MNISTTinyDigitReward,
    # Config function
    get_recommended_reward_config,
)

from .simple_reward import (
    SimpleReward,
    SimpleDigitClassifier,
    compute_digit_reward,
    compute_shape_reward,
)

__all__ = [
    # MNIST rewards
    'MNISTClassifierReward',
    'MNISTConfidenceReward',
    'MNISTQualityReward',
    'MNISTPerceptualReward',
    'MNISTDiversityReward',
    'CombinedMNISTReward',
    'MNISTDigitZeroReward',
    'MNISTYellowReward',
    'MNISTColorReward',
    'MNISTYellowDigitReward',
    'MNISTGeometricReward',
    'MNISTAdversarialRobustnessReward',
    # Toy rewards
    'MNISTBrightDigitReward',
    'MNISTCenteredDigitReward',
    'MNISTSparseDigitReward',
    'MNISTLargeDigitReward',
    'MNISTHighContrastReward',
    'MNISTTinyDigitReward',
    # Config
    'get_recommended_reward_config',
    # Simple rewards
    'SimpleReward',
    'SimpleDigitClassifier',
    'compute_digit_reward',
    'compute_shape_reward',
]
