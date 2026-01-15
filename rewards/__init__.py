"""
Rewards module for FlowGRPO training.

This module contains reward functions for training flow matching models.
All reward functions are consolidated in mnist_rewards.py.
"""

from .mnist_rewards import (
    # Toy rewards
    MNISTBrightDigitReward,
    MNISTCenteredDigitReward,
    MNISTSparseDigitReward,
    MNISTLargeDigitReward,
    MNISTHighContrastReward,
    MNISTTinyDigitReward,
    # Config function
    get_recommended_reward_config,
    # Simple rewards
    SimpleReward,
    SimpleDigitClassifier,
    compute_digit_reward,
    compute_shape_reward,
    compute_reward,
)

__all__ = [
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
    'compute_reward',
]
