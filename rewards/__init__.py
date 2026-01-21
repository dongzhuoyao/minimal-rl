"""
Reward functions for MNIST FlowGRPO training.
"""
from .mnist_rewards import (
    get_recommended_reward_config,
    MNISTBrightDigitReward,
    MNISTCenteredDigitReward,
    MNISTSparseDigitReward,
    MNISTLargeDigitReward,
    MNISTHighContrastReward,
    MNISTTinyDigitReward,
    SimpleReward,
    SimpleDigitClassifier,
    compute_digit_reward,
    compute_shape_reward,
    compute_reward,
)

__all__ = [
    "get_recommended_reward_config",
    "MNISTBrightDigitReward",
    "MNISTCenteredDigitReward",
    "MNISTSparseDigitReward",
    "MNISTLargeDigitReward",
    "MNISTHighContrastReward",
    "MNISTTinyDigitReward",
    "SimpleReward",
    "SimpleDigitClassifier",
    "compute_digit_reward",
    "compute_shape_reward",
    "compute_reward",
]
