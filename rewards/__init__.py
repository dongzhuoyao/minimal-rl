"""
Reward functions for MNIST FlowGRPO training.
"""
from .mnist_rewards import (
    get_recommended_reward_config,
    # Toy rewards (easy to hack)
    MNISTBrightDigitReward,
    MNISTCenteredDigitReward,
    MNISTSparseDigitReward,
    MNISTLargeDigitReward,
    MNISTHighContrastReward,
    MNISTTinyDigitReward,
    # Hack-resistant rewards
    MNISTCompositeReward,
    MNISTBoundedSizeReward,
    MNISTBimodalReward,
    MNISTConnectedReward,
    MNISTStrokeConsistencyReward,
    # Legacy
    SimpleReward,
    SimpleDigitClassifier,
    compute_digit_reward,
    compute_shape_reward,
    compute_reward,
)

__all__ = [
    "get_recommended_reward_config",
    # Toy rewards
    "MNISTBrightDigitReward",
    "MNISTCenteredDigitReward",
    "MNISTSparseDigitReward",
    "MNISTLargeDigitReward",
    "MNISTHighContrastReward",
    "MNISTTinyDigitReward",
    # Hack-resistant rewards
    "MNISTCompositeReward",
    "MNISTBoundedSizeReward",
    "MNISTBimodalReward",
    "MNISTConnectedReward",
    "MNISTStrokeConsistencyReward",
    # Legacy
    "SimpleReward",
    "SimpleDigitClassifier",
    "compute_digit_reward",
    "compute_shape_reward",
    "compute_reward",
]
