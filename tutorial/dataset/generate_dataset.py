"""
Generate a toy dataset for FlowGRPO tutorial.

This creates simple prompts that are easy to evaluate and visualize.
"""
import os
from pathlib import Path

# Create dataset directory
dataset_dir = Path(__file__).parent
dataset_dir.mkdir(exist_ok=True)

# Simple prompts for toy dataset
train_prompts = [
    "a red circle",
    "a blue square",
    "a green triangle",
    "a yellow circle",
    "a red square",
    "a blue triangle",
    "a green circle",
    "a yellow square",
    "a red triangle",
    "a blue circle",
    "a green square",
    "a yellow triangle",
    "a small red circle",
    "a large blue square",
    "a small green triangle",
    "a large yellow circle",
    "a small red square",
    "a large blue triangle",
    "a small green circle",
    "a large yellow square",
]

test_prompts = [
    "a purple circle",
    "a orange square",
    "a pink triangle",
    "a cyan circle",
    "a magenta square",
]

def generate_dataset():
    """Generate train and test dataset files."""
    # Write training prompts
    train_file = dataset_dir / "train.txt"
    with open(train_file, "w") as f:
        for prompt in train_prompts:
            f.write(prompt + "\n")
    print(f"Generated {train_file} with {len(train_prompts)} prompts")
    
    # Write test prompts
    test_file = dataset_dir / "test.txt"
    with open(test_file, "w") as f:
        for prompt in test_prompts:
            f.write(prompt + "\n")
    print(f"Generated {test_file} with {len(test_prompts)} prompts")

if __name__ == "__main__":
    generate_dataset()
