"""
Dataset loader for MNIST.
"""
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """MNIST dataset loader with normalization."""

    def __init__(self, dataset_dir: str, split: str = "train", download: bool = True):
        """
        Args:
            dataset_dir: Directory to store/load MNIST dataset
            split: "train" or "test"
            download: Whether to download MNIST if not present
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.dataset = datasets.MNIST(
            root=str(self.dataset_dir),
            train=(split == "train"),
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
            ])
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, label = self.dataset[idx]
        return {
            "image": image.view(-1),  # Flatten: [1, 28, 28] -> [784]
            "label": label if isinstance(label, torch.Tensor) else torch.tensor(label),
        }
