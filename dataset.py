"""
Dataset loader for MNIST.
"""
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """MNIST dataset loader."""
    
    def __init__(self, dataset_dir, split="train", download=True):
        """
        Args:
            dataset_dir: Directory to store/load MNIST dataset
            split: "train" or "test"
            download: Whether to download MNIST if not present
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        
        # Create directory if it doesn't exist
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Load MNIST dataset
        is_train = (split == "train")
        self.dataset = datasets.MNIST(
            root=str(self.dataset_dir),
            train=is_train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
            ])
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Flatten image to 1D: [1, 28, 28] -> [784]
        image_flat = image.view(-1)
        
        # Handle label - it might be a tensor or an int
        if isinstance(label, (int, float)):
            label_value = int(label)
        else:
            label_value = int(label.item())
        
        return {
            "image": image_flat,
            "label": label,
        }
