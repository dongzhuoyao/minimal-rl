"""
Dataset loader for tutorial prompts.
"""
from pathlib import Path
from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """Simple dataset that loads prompts from a text file."""
    
    def __init__(self, dataset_dir, split="train"):
        """
        Args:
            dataset_dir: Directory containing dataset files
            split: "train" or "test"
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        
        # Load prompts from file
        prompt_file = self.dataset_dir / f"{split}.txt"
        if not prompt_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {prompt_file}. "
                "Run python -m tutorial.dataset.generate_dataset first."
            )
        
        with open(prompt_file, "r") as f:
            self.prompts = [line.strip() for line in f.readlines() if line.strip()]
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "prompt_id": idx,
        }
