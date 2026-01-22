"""
Train a small MNIST classifier for use as a reward function.

This script trains the TinyClassifier (~12K params) on MNIST.
The trained model can be used with MNISTClassifierReward to encourage
generated digits to look like a specific class.

Usage:
    python rewards/reward_train.py
    python rewards/reward_train.py --epochs 10 --lr 0.001
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mnist_rewards import TinyClassifier


def train_classifier(
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = 'cuda',
    save_path: str = None,
):
    """
    Train the TinyClassifier on MNIST.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to train on
        save_path: Path to save the checkpoint
    """
    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'classifier_checkpoint.pt')

    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load MNIST
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model
    model = TinyClassifier().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"TinyClassifier has {num_params:,} parameters")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = train_loss / len(train_loader)

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * test_correct / test_total

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    # Save checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    print(f"Final test accuracy: {test_acc:.2f}%")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train MNIST classifier for reward function')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--save-path', type=str, default=None, help='Save path for checkpoint')
    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    train_classifier(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_path,
    )


if __name__ == '__main__':
    main()
