import os

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_split_cifar100(data_dir='./data/cifar-100-python', batch_size=64, val_size=0.2):
    """
    Load and split the CIFAR-100 dataset into training, validation, and test sets.

    Args:
        data_dir (str): Directory where the CIFAR-100 dataset is stored.
        batch_size (int): Batch size for DataLoader.
        val_size (float): Proportion of the training set to be used as validation.

    Returns:
        tuple: (train_loader, val_loader, test_loader) DataLoaders for the training, validation, and test sets.
    """
    # Data preprocessing and augmentation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-100 dataset

    train_val_set = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform)
    test_set = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)

    # Split train set into train and validation sets
    train_indices, val_indices = train_test_split(list(range(len(train_val_set))), test_size=val_size, random_state=42)
    train_set = Subset(train_val_set, train_indices)
    val_set = Subset(train_val_set, val_indices)

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Example usage
    path = './data/cifar-100-python'

    print(os.path.isdir(path))
    train_loader, val_loader, test_loader = load_split_cifar100(data_dir=path)

    # Quick check of loader functionality
    for images, labels in train_loader:
        print(f"Batch size: {images.size(0)}")
        print(f"Image tensor size: {images.size()}")
        print(f"Label size: {labels.size()}")
        break  # Remove this line to process the entire dataset
