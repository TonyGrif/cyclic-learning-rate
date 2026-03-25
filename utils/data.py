"""Dataset loading utilities for CIFAR-10 and CIFAR-100."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from utils.transforms import get_test_transforms, get_train_transforms

SUPPORTED_DATASETS = ("cifar10", "cifar100")

NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
}


def get_dataloaders(
    dataset_name: str,
    batch_size: int,
    seed: int,
    data_dir: str = "./data",
    num_workers: int = 4,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get train, validation, and test dataloaders for the specified dataset.

    Args:
        dataset_name: Name of the dataset (cifar10, cifar100).
        batch_size: Batch size for all dataloaders.
        seed: Random seed for reproducible train/val split.
        data_dir: Directory to store/load dataset files.
        num_workers: Number of worker processes for data loading.
        val_split: Fraction of training data to use for validation.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).

    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )

    train_transforms = get_train_transforms(dataset_name)
    test_transforms = get_test_transforms(dataset_name)

    dataset_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100

    full_train_dataset = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transforms,
    )

    # Need separate instance with test transforms for validation subset
    full_train_dataset_for_val = dataset_class(
        root=data_dir,
        train=True,
        download=True,
        transform=test_transforms,
    )

    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = random_split(
        range(total_size),
        [train_size, val_size],
        generator=generator,
    )

    train_dataset = Subset(full_train_dataset, train_indices.indices)
    val_dataset = Subset(full_train_dataset_for_val, val_indices.indices)

    test_dataset = dataset_class(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_num_classes(dataset_name: str) -> int:
    """Get the number of classes for the specified dataset.

    Args:
        dataset_name: Name of the dataset (cifar10, cifar100).

    Returns:
        Number of classes in the dataset.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in NUM_CLASSES:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {SUPPORTED_DATASETS}"
        )

    return NUM_CLASSES[dataset_name]
