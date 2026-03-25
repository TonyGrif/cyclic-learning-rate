"""Dataset-specific image transforms for training and evaluation."""

from torchvision import transforms

# CIFAR normalization values (computed from training set)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

SUPPORTED_DATASETS = ("cifar10", "cifar100")


def get_train_transforms(dataset_name: str) -> transforms.Compose:
    """Get training transforms for the specified dataset.

    Args:
        dataset_name: Name of the dataset (cifar10, cifar100).

    Returns:
        Composed transforms for training data augmentation.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_name = dataset_name.lower()

    if dataset_name in ("cifar10", "cifar100"):
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. Supported datasets: {SUPPORTED_DATASETS}"
    )


def get_test_transforms(dataset_name: str) -> transforms.Compose:
    """Get test/validation transforms for the specified dataset.

    Args:
        dataset_name: Name of the dataset (cifar10, cifar100).

    Returns:
        Composed transforms for test/validation data.

    Raises:
        ValueError: If dataset_name is not supported.
    """
    dataset_name = dataset_name.lower()

    if dataset_name in ("cifar10", "cifar100"):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    raise ValueError(
        f"Unsupported dataset: {dataset_name}. Supported datasets: {SUPPORTED_DATASETS}"
    )
