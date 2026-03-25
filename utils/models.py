"""Model loading utilities for ResNet and DenseNet architectures."""

import torch.nn as nn
from torchvision import models

SUPPORTED_MODELS = ("resnet34", "densenet121")


def get_model(model_name: str, num_classes: int) -> nn.Module:
    """Get an untrained model with the specified architecture.

    Args:
        model_name: Name of the model architecture (resnet34, densenet121).
        num_classes: Number of output classes for the final layer.

    Returns:
        Untrained model with adjusted final layer for num_classes.

    Raises:
        ValueError: If model_name is not supported.
    """
    model_name = model_name.lower()

    if model_name == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported model: {model_name}. Supported models: {SUPPORTED_MODELS}"
    )
