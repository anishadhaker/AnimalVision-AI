"""
ResNet-18 Model Architecture for 10-Class Animal Classification.

Architecture:
- Base: torchvision.models.resnet18
- Pretrained Weights: ResNet18_Weights.DEFAULT (ImageNet-1K pretrained)
- Feature Dimension: 512
- Replaced Head: Linear(in_features=512, out_features=num_classes)
- Input: Accepts [batch_size, 3, 224, 224] RGB images
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


def build_resnet18_model(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.0,
) -> nn.Module:
    """
    Constructs a ResNet-18 model adapted for animal classification.

    Args:
        num_classes (int): Number of target classes (default: 10).
        pretrained (bool): If True, loads ImageNet-1K pretrained weights (ResNet18_Weights.DEFAULT).
                          If False, initializes architecture with random weights.
        freeze_backbone (bool): If True, freezes all convolutional backbone layers,
                                keeping only the classification head trainable.
        dropout_rate (float): Optional dropout probability before the final linear layer.

    Returns:
        nn.Module: Adapted ResNet-18 model.
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features  # 512 for ResNet-18

    # Freeze convolutional backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final linear layer with 10-class head
    if dropout_rate > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes, bias=True),
        )
    else:
        model.fc = nn.Linear(in_features, num_classes, bias=True)

    # Ensure classification head parameters are always trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_resnet_layer4(model: nn.Module):
    """
    Utility to unfreeze stage 4 (layer4) residual blocks in addition to the classification head.
    Useful for progressive fine-tuning.
    """
    if hasattr(model, "layer4"):
        for param in model.layer4.parameters():
            param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
