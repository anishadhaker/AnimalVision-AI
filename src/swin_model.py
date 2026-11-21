"""
Swin Transformer Model Adaptation for Animal Classification (Supports 5-Class CIFAR and 10-Class Animals-10).

Architecture:
- Base: torchvision.models.swin_t (Swin Transformer Tiny)
- Pretrained Weights: Swin_T_Weights.DEFAULT (ImageNet-1K pre-trained)
- Feature Dimension: 768
- Replaced Head: Linear(in_features=768, out_features=num_classes, bias=True)
- Input: Accepts [batch_size, 3, 224, 224] RGB images
"""

import sys
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Swin_T_Weights


class PretrainedWeightDownloadError(RuntimeError):
    """Raised when pretrained weights cannot be retrieved or downloaded."""
    pass


def build_swin_model(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.0,
) -> nn.Module:
    """
    Constructs a Swin-Tiny Transformer model adapted for animal classification.

    Args:
        num_classes (int): Number of target classes (default: 10).
        pretrained (bool): If True, downloads/loads ImageNet-1k pretrained weights using
                          torchvision's Swin_T_Weights.DEFAULT.
                          If False, initializes model architecture without pretrained weights.
        freeze_backbone (bool): If True, freezes all parameters except the classification head.
        dropout_rate (float): Optional dropout probability before the final classifier layer.

    Returns:
        nn.Module: Swin-Tiny model with adapted classification head.

    Raises:
        PretrainedWeightDownloadError: If pretrained=True and the weights cannot be downloaded/loaded.
    """
    if pretrained:
        try:
            weights = Swin_T_Weights.DEFAULT
            model = models.swin_t(weights=weights)
        except Exception as err:
            error_msg = (
                "\n[ERROR] Failed to load official pretrained Swin-Tiny weights from PyTorch/Torchvision.\n"
                f"Underlying error: {err}\n\n"
                "Explanation:\n"
                "1. Pretrained Swin-Tiny weights (~114 MB) are fetched from PyTorch's official repository:\n"
                f"   {Swin_T_Weights.DEFAULT.url}\n"
                "2. Please ensure you have an active internet connection to download and cache these weights.\n"
                "3. We DO NOT silently fallback to random initialization because that would invalidate the "
                "pretrained transformer benchmark."
            )
            raise PretrainedWeightDownloadError(error_msg) from err
    else:
        # Explicit unweighted initialization (only if explicitly requested by caller)
        model = models.swin_t(weights=None)

    # In Swin-Tiny, model.head is nn.Linear(in_features=768, out_features=1000)
    in_features = model.head.in_features

    # Freeze backbone parameters if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head with custom head for num_classes
    if dropout_rate > 0.0:
        model.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        )
    else:
        model.head = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

    # Ensure head parameters are always trainable
    for param in model.head.parameters():
        param.requires_grad = True

    return model


def unfreeze_swin_stage4(model: nn.Module):
    """
    Utility to unfreeze the final transformer stage (features[7]) for fine-tuning.
    """
    if hasattr(model, "features") and len(model.features) > 7:
        for param in model.features[7].parameters():
            param.requires_grad = True
        if hasattr(model, "norm"):
            for param in model.norm.parameters():
                param.requires_grad = True


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

