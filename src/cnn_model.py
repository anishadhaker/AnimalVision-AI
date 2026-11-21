"""
CNN Baseline Model for 5-Class Animal Classification.

Architecture:
- 4 Convolutional blocks, each comprising:
  * 2D Convolution (3x3 kernel, padding=1)
  * Batch Normalization (stabilizes and accelerates training)
  * ReLU Activation
  * Max Pooling (2x2 kernel, stride 2)
- Adaptive Average Pooling (produces fixed 1x1 spatial feature maps)
- Classification Head:
  * Dropout (regularization to combat overfitting)
  * Fully Connected (Linear) layer: 256 -> 128
  * ReLU Activation
  * Dropout
  * Output Linear layer: 128 -> 5 classes
"""

from typing import Tuple
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard Convolutional block: Conv2D -> BatchNorm -> ReLU -> MaxPool2D.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AnimalCNN(nn.Module):
    """
    Lightweight, modular CNN baseline designed for 5-class animal image classification.
    Accepts input tensors of shape [batch_size, 3, H, W] (e.g. 224x224).
    """
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # Feature Extractor: 4-stage convolutional hierarchy
        # Input: [B, 3, 224, 224]
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # -> [B, 32, 112, 112]
            ConvBlock(32, 64),   # -> [B, 64, 56, 56]
            ConvBlock(64, 128),  # -> [B, 128, 28, 28]
            ConvBlock(128, 256), # -> [B, 256, 14, 14]
        )

        # Global pooling collapses spatial dimensions [B, 256, H, W] -> [B, 256, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input batch of shape [B, 3, H, W]
        Returns:
            torch.Tensor: Unnormalized class logits of shape [B, num_classes]
        """
        features = self.features(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits


def build_cnn_model(num_classes: int = 5, dropout_rate: float = 0.3) -> AnimalCNN:
    """
    Factory function to instantiate the CNN baseline model.
    """
    return AnimalCNN(num_classes=num_classes, dropout_rate=dropout_rate)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_parameters, trainable_parameters).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =====================================================================
# CNN v2: Residual Network Architecture (AnimalResNet)
# =====================================================================

class ResidualBlock(nn.Module):
    """
    Residual Block with two 3x3 Conv-BN-ReLU operations and projection shortcut.
    F(x) + shortcut(x) prevents vanishing gradients and deepens feature representation.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return self.relu(out)


class AnimalResNet(nn.Module):
    """
    CNN v2: Lightweight Residual Network (ResNet-lite) designed for 5-class animal classification.
    Features:
    - Initial Stem Conv (3 -> 32)
    - 4 Residual Stages with shortcut connections (32 -> 64 -> 128 -> 256)
    - Global Average Pooling (AdaptiveAvgPool2d to 1x1)
    - Fully Connected Classifier with milder dropout (0.15) to prevent underfitting
    Accepts [B, 3, H, W] (e.g. 96x96 or 224x224) and outputs [B, 5] logits.
    """
    def __init__(self, num_classes: int = 5, dropout_rate: float = 0.15):
        super().__init__()
        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 4 Residual Stages
        self.stage1 = ResidualBlock(32, 32, stride=1)
        self.stage2 = ResidualBlock(32, 64, stride=2)
        self.stage3 = ResidualBlock(64, 128, stride=2)
        self.stage4 = ResidualBlock(128, 256, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits


def build_cnn_v2_model(num_classes: int = 5, dropout_rate: float = 0.15) -> AnimalResNet:
    """
    Factory function to instantiate the improved CNN v2 (AnimalResNet) model.
    """
    return AnimalResNet(num_classes=num_classes, dropout_rate=dropout_rate)

