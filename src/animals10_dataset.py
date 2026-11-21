"""
Dataset module for Animals-10 (10-Class Natural Animal Classification).

Classes (10 classes, alphabetical):
  0: butterfly
  1: cat
  2: chicken
  3: cow
  4: dog
  5: elephant
  6: horse
  7: sheep
  8: spider
  9: squirrel

This module:
1. Loads pre-generated 70/15/15 stratified splits from data/animals-10/splits/{train,val,test}.
2. Applies standard 224x224 resizing and ImageNet normalization.
3. Provides training-only data augmentations (flip, rotation, jitter) while keeping val/test deterministic.
4. Computes class weights strictly from the training split for balanced CrossEntropy loss.
5. Returns PyTorch DataLoaders with configurable batch sizes and num_workers.
"""

import os
from typing import Tuple, List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


ANIMALS10_CLASSES: List[str] = [
    "butterfly",
    "cat",
    "chicken",
    "cow",
    "dog",
    "elephant",
    "horse",
    "sheep",
    "spider",
    "squirrel",
]

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(ANIMALS10_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(ANIMALS10_CLASSES)}

# ImageNet normalization standard (used across ResNet and Swin Transformer)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_animals10_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Constructs aspect-ratio-preserving training and evaluation transform pipelines.

    Args:
        img_size (int): Target square dimension (default: 224).

    Returns:
        (train_transform, eval_transform)
    """
    # Scale shorter side to 256 for 224 crops, maintaining natural aspect ratio
    resize_dim = int(img_size * 256 / 224)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def get_animals10_datasets(
    splits_dir: str = "./data/animals-10/splits",
    img_size: int = 224,
) -> Tuple[ImageFolder, ImageFolder, ImageFolder]:
    """
    Loads Train, Validation, and Test datasets using ImageFolder from splits directory.

    Args:
        splits_dir (str): Base directory containing train/, val/, and test/ subdirectories.
        img_size (int): Target resolution (default: 224).

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    train_path = os.path.join(splits_dir, "train")
    val_path = os.path.join(splits_dir, "val")
    test_path = os.path.join(splits_dir, "test")

    for path, split_name in [(train_path, "Train"), (val_path, "Validation"), (test_path, "Test")]:
        if not os.path.isdir(path):
            raise FileNotFoundError(f"{split_name} split directory not found at: {os.path.abspath(path)}")

    train_transform, eval_transform = get_animals10_transforms(img_size=img_size)

    train_dataset = ImageFolder(root=train_path, transform=train_transform)
    val_dataset = ImageFolder(root=val_path, transform=eval_transform)
    test_dataset = ImageFolder(root=test_path, transform=eval_transform)

    # Safety check: Verify that ImageFolder's discovered class order matches our canonical mapping
    assert train_dataset.classes == ANIMALS10_CLASSES, (
        f"Discovered classes {train_dataset.classes} do not match expected {ANIMALS10_CLASSES}"
    )

    return train_dataset, val_dataset, test_dataset


def compute_training_class_weights(
    train_dataset: ImageFolder,
    method: str = "balanced",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes class weights STRICTLY from the training split targets to handle class imbalance.
    Does NOT inspect validation or test sets.

    Formula (balanced / inverse frequency):
        w_c = N_total / (num_classes * count_c)

    Args:
        train_dataset (ImageFolder): Training split dataset.
        method (str): Weighting strategy ('balanced' or 'normalized').

    Returns:
        (weights_tensor, weights_dict)
    """
    targets = [target for _, target in train_dataset.samples]
    total_samples = len(targets)
    num_classes = len(ANIMALS10_CLASSES)

    counts = np.bincount(targets, minlength=num_classes)
    weights = total_samples / (num_classes * counts.astype(np.float64))

    if method == "normalized":
        # Normalize so average weight equals 1.0
        weights = weights / np.mean(weights)

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_dict = {ANIMALS10_CLASSES[i]: round(float(weights[i]), 4) for i in range(num_classes)}

    return weights_tensor, weights_dict


def get_animals10_dataloaders(
    splits_dir: str = "./data/animals-10/splits",
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    High-level factory function providing DataLoader instances for Train, Val, and Test.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    train_ds, val_ds, test_ds = get_animals10_datasets(splits_dir=splits_dir, img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, ANIMALS10_CLASSES
