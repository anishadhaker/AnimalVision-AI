"""
Dataset module for CIFAR-10 Animal Image Classification.

Selected Classes (5 classes):
  - bird (original CIFAR-10 index: 2) -> mapped to 0
  - cat  (original CIFAR-10 index: 3) -> mapped to 1
  - deer (original CIFAR-10 index: 4) -> mapped to 2
  - dog  (original CIFAR-10 index: 5) -> mapped to 3
  - frog (original CIFAR-10 index: 6) -> mapped to 4

This module:
1. Downloads CIFAR-10 automatically via torchvision.
2. Filters out the 5 selected animal classes.
3. Keeps official CIFAR-10 train and test splits separate.
4. Generates a stratified validation split (default 20%) from the training data.
5. Provides PyTorch DataLoaders with identical split assignments for both CNN and Swin models.
"""

import os
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


# Constants
ANIMAL_CLASSES: List[str] = ["bird", "cat", "deer", "dog", "frog"]

# Mapping from original CIFAR-10 class indices to new animal class indices (0..4)
ORIGINAL_TO_ANIMAL_MAP: Dict[int, int] = {
    2: 0,  # bird
    3: 1,  # cat
    4: 2,  # deer
    5: 3,  # dog
    6: 4,  # frog
}

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(ANIMAL_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(ANIMAL_CLASSES)}

# ImageNet normalization standard (used by Swin Transformer and standard CNN backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FilteredAnimalDataset(Dataset):
    """
    A PyTorch Dataset wrapper that stores images and remaps CIFAR-10 animal targets.
    """
    def __init__(self, images: np.ndarray, targets: List[int], transform: Optional[transforms.Compose] = None):
        """
        Args:
            images (np.ndarray): Array of shape (N, H, W, C) containing image data.
            targets (List[int]): List of mapped labels (0..4).
            transform (transforms.Compose, optional): Image augmentation / transform pipeline.
        """
        assert len(images) == len(targets), "Images and targets must have matching lengths."
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]
        target = self.targets[idx]

        # Convert numpy array to PIL Image for torchvision transforms
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Generates training and evaluation transforms.
    
    Args:
        img_size (int): Target square dimension (224 for standard Swin and high-res CNN).
        
    Returns:
        (train_transform, eval_transform)
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def filter_animal_subset(cifar_dataset: torchvision.datasets.CIFAR10) -> Tuple[np.ndarray, List[int]]:
    """
    Filters raw CIFAR-10 dataset to keep only the 5 selected animal classes and remaps labels.
    
    Returns:
        filtered_images: ndarray of shape (N, 32, 32, 3)
        filtered_targets: List of remapped integer labels in [0, 4]
    """
    raw_images = cifar_dataset.data
    raw_targets = cifar_dataset.targets

    filtered_images = []
    filtered_targets = []

    for img, orig_label in zip(raw_images, raw_targets):
        if orig_label in ORIGINAL_TO_ANIMAL_MAP:
            filtered_images.append(img)
            filtered_targets.append(ORIGINAL_TO_ANIMAL_MAP[orig_label])

    return np.array(filtered_images), filtered_targets


def get_animal_datasets(
    data_dir: str = "./data",
    val_split: float = 0.2,
    seed: int = 42,
    img_size: int = 224,
) -> Tuple[FilteredAnimalDataset, FilteredAnimalDataset, FilteredAnimalDataset]:
    """
    Downloads CIFAR-10, filters the 5 animal classes, creates a stratified train/val split,
    and returns PyTorch Dataset instances.
    
    Args:
        data_dir (str): Directory where CIFAR-10 raw data will be cached.
        val_split (float): Fraction of official training set to allocate for validation.
        seed (int): Random seed for reproducible splitting.
        img_size (int): Input image resolution (e.g. 224x224).
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    os.makedirs(data_dir, exist_ok=True)

    # 1. Download official CIFAR-10 splits automatically
    raw_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    raw_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)

    # 2. Filter to 5 animal classes
    train_images_all, train_targets_all = filter_animal_subset(raw_train)
    test_images, test_targets = filter_animal_subset(raw_test)

    # 3. Stratified split of official training data into train and validation sets
    indices = np.arange(len(train_images_all))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        random_state=seed,
        stratify=train_targets_all,
    )

    train_images = train_images_all[train_idx]
    train_targets = [train_targets_all[i] for i in train_idx]

    val_images = train_images_all[val_idx]
    val_targets = [train_targets_all[i] for i in val_idx]

    # 4. Build transforms
    train_transform, eval_transform = get_transforms(img_size=img_size)

    # 5. Create Dataset objects
    train_dataset = FilteredAnimalDataset(train_images, train_targets, transform=train_transform)
    val_dataset = FilteredAnimalDataset(val_images, val_targets, transform=eval_transform)
    test_dataset = FilteredAnimalDataset(test_images, test_targets, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset


def get_animal_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    img_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    High-level factory function to retrieve DataLoader instances for Train, Val, and Test.
    
    Args:
        data_dir (str): Storage path for dataset.
        batch_size (int): Batch size for loaders.
        val_split (float): Proportion of training set reserved for validation.
        num_workers (int): Worker threads (0 recommended for Windows stability).
        seed (int): Seed for deterministic splits.
        img_size (int): Image size (default 224x224).
        
    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    train_ds, val_ds, test_ds = get_animal_datasets(
        data_dir=data_dir,
        val_split=val_split,
        seed=seed,
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, ANIMAL_CLASSES
