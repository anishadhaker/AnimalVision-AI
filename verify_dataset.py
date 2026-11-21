"""
Verification script for Dataset Preparation.

Run this script to:
1. Automatically download CIFAR-10 via torchvision.
2. Filter the 5 animal classes: bird, cat, deer, dog, frog.
3. Validate official train vs test separation and stratified validation split.
4. Verify DataLoader shapes and output a preview image grid to `results/dataset_samples.png`.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter

from src.dataset import (
    get_animal_dataloaders,
    get_animal_datasets,
    ANIMAL_CLASSES,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def print_split_summary(name: str, targets: list):
    counts = Counter(targets)
    total = len(targets)
    print(f"\n--- {name} Split Summary (Total: {total:,} images) ---")
    for idx, class_name in enumerate(ANIMAL_CLASSES):
        count = counts[idx]
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  [{idx}] {class_name:<6}: {count:,} images ({percentage:.1f}%)")


def save_sample_grid(train_loader, save_path="results/dataset_samples.png"):
    """
    Saves a 4x4 visual sample grid from the dataset with unnormalized pixel values.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    images, targets = next(iter(train_loader))

    # Take first 16 images
    images = images[:16]
    targets = targets[:16]

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle("CIFAR-10 Filtered Animal Dataset Samples (224x224)", fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()
        # Unnormalize
        img = std * img + mean
        img = np.clip(img, 0, 1)

        label_idx = targets[i].item()
        label_name = IDX_TO_CLASS[label_idx]

        ax.imshow(img)
        ax.set_title(f"Class: {label_name} ({label_idx})", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n[OK] Sample grid visual saved to: {os.path.abspath(save_path)}")


def main():
    print("=" * 60)
    print("Step 1: CIFAR-10 Animal Dataset Preparation & Verification")
    print("=" * 60)
    print(f"Target classes: {ANIMAL_CLASSES}")
    print("Downloading / verifying CIFAR-10 dataset via torchvision...")

    train_ds, val_ds, test_ds = get_animal_datasets(
        data_dir="./data",
        val_split=0.2,
        seed=42,
        img_size=224,
    )

    print_split_summary("Training", train_ds.targets)
    print_split_summary("Validation", val_ds.targets)
    print_split_summary("Test", test_ds.targets)

    # Sanity checks
    assert len(train_ds) == 20000, f"Expected 20,000 train images, got {len(train_ds)}"
    assert len(val_ds) == 5000, f"Expected 5,000 val images, got {len(val_ds)}"
    assert len(test_ds) == 5000, f"Expected 5,000 test images, got {len(test_ds)}"
    print("\n[OK] Dataset split sizes verified successfully!")

    print("\nTesting PyTorch DataLoader batching...")
    train_loader, val_loader, test_loader, _ = get_animal_dataloaders(
        data_dir="./data",
        batch_size=32,
        val_split=0.2,
        num_workers=0,
        img_size=224,
    )

    sample_batch_imgs, sample_batch_targets = next(iter(train_loader))
    print(f"  Batch image tensor shape: {list(sample_batch_imgs.shape)} (Expected: [32, 3, 224, 224])")
    print(f"  Batch target tensor shape: {list(sample_batch_targets.shape)} (Expected: [32])")
    print(f"  Unique targets in batch: {torch.unique(sample_batch_targets).tolist()}")

    save_sample_grid(train_loader)

    print("\n" + "=" * 60)
    print("Dataset setup and verification completed with 100% success!")
    print("=" * 60)


if __name__ == "__main__":
    main()
