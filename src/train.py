"""
Training Pipeline for CNN and Swin-Tiny Animal Classification.

Features:
- Supports training either 'cnn' (baseline) or 'swin' (Swin-Tiny) via CLI arguments.
- Strictly reuses existing train and validation splits (never touches test set).
- Deterministic, stratified dataset limiting via --train-limit and --val-limit.
- Configurable image resolution (--img-size 96 or 224) for CPU acceleration (~6x faster at 96x96).
- Granular progress output before, during, and after every batch/epoch with flush=True.
- Precise batch-by-batch timing to diagnose bottlenecks (data loading vs forward/backward pass).
- Uses CrossEntropyLoss and AdamW optimizer with ReduceLROnPlateau scheduler.
- Tracks epoch loss, accuracy, and timing on actual data batches (no fabricated numbers).
- Saves best checkpoint according to validation accuracy in models/.
- Logs epoch-by-epoch training history as CSV in results/.
- Saves experiment metadata JSON with exact sample counts, img_size, and configuration.
- Supports early stopping and smoke-test mode (--smoke-test).
- Preserves official pretrained weights for Swin-Tiny without accidental reinitialization.
- Supports --freeze-backbone for CPU-friendly Swin-Tiny transfer learning.
"""

import os
import sys
import time
import json
import argparse
import csv
from typing import Tuple, Dict, Any, List, Optional
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import (
    get_animal_datasets,
    ANIMAL_CLASSES,
)
from src.cnn_model import build_cnn_model, build_cnn_v2_model, count_parameters as count_cnn_params
from src.swin_model import (
    build_swin_model,
    count_parameters as count_swin_params,
    PretrainedWeightDownloadError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN Baseline, CNN v2, or Swin-Tiny on 5 CIFAR-10 Animal Classes."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn",
        choices=["cnn", "swin", "cnn_v2"],
        help="Model architecture to train: 'cnn', 'cnn_v2', or 'swin' (default: 'cnn').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and validation loaders (default: 32).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=96,
        choices=[96, 224],
        help="Input image resolution: 96 (CPU-friendly, ~6x faster) or 224 (standard). Default: 96.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate. Default: 1e-3 for CNN, 1e-4 for Swin.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="For Swin model: freeze transformer backbone and train head only (recommended on CPU).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience (epochs without validation accuracy improvement). Default: 3.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Deterministic stratified limit on training samples (e.g. 500 or 5000 for CPU).",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=None,
        help="Deterministic stratified limit on validation samples (e.g. 100 or 1000 for CPU).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny CPU smoke test (64 train samples, 32 val samples, 1 epoch).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Print batch progress every N batches (default: 1 for immediate live feedback).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path where CIFAR-10 data resides (default: ./data).",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory to save model checkpoints (default: ./models).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save training CSV logs (default: ./results).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="Number of CPU threads to use for PyTorch (default: system default).",
    )
    return parser.parse_args()


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_stratified_subset_indices(targets: List[int], limit: Optional[int], seed: int = 42) -> List[int]:
    """
    Deterministically selects `limit` indices using stratified sampling across class labels.
    Guarantees equal class distribution and identical index subsets across models.
    """
    total = len(targets)
    if limit is None or limit >= total:
        return list(range(total))

    indices = np.arange(total)
    subset_idx, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        stratify=targets,
    )
    return sorted(subset_idx.tolist())


def format_class_distribution(targets: List[int], indices: List[int]) -> Dict[str, int]:
    """
    Returns counts per animal class for the selected subset of indices.
    """
    selected_targets = [targets[i] for i in indices]
    counts = Counter(selected_targets)
    return {name: counts.get(idx, 0) for idx, name in enumerate(ANIMAL_CLASSES)}


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_interval: int = 1,
) -> Tuple[float, float]:
    """
    Runs one training epoch on actual batches with live per-batch progress.
    Returns:
        (avg_loss, accuracy_percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(dataloader)
    epoch_start_time = time.time()

    print(
        f"\n--- [Epoch {epoch:02d}/{total_epochs:02d}] Training Started "
        f"({total_batches} batches of size {dataloader.batch_size}) ---",
        flush=True,
    )

    for step, (images, targets) in enumerate(dataloader, 1):
        batch_start = time.time()
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        batch_size = images.size(0)
        running_loss += batch_loss * batch_size

        _, preds = torch.max(outputs, 1)
        batch_correct = (preds == targets).sum().item()
        correct += batch_correct
        total += batch_size

        batch_time = time.time() - batch_start
        elapsed = time.time() - epoch_start_time

        # Granular live batch progress output
        if step % log_interval == 0 or step == total_batches:
            current_loss = running_loss / total
            current_acc = (correct / total) * 100.0
            print(
                f"  [Batch {step:03d}/{total_batches:03d}] "
                f"Batch Loss: {batch_loss:.4f} | Running Acc: {current_acc:.1f}% | "
                f"Batch Time: {batch_time:.2f}s | Elapsed: {elapsed:.1f}s",
                flush=True,
            )

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    epoch_duration = time.time() - epoch_start_time
    print(
        f"--- [Epoch {epoch:02d}/{total_epochs:02d}] Training Finished in {epoch_duration:.1f}s "
        f"-> Avg Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% ---",
        flush=True,
    )
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_interval: int = 1,
) -> Tuple[float, float]:
    """
    Runs evaluation on validation DataLoader in eval mode without gradient updates.
    Returns:
        (avg_loss, accuracy_percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    val_start = time.time()
    total_batches = len(dataloader)

    print(f"\n--- [Epoch {epoch:02d}/{total_epochs:02d}] Validation Started ({total_batches} batches) ---", flush=True)

    with torch.no_grad():
        for step, (images, targets) in enumerate(dataloader, 1):
            batch_start = time.time()
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_loss = loss.item()
            batch_size = images.size(0)
            running_loss += batch_loss * batch_size

            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += batch_size

            batch_time = time.time() - batch_start
            if step % log_interval == 0 or step == total_batches:
                current_acc = (correct / total) * 100.0
                print(
                    f"  [Val Batch {step:03d}/{total_batches:03d}] "
                    f"Batch Loss: {batch_loss:.4f} | Running Val Acc: {current_acc:.1f}% | Batch Time: {batch_time:.2f}s",
                    flush=True,
                )

    val_elapsed = time.time() - val_start
    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(
        f"--- [Epoch {epoch:02d}/{total_epochs:02d}] Validation Finished in {val_elapsed:.1f}s "
        f"-> Avg Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}% ---",
        flush=True,
    )
    return epoch_loss, epoch_acc


def save_checkpoint(state: Dict[str, Any], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    print(f"  [Checkpoint Saved] -> {os.path.abspath(save_path)}", flush=True)


def save_history_csv(history: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "duration_seconds"]
    with open(save_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"  [History Saved]    -> {os.path.abspath(save_path)}", flush=True)


def save_config_json(config: Dict[str, Any], save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode="w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"  [Config Saved]     -> {os.path.abspath(save_path)}", flush=True)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 65, flush=True)
    print(f"Experiment Training Pipeline: Model = {args.model.upper()}", flush=True)
    print("=" * 65, flush=True)
    print(f"Compute Device:   {device} (PyTorch CPU threads: {torch.get_num_threads()})", flush=True)
    print(f"Random Seed:      {args.seed}", flush=True)
    print(f"Image Resolution: {args.img_size}x{args.img_size}", flush=True)

    # Set default model-specific learning rate if not supplied
    if args.lr is None:
        args.lr = 1e-3 if args.model in ["cnn", "cnn_v2"] else 1e-4
    print(f"Learning Rate:    {args.lr}", flush=True)
    print(f"Batch Size:       {args.batch_size}", flush=True)
    print(f"Planned Epochs:   {args.epochs}", flush=True)

    # 2. Load Datasets
    print("\n[1/4] Preparing Dataset Splits...", flush=True)
    t_data_start = time.time()
    full_train_dataset, full_val_dataset, _ = get_animal_datasets(
        data_dir=args.data_dir,
        val_split=0.2,
        seed=args.seed,
        img_size=args.img_size,
    )
    print(
        f"  Verified Full Splits Loaded in {time.time()-t_data_start:.2f}s "
        f"(Train: {len(full_train_dataset):,}, Val: {len(full_val_dataset):,})",
        flush=True,
    )

    # Determine subset indices deterministically and stratified
    if args.smoke_test:
        train_indices = get_stratified_subset_indices(full_train_dataset.targets, limit=64, seed=args.seed)
        val_indices = get_stratified_subset_indices(full_val_dataset.targets, limit=32, seed=args.seed)
        print("  [SMOKE-TEST MODE] Active: Train=64, Val=32, 1 epoch", flush=True)
        args.epochs = 1
    elif args.train_limit or args.val_limit:
        train_indices = get_stratified_subset_indices(full_train_dataset.targets, limit=args.train_limit, seed=args.seed)
        val_indices = get_stratified_subset_indices(full_val_dataset.targets, limit=args.val_limit, seed=args.seed)
        print(f"  [LIMITED CPU EXPERIMENT] Active: Train limit={len(train_indices):,}, Val limit={len(val_indices):,}", flush=True)
    else:
        train_indices = list(range(len(full_train_dataset)))
        val_indices = list(range(len(full_val_dataset)))

    train_class_counts = format_class_distribution(full_train_dataset.targets, train_indices)
    val_class_counts = format_class_distribution(full_val_dataset.targets, val_indices)

    print(f"  Selected Train Samples: {len(train_indices):,} -> Distribution: {train_class_counts}", flush=True)
    print(f"  Selected Val Samples:   {len(val_indices):,} -> Distribution: {val_class_counts}", flush=True)

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Train Batches per epoch: {len(train_loader)}", flush=True)
    print(f"  Val Batches per epoch:   {len(val_loader)}", flush=True)

    # 3. Build Model
    print(f"\n[2/4] Instantiating Model Architecture ({args.model.upper()})...", flush=True)
    try:
        if args.model == "cnn":
            model = build_cnn_model(num_classes=5)
            total_params, trainable_params = count_cnn_params(model)
            checkpoint_name = "cnn_best.pth"
            history_name = "cnn_history.csv"
            config_name = "cnn_config.json"
        elif args.model == "cnn_v2":
            model = build_cnn_v2_model(num_classes=5, dropout_rate=0.15)
            total_params, trainable_params = count_cnn_params(model)
            checkpoint_name = "cnn_v2_best.pth"
            history_name = "cnn_v2_history.csv"
            config_name = "cnn_v2_config.json"
        else:
            model = build_swin_model(
                num_classes=5,
                pretrained=True,
                freeze_backbone=args.freeze_backbone,
            )
            total_params, trainable_params = count_swin_params(model)
            checkpoint_name = "swin_tiny_best.pth"
            history_name = "swin_tiny_history.csv"
            config_name = "swin_tiny_config.json"

        model = model.to(device)
        print(f"  Total Parameters:     {total_params:,}", flush=True)
        print(f"  Trainable Parameters: {trainable_params:,}", flush=True)
        if args.model == "swin" and args.freeze_backbone:
            print("  [Info] Swin Backbone is FROZEN. Only classification head will update.", flush=True)

    except PretrainedWeightDownloadError as err:
        print(f"\n{err}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as err:
        print(f"\n[ERROR] Failed to build {args.model}: {err}", file=sys.stderr, flush=True)
        sys.exit(1)

    # 4. Setup Loss, Optimizer, and LR Scheduler
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if args.model == "cnn_v2":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-5,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
        )

    checkpoint_path = os.path.join(args.models_dir, checkpoint_name)
    history_path = os.path.join(args.results_dir, history_name)
    config_path = os.path.join(args.results_dir, config_name)

    # Automatically set log_interval
    log_interval = args.log_interval

    # 5. Training Loop
    print("\n[3/4] Starting Training Loop...", flush=True)
    best_val_acc = -1.0
    patience_counter = 0
    history = []
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=log_interval,
        )

        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=log_interval,
        )

        epoch_duration = time.time() - epoch_start
        if args.model == "cnn_v2":
            scheduler.step()
        else:
            scheduler.step(val_acc)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "duration_seconds": round(epoch_duration, 1),
        })

        print(
            f"\nEpoch [{epoch:02d}/{args.epochs:02d}] Summary -> "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Total Epoch Time: {epoch_duration:.1f}s",
            flush=True,
        )

        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_type": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "class_names": ANIMAL_CLASSES,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "img_size": args.img_size,
                    "train_class_counts": train_class_counts,
                    "val_class_counts": val_class_counts,
                    "hyperparameters": vars(args),
                },
                save_path=checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience and not args.smoke_test:
                print(
                    f"\n[Early Stopping Triggered] No validation accuracy improvement for "
                    f"{args.patience} consecutive epochs. Stopping to conserve CPU compute.",
                    flush=True,
                )
                break

    total_training_time = time.time() - start_total_time
    minutes = int(total_training_time // 60)
    seconds = int(total_training_time % 60)

    # 6. Save history CSV and experiment config JSON
    print("\n[4/4] Finalizing Logs...", flush=True)
    save_history_csv(history, history_path)

    config_record = {
        "model": args.model,
        "seed": args.seed,
        "device": str(device),
        "img_size": args.img_size,
        "total_train_samples": len(train_dataset),
        "total_val_samples": len(val_dataset),
        "train_class_distribution": train_class_counts,
        "val_class_distribution": val_class_counts,
        "epochs_planned": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "freeze_backbone": args.freeze_backbone if args.model == "swin" else False,
        "best_val_accuracy": round(best_val_acc, 2),
        "total_duration_seconds": round(total_training_time, 2),
        "preliminary_experiment": bool(args.train_limit or args.val_limit or args.smoke_test),
        "checkpoint_file": checkpoint_name,
        "history_file": history_name,
    }
    save_config_json(config_record, config_path)

    print("\n" + "=" * 65, flush=True)
    print("TRAINING SESSION SUMMARY (Actual Measured Metrics):", flush=True)
    print(f"  Model Architecture:       {args.model.upper()}", flush=True)
    print(f"  Image Resolution:         {args.img_size}x{args.img_size}", flush=True)
    print(f"  Train Samples Used:       {len(train_dataset):,}", flush=True)
    print(f"  Validation Samples Used:  {len(val_dataset):,}", flush=True)
    print(f"  Epochs Completed:         {len(history)} / {args.epochs}", flush=True)
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f"  Total Duration:           {minutes}m {seconds}s ({total_training_time:.1f}s)", flush=True)
    print(f"  Best Checkpoint:          {os.path.abspath(checkpoint_path)}", flush=True)
    print(f"  History Log:              {os.path.abspath(history_path)}", flush=True)
    print(f"  Config Metadata:          {os.path.abspath(config_path)}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
