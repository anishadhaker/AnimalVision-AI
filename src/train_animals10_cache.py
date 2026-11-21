"""
Phase 3A: Frozen-Backbone Feature-Based Transfer Learning for Animals-10.

Protocol:
1. Feature Extraction:
   - ResNet-18: Frozen ImageNet-1K backbone -> 512-dim pooled embeddings.
   - Swin-Tiny: Frozen ImageNet-1K backbone -> 768-dim pooled embeddings.
   - Extracted on Training split (16,487 samples) and Validation split (3,533 samples).
   - Test split (3,534 samples) is strictly excluded and untouched.
   - Training augmentation (RandomResizedCrop, Flip, Jitter) is sampled once during extraction.
   - Feature caches are saved to data/animals-10/cache/.

2. Classification Head Training:
   - Linear head: Linear(512, 10) for ResNet-18, Linear(768, 10) for Swin-Tiny.
   - Weighted CrossEntropyLoss using exact training-split inverse-frequency class weights.
   - AdamW optimizer (lr=1e-3, weight_decay=1e-4) with CosineAnnealingLR.
   - 10 training epochs with validation accuracy and Macro F1 tracking.
   - Saves best checkpoint to models/resnet18_animals10_best.pth and models/swin_tiny_animals10_best.pth.
   - Each checkpoint includes full model_state_dict (backbone + trained head) for seamless inference.
"""

import os
import sys
import time
import json
import csv
import argparse
from typing import Tuple, Dict, Any, List

import numpy as np
import psutil
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

WORKSPACE = r"c:\Users\Laptop\OneDrive\Desktop\Swin vs CNN"
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from src.animals10_dataset import (
    ANIMALS10_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_animals10_datasets,
    compute_training_class_weights,
)
from src.resnet_model import build_resnet18_model, count_parameters as count_resnet_params
from src.swin_model import build_swin_model, count_parameters as count_swin_params

CACHE_DIR = os.path.join(WORKSPACE, "data", "animals-10", "cache")
MODELS_DIR = os.path.join(WORKSPACE, "models")
RESULTS_DIR = os.path.join(WORKSPACE, "results", "animals10")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_features(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
    desc: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Passes a dataset through a frozen feature extractor in eval mode.
    Returns:
        (features_tensor, targets_tensor)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    features_list = []
    targets_list = []

    total_batches = len(loader)
    print(f"  Starting feature extraction for [{desc}] ({len(dataset):,} samples, {total_batches} batches)...", flush=True)

    t0 = time.time()
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device)
            feats = model(images)
            features_list.append(feats.cpu())
            targets_list.append(targets.cpu())

            if step % 50 == 0 or step == total_batches:
                elapsed = time.time() - t0
                pct = (step / total_batches) * 100.0
                rate = (step * batch_size) / max(elapsed, 0.001)
                proc_ram = psutil.Process().memory_info().rss / (1024**2)
                print(
                    f"    [{desc}] Batch {step:03d}/{total_batches} ({pct:5.1f}%) | "
                    f"Elapsed: {elapsed:5.1f}s | Speed: {rate:4.1f} img/s | Process RAM: {proc_ram:5.1f} MB",
                    flush=True,
                )

    all_features = torch.cat(features_list, dim=0)
    all_targets = torch.cat(targets_list, dim=0)
    print(f"  Extraction finished for [{desc}]: Features shape {list(all_features.shape)} in {time.time()-t0:.1f}s", flush=True)
    return all_features, all_targets


def get_or_create_feature_cache(
    model_name: str,
    batch_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Retrieves cached features from disk or extracts them if not yet cached.
    """
    train_cache_path = os.path.join(CACHE_DIR, f"{model_name}_train_features.pt")
    val_cache_path = os.path.join(CACHE_DIR, f"{model_name}_val_features.pt")

    if os.path.exists(train_cache_path) and os.path.exists(val_cache_path):
        print(f"\n[Feature Cache Found] Loading existing cached embeddings for {model_name} from {CACHE_DIR}...", flush=True)
        train_data = torch.load(train_cache_path)
        val_data = torch.load(val_cache_path)
        return train_data["features"], train_data["targets"], val_data["features"], val_data["targets"]

    print(f"\n[Extracting Features] Caches not found. Performing one-time feature extraction for {model_name}...", flush=True)
    train_ds, val_ds, _ = get_animals10_datasets(
        splits_dir=os.path.join(WORKSPACE, "data", "animals-10", "splits"),
        img_size=224,
    )

    if model_name == "resnet18":
        extractor = build_resnet18_model(num_classes=10, pretrained=True, freeze_backbone=True)
        extractor.fc = nn.Identity()
    elif model_name == "swin_tiny":
        extractor = build_swin_model(num_classes=10, pretrained=True, freeze_backbone=True)
        extractor.head = nn.Identity()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    train_feats, train_targets = extract_features(extractor, train_ds, batch_size=batch_size, desc="TRAIN")
    val_feats, val_targets = extract_features(extractor, val_ds, batch_size=batch_size, desc="VAL")

    # Save to disk
    torch.save({"features": train_feats, "targets": train_targets}, train_cache_path)
    torch.save({"features": val_feats, "targets": val_targets}, val_cache_path)
    print(f"  Saved train feature cache to: {train_cache_path} ({os.path.getsize(train_cache_path)/(1024**2):.1f} MB)", flush=True)
    print(f"  Saved val feature cache to:   {val_cache_path} ({os.path.getsize(val_cache_path)/(1024**2):.1f} MB)", flush=True)

    return train_feats, train_targets, val_feats, val_targets


def train_classification_head(
    model_name: str,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    extract_batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Trains the linear classification head on cached features.
    """
    print("=" * 70, flush=True)
    print(f"PHASE 3A TRAINING: Model = {model_name.upper()} (Animals-10, 10 Classes)", flush=True)
    print("Strategy: Frozen ImageNet-1K Backbone Transfer Learning", flush=True)
    print("=" * 70, flush=True)

    t_start = time.time()

    # 1. Obtain Feature Caches
    train_feats, train_targets, val_feats, val_targets = get_or_create_feature_cache(
        model_name=model_name,
        batch_size=extract_batch_size,
    )

    in_dim = train_feats.shape[1]
    num_classes = len(ANIMALS10_CLASSES)

    # 2. Compute training class weights strictly from train targets
    counts = np.bincount(train_targets.numpy(), minlength=num_classes)
    total_train = len(train_targets)
    weights = total_train / (num_classes * counts.astype(np.float64))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    train_dataset = TensorDataset(train_feats, train_targets)
    val_dataset = TensorDataset(val_feats, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Instantiate Linear Head
    head = nn.Linear(in_dim, num_classes, bias=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\nTraining Classification Head (Input Dim: {in_dim} -> Output Dim: {num_classes})...", flush=True)
    print(f"Hyperparameters: epochs={epochs}, lr={lr}, weight_decay={weight_decay}, batch_size={batch_size}", flush=True)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_head_state = None
    history = []

    for epoch in range(1, epochs + 1):
        ep_start = time.time()

        # Training Epoch
        head.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for feats, targets in train_loader:
            optimizer.zero_grad()
            outputs = head(feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=-1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        train_loss = train_loss_sum / train_total
        train_acc = (train_correct / train_total) * 100.0

        # Validation Epoch
        head.eval()
        val_loss_sum = 0.0
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for feats, targets in val_loader:
                outputs = head(feats)
                loss = criterion(outputs, targets)
                val_loss_sum += loss.item() * targets.size(0)

                preds = outputs.argmax(dim=-1)
                val_preds_list.extend(preds.cpu().numpy().tolist())
                val_targets_list.extend(targets.cpu().numpy().tolist())

        val_total = len(val_targets_list)
        val_loss = val_loss_sum / val_total
        val_correct = sum(p == t for p, t in zip(val_preds_list, val_targets_list))
        val_acc = (val_correct / val_total) * 100.0
        val_macro_f1 = f1_score(val_targets_list, val_preds_list, average="macro") * 100.0

        ep_duration = time.time() - ep_start
        is_best = val_acc > best_val_acc

        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_macro_f1": round(val_macro_f1, 2),
            "duration_seconds": round(ep_duration, 2),
        })

        best_flag = " [*BEST*]" if is_best else ""
        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | Val F1: {val_macro_f1:5.2f}% | "
            f"Time: {ep_duration:4.1f}s{best_flag}",
            flush=True,
        )

    total_training_duration = time.time() - t_start

    # 4. Assemble Full Model with Trained Head and Save Checkpoint
    checkpoint_filename = f"{model_name}_animals10_best.pth"
    checkpoint_path = os.path.join(MODELS_DIR, checkpoint_filename)

    print(f"\nAssembling complete full-pipeline model for {checkpoint_filename}...", flush=True)
    if model_name == "resnet18":
        full_model = build_resnet18_model(num_classes=num_classes, pretrained=True, freeze_backbone=True)
        full_model.fc.load_state_dict(best_head_state)
        total_p, trainable_p = count_resnet_params(full_model)
    elif model_name == "swin_tiny":
        full_model = build_swin_model(num_classes=num_classes, pretrained=True, freeze_backbone=True)
        full_model.head.load_state_dict(best_head_state)
        total_p, trainable_p = count_swin_params(full_model)

    checkpoint_payload = {
        "model_name": model_name,
        "dataset": "Rapidata/Animals-10",
        "num_classes": num_classes,
        "class_names": ANIMALS10_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "img_size": 224,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 2),
        "best_val_loss": round(best_val_loss, 4),
        "total_parameters": total_p,
        "trainable_parameters": trainable_p,
        "training_protocol": "Frozen-Backbone Feature-Based Transfer Learning",
        "model_state_dict": full_model.state_dict(),
        "head_state_dict": best_head_state,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    print(f"  Saved best checkpoint to: {checkpoint_path} ({os.path.getsize(checkpoint_path)/(1024**2):.1f} MB)", flush=True)

    # 5. Save History CSV and Configuration JSON
    history_csv_path = os.path.join(RESULTS_DIR, f"{model_name}_history.csv")
    with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "duration_seconds"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    print(f"  Saved training history to: {history_csv_path}", flush=True)

    config_json_path = os.path.join(RESULTS_DIR, f"{model_name}_config.json")
    config_payload = {
        "model": model_name,
        "dataset": "Rapidata/Animals-10",
        "num_classes": 10,
        "img_size": 224,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "best_val_accuracy": round(best_val_acc, 2),
        "best_val_loss": round(best_val_loss, 4),
        "best_epoch": best_epoch,
        "total_duration_seconds": round(total_training_duration, 1),
        "total_parameters": total_p,
        "trainable_parameters": trainable_p,
        "frozen_parameters": total_p - trainable_p,
        "training_protocol": "Frozen-Backbone Feature-Based Transfer Learning (Pre-computed Feature Cache)",
        "checkpoint_file": checkpoint_filename,
        "history_file": f"{model_name}_history.csv",
    }
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)
    print(f"  Saved config metadata to: {config_json_path}", flush=True)

    return config_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3A: Animals-10 Feature-Based Transfer Learning")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "swin_tiny"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    args = parser.parse_args()

    train_classification_head(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        extract_batch_size=args.extract_batch_size,
    )
