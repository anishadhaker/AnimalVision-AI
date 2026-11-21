"""
Official Test Set Evaluation and Comparative Benchmark.

Evaluates ONLY the two selected trained checkpoints on the untouched official CIFAR-10 test set:
1. CNN v2 ResNet-lite (models/cnn_v2_best.pth)
2. Swin-Tiny Transformer (models/swin_tiny_best.pth)

Enforces:
- Dedicated official test-set loader isolating CIFAR-10 test split (train=False).
- Strict zero-overwrite policy: execution aborts if any target output file exists.
- Fully dynamic comparative analysis: no hardcoded winners or assumed outcomes.
- Accurate distinction between total parameters and parameters trainable during training.
- No model retraining, weight updates, or test-set hyperparameter tuning.
"""

import os
import sys
import time
import json
import csv
import argparse
from typing import Dict, Any, List, Tuple
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.dataset import (
    filter_animal_subset,
    FilteredAnimalDataset,
    ANIMAL_CLASSES,
    ORIGINAL_TO_ANIMAL_MAP,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.cnn_model import build_cnn_v2_model
from src.swin_model import build_swin_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Exact target output files
CSV_OUTPUT = os.path.join(RESULTS_DIR, "test_metrics_comparison.csv")
JSON_OUTPUT = os.path.join(RESULTS_DIR, "test_classification_reports.json")
PLOT_OUTPUT = os.path.join(RESULTS_DIR, "test_confusion_matrices.png")
MD_OUTPUT = os.path.join(RESULTS_DIR, "test_evaluation_summary.md")

TARGET_OUTPUT_FILES = [CSV_OUTPUT, JSON_OUTPUT, PLOT_OUTPUT, MD_OUTPUT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official Test Set Evaluation for CNN v2 (ResNet-lite) and Swin-Tiny."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to CIFAR-10 data directory (default: ./data).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for test evaluation dataloader (default: 64).",
    )
    parser.add_argument(
        "--latency-runs",
        type=int,
        default=200,
        help="Number of single-image inference runs for latency benchmark (default: 200).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=50,
        help="Number of warm-up runs before measuring latency (default: 50).",
    )
    return parser.parse_args()


def enforce_zero_overwrite_policy(target_files: List[str]) -> None:
    """
    Halts execution immediately if any of the target output files already exist.
    Prevents overwriting any prior experimental data, reports, or plots.
    """
    conflicts = [f for f in target_files if os.path.exists(f)]
    if conflicts:
        print("\n" + "=" * 70, file=sys.stderr)
        print("[ERROR: ZERO-OVERWRITE POLICY ENFORCED]", file=sys.stderr)
        print("Evaluation aborted because target output file(s) already exist:", file=sys.stderr)
        for path in conflicts:
            print(f"  -> {os.path.abspath(path)}", file=sys.stderr)
        print("\nPlease archive, move, or remove these files before running evaluation.", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)


def load_official_test_dataset(
    data_dir: str = "./data",
    img_size: int = 96,
) -> Tuple[FilteredAnimalDataset, Dict[str, Any]]:
    """
    Dedicated official CIFAR-10 test set loader.
    Exclusively loads torchvision CIFAR-10 test split (train=False),
    filters only the 5 animal classes (original labels 2, 3, 4, 5, 6),
    remaps them to 0..4 (bird=0, cat=1, deer=2, dog=3, frog=4),
    and applies deterministic evaluation transforms (Resize, ToTensor, Normalize).
    """
    os.makedirs(data_dir, exist_ok=True)

    # 1. Load official test split only (train=False)
    raw_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    raw_total = len(raw_test)

    # 2. Filter animal classes
    filtered_images, filtered_targets = filter_animal_subset(raw_test)
    filtered_total = len(filtered_images)

    # Verify runtime invariants
    target_counts = Counter(filtered_targets)
    assert filtered_total == 5000, f"Expected 5,000 test images, got {filtered_total}"
    for idx, name in enumerate(ANIMAL_CLASSES):
        assert target_counts[idx] == 1000, f"Expected 1,000 images for class {name} ({idx}), got {target_counts[idx]}"

    # 3. Deterministic evaluation transforms (no data augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_dataset = FilteredAnimalDataset(
        filtered_images,
        filtered_targets,
        transform=eval_transform,
    )

    metadata = {
        "raw_cifar_test_total": raw_total,
        "filtered_animal_test_total": filtered_total,
        "per_class_counts": {ANIMAL_CLASSES[i]: target_counts[i] for i in range(5)},
        "img_size": img_size,
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
    }

    return test_dataset, metadata


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_type: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Loads checkpoint state_dict into the appropriate architecture with robust error handling.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            "Please ensure the model training completed and the checkpoint file exists in models/."
        )

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
    except Exception as err:
        raise RuntimeError(
            f"Failed to load checkpoint file at {checkpoint_path}: {err}"
        ) from err

    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {checkpoint_path} is missing 'model_state_dict'. Available keys: {list(ckpt.keys())}"
        )

    # Verify saved class names
    saved_classes = ckpt.get("class_names")
    if saved_classes and saved_classes != ANIMAL_CLASSES:
        raise ValueError(
            f"Checkpoint class mismatch: saved {saved_classes} vs expected {ANIMAL_CLASSES}"
        )

    if model_type == "cnn_v2":
        model = build_cnn_v2_model(num_classes=5, dropout_rate=0.15)
    elif model_type == "swin":
        # Initialize architecture offline without re-downloading weights; full state_dict will be loaded
        model = build_swin_model(num_classes=5, pretrained=False)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'. Expected 'cnn_v2' or 'swin'.")

    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception as err:
        raise RuntimeError(
            f"Error loading state_dict into {model_type} architecture from {checkpoint_path}: {err}"
        ) from err

    model.to(device)
    model.eval()

    return model, ckpt


def measure_cpu_latency(
    model: nn.Module,
    img_size: int,
    device: torch.device,
    warmup_runs: int = 50,
    test_runs: int = 200,
) -> Dict[str, float]:
    """
    Measures CPU deployment inference latency on single images (batch_size=1)
    with warm-up iterations to eliminate cold-start JIT and cache effects.
    """
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    # 1. Warm-up passes
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # 2. Measured passes
    latencies = []
    with torch.no_grad():
        for _ in range(test_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # convert to milliseconds

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    median_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "mean_ms": round(mean_lat, 2),
        "median_ms": round(median_lat, 2),
        "p95_ms": round(p95_lat, 2),
        "throughput_fps": round(throughput, 1),
    }


def evaluate_model_on_test(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Performs a single evaluation pass across the entire official test set without gradient calculation.
    Returns:
        (all_predictions, all_ground_truth_labels, total_eval_time_seconds)
    """
    all_preds = []
    all_targets = []
    t0 = time.time()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    elapsed = time.time() - t0
    return np.array(all_preds), np.array(all_targets), elapsed


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculates classification metrics using explicit labels [0, 1, 2, 3, 4] and zero_division=0.
    """
    target_labels = [0, 1, 2, 3, 4]

    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_p = precision_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0) * 100.0
    macro_r = recall_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0) * 100.0
    macro_f1 = f1_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0) * 100.0

    weighted_p = precision_score(y_true, y_pred, labels=target_labels, average="weighted", zero_division=0) * 100.0
    weighted_r = recall_score(y_true, y_pred, labels=target_labels, average="weighted", zero_division=0) * 100.0
    weighted_f1 = f1_score(y_true, y_pred, labels=target_labels, average="weighted", zero_division=0) * 100.0

    cm = confusion_matrix(y_true, y_pred, labels=target_labels)
    cr = classification_report(
        y_true,
        y_pred,
        labels=target_labels,
        target_names=ANIMAL_CLASSES,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": round(acc, 2),
        "macro_precision": round(macro_p, 2),
        "macro_recall": round(macro_r, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_precision": round(weighted_p, 2),
        "weighted_recall": round(weighted_r, 2),
        "weighted_f1": round(weighted_f1, 2),
        "confusion_matrix": cm,
        "classification_report": cr,
    }


def save_confusion_matrices_plot(
    results_map: Dict[str, Dict[str, Any]],
    save_path: str,
):
    """
    Plots exactly two side-by-side confusion matrices (CNN v2 vs Swin-Tiny)
    with cell count annotations and identical class axes.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = list(results_map.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, name in zip(axes, models):
        cm = results_map[name]["metrics"]["confusion_matrix"]
        acc = results_map[name]["metrics"]["accuracy"]

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name}\nOfficial Test Accuracy: {acc:.2f}%", fontsize=11, fontweight="bold", pad=10)

        tick_marks = np.arange(len(ANIMAL_CLASSES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(ANIMAL_CLASSES, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(ANIMAL_CLASSES, fontsize=10)

        # Print cell counts
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=9,
                )

        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [Saved Plot] Confusion matrices -> {os.path.abspath(save_path)}", flush=True)


def save_markdown_summary(
    results_map: Dict[str, Dict[str, Any]],
    dataset_meta: Dict[str, Any],
    save_path: str,
):
    """
    Generates a scientific markdown summary where all metrics, comparisons,
    and observations are synthesized dynamically from the actual measured data.
    No predetermined winner or hardcoded outcome is assumed.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = list(results_map.keys())
    m1_name, m2_name = models[0], models[1]
    m1_data, m2_data = results_map[m1_name], results_map[m2_name]

    acc1 = m1_data["metrics"]["accuracy"]
    acc2 = m2_data["metrics"]["accuracy"]

    lat1 = m1_data["latency"]["mean_ms"]
    lat2 = m2_data["latency"]["mean_ms"]

    with open(save_path, mode="w", encoding="utf-8") as f:
        f.write("# Official CIFAR-10 Test Set Benchmark Results\n\n")
        f.write("## 1. Evaluation Setup & Dataset Details\n\n")
        f.write("- **Dataset**: Official torchvision CIFAR-10 Test Split (`train=False`, untouched during training)\n")
        f.write(f"- **Classes Evaluated**: `{', '.join(ANIMAL_CLASSES)}` (5 classes)\n")
        f.write(f"- **Total Test Samples**: Exactly {dataset_meta['filtered_animal_test_total']:,} images (1,000 images per class, perfectly balanced)\n")
        f.write("- **Original-to-Target Label Mapping**:\n")
        for orig, target in sorted(ORIGINAL_TO_ANIMAL_MAP.items()):
            f.write(f"  - CIFAR-10 class {orig} (`{ANIMAL_CLASSES[target]}`) -> Target {target}\n")
        f.write(f"- **Evaluation Preprocessing**: Resize to {dataset_meta['img_size']}×{dataset_meta['img_size']}, `ToTensor()`, Normalize with ImageNet mean `{dataset_meta['normalization_mean']}` and std `{dataset_meta['normalization_std']}`\n")
        f.write("- **Compute Device**: CPU (Single machine benchmark)\n\n")

        f.write("## 2. Overall Performance Comparison\n\n")
        f.write("| Model Architecture | Pretraining / Training Setup | Total Model Params | Trainable Params (Training) | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Mean Latency (ms) | Throughput (FPS) | Batch Eval Time (s) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for name, data in results_map.items():
            m = data["metrics"]
            lat = data["latency"]
            p_tot = data["params_total"]
            p_tr = data["params_trainable_training"]
            setup = data["training_setup"]
            f.write(
                f"| {name} | {setup} | {p_tot:,} | {p_tr:,} | {m['accuracy']:.2f}% | "
                f"{m['macro_precision']:.2f}% | {m['macro_recall']:.2f}% | {m['macro_f1']:.2f}% | "
                f"{m['weighted_f1']:.2f}% | {lat['mean_ms']:.2f} ms | {lat['throughput_fps']:.1f} fps | "
                f"{data['eval_duration_seconds']:.2f}s |\n"
            )

        f.write("\n## 3. Per-Class Performance Breakdown\n\n")
        f.write(f"| Animal Class | Support | {m1_name} Precision | {m1_name} Recall | {m1_name} F1 | {m2_name} Precision | {m2_name} Recall | {m2_name} F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        cr1 = m1_data["metrics"]["classification_report"]
        cr2 = m2_data["metrics"]["classification_report"]

        for cls_name in ANIMAL_CLASSES:
            supp = int(cr1.get(cls_name, {}).get("support", 1000))
            p_1 = cr1.get(cls_name, {}).get("precision", 0.0) * 100
            r_1 = cr1.get(cls_name, {}).get("recall", 0.0) * 100
            f1_1 = cr1.get(cls_name, {}).get("f1-score", 0.0) * 100

            p_2 = cr2.get(cls_name, {}).get("precision", 0.0) * 100
            r_2 = cr2.get(cls_name, {}).get("recall", 0.0) * 100
            f1_2 = cr2.get(cls_name, {}).get("f1-score", 0.0) * 100

            # Dynamic emphasis strictly reflecting measured metric
            f1_1_str = f"**{f1_1:.1f}%**" if f1_1 > f1_2 else f"{f1_1:.1f}%"
            f1_2_str = f"**{f1_2:.1f}%**" if f1_2 > f1_1 else f"{f1_2:.1f}%"

            f.write(
                f"| {cls_name.capitalize()} | {supp:,} | "
                f"{p_1:.1f}% | {r_1:.1f}% | {f1_1_str} | "
                f"{p_2:.1f}% | {r_2:.1f}% | {f1_2_str} |\n"
            )

        f.write("\n## 4. CPU Deployment Latency Benchmark\n\n")
        f.write("- **Methodology**: 50 warm-up runs followed by 200 timed single-image forward passes (`[1, 3, 96, 96]`) using `time.perf_counter()` on CPU.\n")
        f.write("- **Model Loading**: Excluded from forward-pass latency measurements.\n\n")
        f.write(f"| Metric | {m1_name} | {m2_name} |\n")
        f.write("| :--- | :---: | :---: |\n")
        for key, label in [
            ("mean_ms", "Mean Latency"),
            ("median_ms", "Median Latency"),
            ("p95_ms", "P95 Latency"),
            ("throughput_fps", "Single-Image Throughput"),
        ]:
            val1 = m1_data["latency"].get(key, 0.0)
            val2 = m2_data["latency"].get(key, 0.0)
            unit = " ms" if "ms" in key else " FPS"
            f.write(f"| {label} | {val1}{unit} | {val2}{unit} |\n")

        # 5. Dynamically computed findings
        f.write("\n## 5. Measured Observations & Research Disclosures\n\n")

        # Accuracy comparison
        if acc1 > acc2:
            f.write(f"1. **Accuracy Measurement**:\n")
            f.write(f"   - {m1_name} achieved higher test accuracy ({acc1:.2f}%) than {m2_name} ({acc2:.2f}%), with an absolute difference of {acc1 - acc2:.2f}%.\n")
        elif acc2 > acc1:
            f.write(f"1. **Accuracy Measurement**:\n")
            f.write(f"   - {m2_name} achieved higher test accuracy ({acc2:.2f}%) than {m1_name} ({acc1:.2f}%), with an absolute difference of {acc2 - acc1:.2f}%.\n")
        else:
            f.write(f"1. **Accuracy Measurement**:\n")
            f.write(f"   - Both models achieved identical test accuracy ({acc1:.2f}%).\n")

        # Latency comparison
        if lat1 < lat2:
            ratio = lat2 / lat1 if lat1 > 0 else 0.0
            f.write(f"2. **Inference Latency & Throughput**:\n")
            f.write(f"   - {m1_name} demonstrated lower CPU inference latency ({lat1:.2f} ms vs. {lat2:.2f} ms), operating approximately {ratio:.1f}x faster on single-image inference.\n")
        elif lat2 < lat1:
            ratio = lat1 / lat2 if lat2 > 0 else 0.0
            f.write(f"2. **Inference Latency & Throughput**:\n")
            f.write(f"   - {m2_name} demonstrated lower CPU inference latency ({lat2:.2f} ms vs. {lat1:.2f} ms), operating approximately {ratio:.1f}x faster on single-image inference.\n")
        else:
            f.write(f"2. **Inference Latency & Throughput**:\n")
            f.write(f"   - Both models demonstrated identical mean CPU latency ({lat1:.2f} ms).\n")

        # Training regime disclosure
        f.write("3. **Training Regime & Parameter Setup**:\n")
        f.write(f"   - **{m1_name}**: {m1_data['training_setup']} (Total params: {m1_data['params_total']:,}; Trainable during training: {m1_data['params_trainable_training']:,}; Frozen during training: {m1_data['params_frozen_training']:,}).\n")
        f.write(f"   - **{m2_name}**: {m2_data['training_setup']} (Total params: {m2_data['params_total']:,}; Trainable during training: {m2_data['params_trainable_training']:,}; Frozen during training: {m2_data['params_frozen_training']:,}).\n")
        f.write("   - **Scientific Scope**: Because pretraining status (scratch initialization vs. ImageNet-1K pretrained backbone) and trainable parameter counts differ substantially between the two setups, observed performance differences reflect the joint impact of architecture and pretraining, rather than architecture alone.\n")

        # Evaluation integrity
        f.write("4. **Evaluation Integrity**:\n")
        f.write("   - Conducted on the official torchvision CIFAR-10 test set without prior exposure, post-hoc threshold selection, or hyperparameter tuning.\n")

    print(f"  [Saved Summary] Markdown summary -> {os.path.abspath(save_path)}", flush=True)


def main():
    args = parse_args()
    device = torch.device("cpu")

    # Enforce Zero-Overwrite Policy immediately before any work begins
    enforce_zero_overwrite_policy(TARGET_OUTPUT_FILES)

    print("=" * 65, flush=True)
    print("Official Test Set Evaluation: CNN v2 vs Swin-Tiny", flush=True)
    print("=" * 65, flush=True)
    print(f"Compute Device:      {device}", flush=True)
    print(f"Test Batch Size:     {args.batch_size}", flush=True)
    print(f"Latency Benchmark:   {args.warmup_runs} warm-up passes, {args.latency_runs} measured single-image runs", flush=True)

    # 1. Load official test dataset via dedicated loader
    print("\n[1/4] Loading Official Untouched CIFAR-10 Test Set...", flush=True)
    t_load_start = time.time()
    test_dataset, dataset_meta = load_official_test_dataset(
        data_dir=args.data_dir,
        img_size=96,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    print(f"  Official Test Samples: {len(test_dataset):,} images (exactly 1,000 per class)", flush=True)
    print(f"  Test Batches:          {len(test_loader)} batches (batch_size={args.batch_size})", flush=True)
    print(f"  Dataset Prep Time:     {time.time()-t_load_start:.2f}s", flush=True)

    # 2. Define model evaluations (ONLY CNN v2 and Swin-Tiny)
    # Trainable parameter counts reflect actual training configuration
    model_configs = [
        {
            "name": "CNN v2 (ResNet-lite)",
            "type": "cnn_v2",
            "checkpoint": os.path.join(MODELS_DIR, "cnn_v2_best.pth"),
            "training_setup": "Trained from scratch (5k samples, 10 epochs)",
            "params_trainable_training": 1259941,
            "params_frozen_training": 0,
        },
        {
            "name": "Swin-Tiny Transformer",
            "type": "swin",
            "checkpoint": os.path.join(MODELS_DIR, "swin_tiny_best.pth"),
            "training_setup": "Frozen ImageNet-1K pretrained backbone (5k samples, 5 epochs)",
            "params_trainable_training": 3845,
            "params_frozen_training": 27519354,
        },
    ]

    results_map = {}
    overall_eval_start = time.time()

    # 3. Evaluate each model
    print("\n[2/4] Running Single-Pass Evaluation & Latency Benchmarks...", flush=True)

    for cfg in model_configs:
        name = cfg["name"]
        ckpt_path = cfg["checkpoint"]
        m_type = cfg["type"]
        setup = cfg["training_setup"]

        print(f"\n---> Evaluating: {name} ({os.path.basename(ckpt_path)}) ---", flush=True)

        model, ckpt_meta = load_model_from_checkpoint(ckpt_path, m_type, device)
        tot_params = sum(p.numel() for p in model.parameters())

        # Verify parameter count consistency
        if m_type == "cnn_v2":
            assert tot_params == cfg["params_trainable_training"], f"CNN v2 parameter mismatch: {tot_params} vs {cfg['params_trainable_training']}"
        elif m_type == "swin":
            head_params = sum(p.numel() for p in model.head.parameters())
            assert head_params == cfg["params_trainable_training"], f"Swin head parameter mismatch: {head_params} vs {cfg['params_trainable_training']}"

        # Full test set forward pass
        print(f"  Running inference across {len(test_dataset):,} test images...", end="", flush=True)
        preds, targets, eval_duration = evaluate_model_on_test(model, test_loader, device)
        print(f" Done ({eval_duration:.1f}s)", flush=True)

        # Compute metrics
        metrics = compute_metrics(targets, preds)
        print(f"  Test Accuracy:     {metrics['accuracy']:.2f}%", flush=True)
        print(f"  Macro F1-Score:    {metrics['macro_f1']:.2f}%", flush=True)
        print(f"  Weighted F1-Score: {metrics['weighted_f1']:.2f}%", flush=True)

        # Measure latency
        print(f"  Measuring CPU latency ({args.warmup_runs} warm-up, {args.latency_runs} runs)...", end="", flush=True)
        latency = measure_cpu_latency(
            model=model,
            img_size=96,
            device=device,
            warmup_runs=args.warmup_runs,
            test_runs=args.latency_runs,
        )
        print(f" Done! Mean Latency: {latency['mean_ms']} ms ({latency['throughput_fps']} FPS)", flush=True)

        results_map[name] = {
            "model_type": m_type,
            "checkpoint": ckpt_path,
            "training_setup": setup,
            "params_total": tot_params,
            "params_trainable_training": cfg["params_trainable_training"],
            "params_frozen_training": cfg["params_frozen_training"],
            "eval_duration_seconds": round(eval_duration, 2),
            "metrics": metrics,
            "latency": latency,
        }

    # 4. Save results (with secondary Zero-Overwrite guard)
    print("\n[3/4] Exporting Test Benchmark Artifacts to results/...", flush=True)
    enforce_zero_overwrite_policy(TARGET_OUTPUT_FILES)

    # A. Save CSV comparison
    csv_rows = []
    for name, data in results_map.items():
        m = data["metrics"]
        lat = data["latency"]
        csv_rows.append({
            "model": name,
            "training_setup": data["training_setup"],
            "total_params": data["params_total"],
            "trainable_params_training": data["params_trainable_training"],
            "frozen_params_training": data["params_frozen_training"],
            "test_accuracy": m["accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_precision": m["weighted_precision"],
            "weighted_recall": m["weighted_recall"],
            "weighted_f1": m["weighted_f1"],
            "mean_latency_ms": lat["mean_ms"],
            "median_latency_ms": lat["median_ms"],
            "p95_latency_ms": lat["p95_ms"],
            "throughput_fps": lat["throughput_fps"],
            "eval_duration_seconds": data["eval_duration_seconds"],
        })

    with open(CSV_OUTPUT, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = list(csv_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  [Saved CSV] Metrics comparison  -> {os.path.abspath(CSV_OUTPUT)}", flush=True)

    # B. Save JSON reports
    json_export = {
        "dataset_metadata": dataset_meta,
        "models": {},
    }
    for name, data in results_map.items():
        json_export["models"][name] = {
            "training_setup": data["training_setup"],
            "total_parameters": data["params_total"],
            "trainable_parameters_during_training": data["params_trainable_training"],
            "frozen_parameters_during_training": data["params_frozen_training"],
            "batch_eval_duration_seconds": data["eval_duration_seconds"],
            "summary_metrics": {
                "test_accuracy": data["metrics"]["accuracy"],
                "macro_precision": data["metrics"]["macro_precision"],
                "macro_recall": data["metrics"]["macro_recall"],
                "macro_f1": data["metrics"]["macro_f1"],
                "weighted_precision": data["metrics"]["weighted_precision"],
                "weighted_recall": data["metrics"]["weighted_recall"],
                "weighted_f1": data["metrics"]["weighted_f1"],
                "latency_mean_ms": data["latency"]["mean_ms"],
                "latency_median_ms": data["latency"]["median_ms"],
                "latency_p95_ms": data["latency"]["p95_ms"],
                "throughput_fps": data["latency"]["throughput_fps"],
            },
            "per_class_report": data["metrics"]["classification_report"],
            "confusion_matrix": data["metrics"]["confusion_matrix"].tolist(),
        }
    with open(JSON_OUTPUT, mode="w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2)
    print(f"  [Saved JSON] Classification reports -> {os.path.abspath(JSON_OUTPUT)}", flush=True)

    # C. Save Confusion Matrices Plot
    save_confusion_matrices_plot(results_map, PLOT_OUTPUT)

    # D. Save Markdown Summary
    save_markdown_summary(results_map, dataset_meta, MD_OUTPUT)

    total_time = time.time() - overall_eval_start
    mins = int(total_time // 60)
    secs = int(total_time % 60)

    print("\n" + "=" * 65, flush=True)
    print("TEST EVALUATION COMPLETE", flush=True)
    print(f"Total Execution Time: {mins}m {secs}s ({total_time:.1f}s)", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
