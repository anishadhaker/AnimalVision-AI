import os
import sys
import time
import psutil
import torch
import torch.nn as nn
import numpy as np

WORKSPACE = r"c:\Users\Laptop\OneDrive\Desktop\Swin vs CNN"
sys.path.insert(0, WORKSPACE)

from src.animals10_dataset import (
    get_animals10_datasets,
    compute_training_class_weights,
    ANIMALS10_CLASSES,
    CLASS_TO_IDX,
)
from src.resnet_model import build_resnet18_model
from src.swin_model import build_swin_model

print("=" * 70)
print("TRAINING READINESS & CPU TIMING BENCHMARK")
print("=" * 70)

process = psutil.Process()
mem_before = process.memory_info().rss / (1024**2)
sys_avail = psutil.virtual_memory().available / (1024**3)
print(f"Initial Process RAM: {mem_before:.1f} MB | Available System RAM: {sys_avail:.2f} GB")

# 1. Check Class Weights Alignment
print("\n--- 1. VERIFYING CLASS-WEIGHT TO CLASS-INDEX ALIGNMENT ---")
train_ds, val_ds, test_ds = get_animals10_datasets(
    splits_dir=os.path.join(WORKSPACE, "data", "animals-10", "splits"),
    img_size=224,
)
weights_tensor, weights_dict = compute_training_class_weights(train_ds, method="balanced")

print(f"Dataset classes list:    {train_ds.classes}")
print(f"Dataset class_to_idx:    {train_ds.class_to_idx}")
print(f"ANIMALS10_CLASSES:       {ANIMALS10_CLASSES}")
print(f"CLASS_TO_IDX:            {CLASS_TO_IDX}")

alignment_ok = True
for idx, name in enumerate(ANIMALS10_CLASSES):
    ds_idx = train_ds.class_to_idx[name]
    if ds_idx != idx:
        print(f"MISMATCH for class {name}: ds_idx={ds_idx} vs expected={idx}")
        alignment_ok = False
    weight_val = float(weights_tensor[idx])
    dict_val = weights_dict[name]
    if abs(weight_val - dict_val) > 1e-4:
        print(f"Weight mismatch for {name}: tensor={weight_val} vs dict={dict_val}")
        alignment_ok = False

if alignment_ok:
    print("Alignment Check: 100% PERFECT! Every class index, folder name, and weight index match exactly.")

# 2. Benchmark ResNet-18 (Batch Size 8)
print("\n--- 2. BENCHMARKING RESNET-18 (BATCH SIZE 8, FROZEN BACKBONE) ---")
train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

resnet = build_resnet18_model(num_classes=10, pretrained=True, freeze_backbone=True)
resnet.train()
optimizer_res = torch.optim.AdamW(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

res_step_times = []
res_data_times = []
res_fwd_times = []
res_bwd_times = []

# Run 1 warmup step + 10 benchmark steps
step = 0
t_batch_start = time.perf_counter()
for images, targets in train_loader:
    data_time = (time.perf_counter() - t_batch_start) * 1000
    
    t_fwd_start = time.perf_counter()
    outputs = resnet(images)
    loss = criterion(outputs, targets)
    fwd_time = (time.perf_counter() - t_fwd_start) * 1000
    
    t_bwd_start = time.perf_counter()
    optimizer_res.zero_grad()
    loss.backward()
    optimizer_res.step()
    bwd_time = (time.perf_counter() - t_bwd_start) * 1000
    
    step_time = (time.perf_counter() - t_batch_start) * 1000
    step += 1
    
    if step > 1: # exclude initial warmup step
        res_data_times.append(data_time)
        res_fwd_times.append(fwd_time)
        res_bwd_times.append(bwd_time)
        res_step_times.append(step_time)
        
    t_batch_start = time.perf_counter()
    if step >= 11:
        break

res_mem = process.memory_info().rss / (1024**2)
res_avg_step_ms = np.mean(res_step_times)
res_avg_data_ms = np.mean(res_data_times)
res_avg_fwd_ms = np.mean(res_fwd_times)
res_avg_bwd_ms = np.mean(res_bwd_times)

print(f"ResNet-18 Benchmark (10 steps average):")
print(f"  Data Loading:    {res_avg_data_ms:6.1f} ms / step")
print(f"  Forward Pass:    {res_avg_fwd_ms:6.1f} ms / step")
print(f"  Backward Pass:   {res_avg_bwd_ms:6.1f} ms / step")
print(f"  Total Step Time: {res_avg_step_ms:6.1f} ms / step ({1000/res_avg_step_ms*8:.1f} images/sec)")
print(f"  Process RAM:     {res_mem:.1f} MB (Delta: +{res_mem - mem_before:.1f} MB)")

# 3. Benchmark Swin-Tiny (Batch Size 8)
print("\n--- 3. BENCHMARKING SWIN-TINY (BATCH SIZE 8, FROZEN BACKBONE) ---")
swin = build_swin_model(num_classes=10, pretrained=True, freeze_backbone=True)
swin.train()
optimizer_swin = torch.optim.AdamW(filter(lambda p: p.requires_grad, swin.parameters()), lr=1e-4)

swin_step_times = []
swin_data_times = []
swin_fwd_times = []
swin_bwd_times = []

step = 0
t_batch_start = time.perf_counter()
for images, targets in train_loader:
    data_time = (time.perf_counter() - t_batch_start) * 1000
    
    t_fwd_start = time.perf_counter()
    outputs = swin(images)
    loss = criterion(outputs, targets)
    fwd_time = (time.perf_counter() - t_fwd_start) * 1000
    
    t_bwd_start = time.perf_counter()
    optimizer_swin.zero_grad()
    loss.backward()
    optimizer_swin.step()
    bwd_time = (time.perf_counter() - t_bwd_start) * 1000
    
    step_time = (time.perf_counter() - t_batch_start) * 1000
    step += 1
    
    if step > 1: # exclude warmup
        swin_data_times.append(data_time)
        swin_fwd_times.append(fwd_time)
        swin_bwd_times.append(bwd_time)
        swin_step_times.append(step_time)
        
    t_batch_start = time.perf_counter()
    if step >= 11:
        break

swin_mem = process.memory_info().rss / (1024**2)
swin_avg_step_ms = np.mean(swin_step_times)
swin_avg_data_ms = np.mean(swin_data_times)
swin_avg_fwd_ms = np.mean(swin_fwd_times)
swin_avg_bwd_ms = np.mean(swin_bwd_times)

print(f"Swin-Tiny Benchmark (10 steps average):")
print(f"  Data Loading:    {swin_avg_data_ms:6.1f} ms / step")
print(f"  Forward Pass:    {swin_avg_fwd_ms:6.1f} ms / step")
print(f"  Backward Pass:   {swin_avg_bwd_ms:6.1f} ms / step")
print(f"  Total Step Time: {swin_avg_step_ms:6.1f} ms / step ({1000/swin_avg_step_ms*8:.1f} images/sec)")
print(f"  Process RAM:     {swin_mem:.1f} MB (Delta: +{swin_mem - res_mem:.1f} MB)")

# 4. Projections for Full Dataset (16,487 training samples)
batches_per_epoch = int(np.ceil(len(train_ds) / 8))
print(f"\n--- 4. FULL DATASET TRAINING DURATION PROJECTIONS ---")
print(f"Total training images: {len(train_ds):,} | Batches per epoch (batch_size=8): {batches_per_epoch:,}")

resnet_epoch_sec = (batches_per_epoch * res_avg_step_ms) / 1000.0
swin_epoch_sec = (batches_per_epoch * swin_avg_step_ms) / 1000.0

print(f"\nResNet-18 (batch_size=8):")
print(f"  Estimated 1 Epoch:  {resnet_epoch_sec / 60:.1f} minutes ({resnet_epoch_sec:.0f} seconds)")
print(f"  Estimated 5 Epochs: {resnet_epoch_sec * 5 / 60:.1f} minutes ({(resnet_epoch_sec * 5) / 3600:.2f} hours)")

print(f"\nSwin-Tiny (batch_size=8):")
print(f"  Estimated 1 Epoch:  {swin_epoch_sec / 60:.1f} minutes ({swin_epoch_sec:.0f} seconds)")
print(f"  Estimated 5 Epochs: {swin_epoch_sec * 5 / 60:.1f} minutes ({(swin_epoch_sec * 5) / 3600:.2f} hours)")

# 5. Also evaluate caching feature vectors / embeddings strategy!
print("\n--- 5. ALTERNATIVE ULTRA-FAST STRATEGY: PRE-EXTRACTED EMBEDDINGS ---")
print("Since backbones are frozen, passing the dataset ONCE through ResNet-18 and Swin-Tiny")
print("to cache the 512-dim and 768-dim embeddings reduces multi-epoch training from hours to SECONDS!")
emb_resnet_mb = (len(train_ds) * 512 * 4) / (1024**2)
emb_swin_mb = (len(train_ds) * 768 * 4) / (1024**2)
print(f"  ResNet embeddings size: {emb_resnet_mb:.1f} MB")
print(f"  Swin embeddings size:   {emb_swin_mb:.1f} MB")

# Save benchmark results
bench_results = {
    "system_available_ram_gb": round(sys_avail, 2),
    "batch_size": 8,
    "batches_per_epoch": batches_per_epoch,
    "resnet18": {
        "step_total_ms": round(res_avg_step_ms, 1),
        "data_loading_ms": round(res_avg_data_ms, 1),
        "forward_ms": round(res_avg_fwd_ms, 1),
        "backward_ms": round(res_avg_bwd_ms, 1),
        "throughput_img_per_sec": round((1000 / res_avg_step_ms) * 8, 2),
        "process_ram_mb": round(res_mem, 1),
        "est_1_epoch_min": round(resnet_epoch_sec / 60, 2),
        "est_5_epochs_min": round((resnet_epoch_sec * 5) / 60, 2),
    },
    "swin_tiny": {
        "step_total_ms": round(swin_avg_step_ms, 1),
        "data_loading_ms": round(swin_avg_data_ms, 1),
        "forward_ms": round(swin_avg_fwd_ms, 1),
        "backward_ms": round(swin_avg_bwd_ms, 1),
        "throughput_img_per_sec": round((1000 / swin_avg_step_ms) * 8, 2),
        "process_ram_mb": round(swin_mem, 1),
        "est_1_epoch_min": round(swin_epoch_sec / 60, 2),
        "est_5_epochs_min": round((swin_epoch_sec * 5) / 60, 2),
    }
}

bench_path = os.path.join(WORKSPACE, "results", "animals10", "cpu_training_readiness_benchmark.json")
import json
with open(bench_path, "w") as f:
    json.dump(bench_results, f, indent=2)
print(f"\nSaved benchmark results to: {bench_path}")
