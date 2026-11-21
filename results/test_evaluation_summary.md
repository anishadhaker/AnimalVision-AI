# Official CIFAR-10 Test Set Benchmark Results

## 1. Evaluation Setup & Dataset Details

- **Dataset**: Official torchvision CIFAR-10 Test Split (`train=False`, untouched during training)
- **Classes Evaluated**: `bird, cat, deer, dog, frog` (5 classes)
- **Total Test Samples**: Exactly 5,000 images (1,000 images per class, perfectly balanced)
- **Original-to-Target Label Mapping**:
  - CIFAR-10 class 2 (`bird`) -> Target 0
  - CIFAR-10 class 3 (`cat`) -> Target 1
  - CIFAR-10 class 4 (`deer`) -> Target 2
  - CIFAR-10 class 5 (`dog`) -> Target 3
  - CIFAR-10 class 6 (`frog`) -> Target 4
- **Evaluation Preprocessing**: Resize to 96×96, `ToTensor()`, Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`
- **Compute Device**: CPU (Single machine benchmark)

## 2. Overall Performance Comparison

| Model Architecture | Pretraining / Training Setup | Total Model Params | Trainable Params (Training) | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Mean Latency (ms) | Throughput (FPS) | Batch Eval Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CNN v2 (ResNet-lite) | Trained from scratch (5k samples, 10 epochs) | 1,259,941 | 1,259,941 | 55.94% | 55.83% | 55.94% | 55.44% | 55.44% | 14.08 ms | 71.0 fps | 83.23s |
| Swin-Tiny Transformer | Frozen ImageNet-1K pretrained backbone (5k samples, 5 epochs) | 27,523,199 | 3,845 | 84.04% | 84.11% | 84.04% | 84.06% | 84.06% | 47.91 ms | 20.9 fps | 181.09s |

## 3. Per-Class Performance Breakdown

| Animal Class | Support | CNN v2 (ResNet-lite) Precision | CNN v2 (ResNet-lite) Recall | CNN v2 (ResNet-lite) F1 | Swin-Tiny Transformer Precision | Swin-Tiny Transformer Recall | Swin-Tiny Transformer F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Bird | 1,000 | 58.6% | 45.1% | 51.0% | 87.1% | 83.7% | **85.4%** |
| Cat | 1,000 | 45.6% | 41.3% | 43.4% | 76.9% | 78.7% | **77.8%** |
| Deer | 1,000 | 53.2% | 67.8% | 59.6% | 84.7% | 87.3% | **86.0%** |
| Dog | 1,000 | 57.6% | 53.0% | 55.2% | 83.4% | 83.4% | **83.4%** |
| Frog | 1,000 | 64.0% | 72.5% | 68.0% | 88.5% | 87.1% | **87.8%** |

## 4. CPU Deployment Latency Benchmark

- **Methodology**: 50 warm-up runs followed by 200 timed single-image forward passes (`[1, 3, 96, 96]`) using `time.perf_counter()` on CPU.
- **Model Loading**: Excluded from forward-pass latency measurements.

| Metric | CNN v2 (ResNet-lite) | Swin-Tiny Transformer |
| :--- | :---: | :---: |
| Mean Latency | 14.08 ms | 47.91 ms |
| Median Latency | 13.39 ms | 46.9 ms |
| P95 Latency | 17.79 ms | 56.31 ms |
| Single-Image Throughput | 71.0 FPS | 20.9 FPS |

## 5. Measured Observations & Research Disclosures

1. **Accuracy Measurement**:
   - Swin-Tiny Transformer achieved higher test accuracy (84.04%) than CNN v2 (ResNet-lite) (55.94%), with an absolute difference of 28.10%.
2. **Inference Latency & Throughput**:
   - CNN v2 (ResNet-lite) demonstrated lower CPU inference latency (14.08 ms vs. 47.91 ms), operating approximately 3.4x faster on single-image inference.
3. **Training Regime & Parameter Setup**:
   - **CNN v2 (ResNet-lite)**: Trained from scratch (5k samples, 10 epochs) (Total params: 1,259,941; Trainable during training: 1,259,941; Frozen during training: 0).
   - **Swin-Tiny Transformer**: Frozen ImageNet-1K pretrained backbone (5k samples, 5 epochs) (Total params: 27,523,199; Trainable during training: 3,845; Frozen during training: 27,519,354).
   - **Scientific Scope**: Because pretraining status (scratch initialization vs. ImageNet-1K pretrained backbone) and trainable parameter counts differ substantially between the two setups, observed performance differences reflect the joint impact of architecture and pretraining, rather than architecture alone.
4. **Evaluation Integrity**:
   - Conducted on the official torchvision CIFAR-10 test set without prior exposure, post-hoc threshold selection, or hyperparameter tuning.
