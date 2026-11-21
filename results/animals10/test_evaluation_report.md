# Animals-10 Held-Out Test Evaluation & Comparative Benchmark Report

**Evaluation Timestamp:** 2026-11-15 09:33:03 UTC  
**Hardware Platform:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel (Windows-10-10.0.26300-SP0)  
**PyTorch Version:** 2.14.1+cpu (CPU Threads: 6)  
**Dataset:** Rapidata/Animals-10 (Held-Out Test Split, 3,534 images across 10 classes)  
**Input Preprocessing:** Deterministic `Resize(256) + CenterCrop(224)` with ImageNet Normalization  

---

## 1. Executive Summary & Master Comparative Benchmark

| Evaluation Metric | ResNet-18 (Residual CNN) | Swin-Tiny (Shifted-Window ViT) | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **96.49%** | **98.64%** | **+2.15%** (Swin leads) |
| **Macro Precision** | 95.72% | 98.29% | +2.57% |
| **Macro Recall** | 96.17% | 98.47% | +2.30% |
| **Macro F1-Score** | **95.92%** | **98.38%** | **+2.46%** |
| **Weighted F1-Score** | 96.49% | 98.64% | +2.15% |
| **Total Parameters** | 11,181,642 | 27,527,044 | Swin has 2.46x more parameters |
| **Trainable Head Params** | 5,130 (0.046%) | 7,690 (0.028%) | Both trained feature heads only |
| **CPU Latency (Mean)** | **49.90 ms** | **178.42 ms** | **ResNet-18 is 3.58x faster** |
| **CPU Latency (Median)** | 45.37 ms | 170.91 ms | ResNet-18 is 3.77x faster |
| **CPU Latency (P95)** | 72.64 ms | 235.09 ms | ResNet-18 has tighter tail latency |
| **Throughput (BS=1)** | **20.0 FPS** | **5.6 FPS** | ResNet-18 processes +14.4 FPS |
| **Batch Eval Time (3.5k imgs)** | 193.1s | 731.2s | ResNet batch eval was 3.79x faster |

## 2. Per-Class Precision, Recall, and F1-Score Breakdown

| Class Name | Test Support | ResNet-18 Precision | ResNet-18 Recall | ResNet-18 F1 | Swin-Tiny Precision | Swin-Tiny Recall | Swin-Tiny F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Butterfly** | 241 | 96.36% | 98.76% | **97.54%** | 99.17% | 98.76% | **98.96%** |
| **Cat** | 180 | 92.67% | 98.33% | **95.42%** | 97.28% | 99.44% | **98.35%** |
| **Chicken** | 449 | 99.08% | 96.44% | **97.74%** | 99.78% | 99.33% | **99.55%** |
| **Cow** | 270 | 92.66% | 88.89% | **90.74%** | 95.27% | 97.04% | **96.15%** |
| **Dog** | 722 | 97.34% | 96.40% | **96.87%** | 99.16% | 98.34% | **98.75%** |
| **Elephant** | 155 | 98.04% | 96.77% | **97.40%** | 98.72% | 99.35% | **99.04%** |
| **Horse** | 390 | 94.24% | 96.41% | **95.31%** | 98.46% | 98.21% | **98.33%** |
| **Sheep** | 206 | 90.14% | 93.20% | **91.65%** | 97.03% | 95.15% | **96.08%** |
| **Spider** | 651 | 99.23% | 99.08% | **99.15%** | 99.54% | 99.85% | **99.69%** |
| **Squirrel** | 270 | 97.41% | 97.41% | **97.41%** | 98.53% | 99.26% | **98.89%** |

## 3. Checkpoint File Integrity Verification

Both saved model checkpoints were verified bit-for-bit before and after held-out test evaluation:

- `models/resnet18_animals10_best.pth`  
  - SHA-256: `BBF2A7EA36EADE566AF178CB4181894391F50E2162457991D801BA6912B8B933`  
  - Status: **VERIFIED UNCHANGED**
- `models/swin_tiny_animals10_best.pth`  
  - SHA-256: `649E5A7794CEFD7797249B27508E39FBE46CD31412FFA4E631EA329B7DB8AB6B`  
  - Status: **VERIFIED UNCHANGED**

## 4. Architectural Analysis & Hardware Trade-Offs

1. **Representational Fidelity**: Swin-Tiny achieves higher accuracy on fine-grained, high-resolution animal patterns (e.g. fur textures, posture) due to hierarchical shifted-window self-attention.
2. **CPU Inference Latency**: ResNet-18 delivers a substantial speed advantage (49.90 ms vs 178.42 ms), making it 3.58x faster for real-time edge CPU deployments.
3. **Tail Latency**: ResNet-18 exhibits significantly lower P95 latency variance under CPU load compared to the multi-head self-attention window shifting kernels in Swin.
