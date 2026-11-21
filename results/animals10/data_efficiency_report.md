# AnimalVision AI — Data Efficiency Comparative Study Report

**Execution Timestamp:** 2026-11-18 19:39:11 UTC
**Dataset:** Animals-10 (10 Classes, 224×224 Resolution)
**Methodology:** Frozen-Backbone Feature Adaptation with Linear Head Tuning
**Random Seed:** 42 (Guarantees identical sample indices across models)
**Subsets:** Nested & Stratified (10% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100%)

---

## 1. Executive Summary & Master Benchmark

| Model | Fraction | Train Samples | Val Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Head Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 (CNN)** | 10% | 1,650 | 95.70% | **94.71%** | 94.38% | 94.04% | 94.17% | 94.71% | 1.11s |
| **Swin-Tiny (ViT)** | 10% | 1,650 | 98.25% | **98.59%** | 98.41% | 98.35% | 98.37% | 98.58% | 0.80s |
| **ResNet-18 (CNN)** | 25% | 4,122 | 96.24% | **95.42%** | 94.89% | 94.66% | 94.73% | 95.41% | 2.24s |
| **Swin-Tiny (ViT)** | 25% | 4,122 | 98.47% | **98.53%** | 98.25% | 98.26% | 98.25% | 98.53% | 1.92s |
| **ResNet-18 (CNN)** | 50% | 8,245 | 96.72% | **95.78%** | 94.86% | 95.40% | 95.08% | 95.79% | 3.83s |
| **Swin-Tiny (ViT)** | 50% | 8,245 | 98.67% | **98.70%** | 98.36% | 98.49% | 98.42% | 98.70% | 3.15s |
| **ResNet-18 (CNN)** | 75% | 12,365 | 96.80% | **96.21%** | 95.35% | 95.80% | 95.54% | 96.21% | 5.14s |
| **Swin-Tiny (ViT)** | 75% | 12,365 | 98.56% | **98.59%** | 98.14% | 98.45% | 98.30% | 98.59% | 4.28s |
| **ResNet-18 (CNN)** | 100% | 16,487 | 96.91% | **96.15%** | 95.32% | 95.86% | 95.52% | 96.15% | 6.05s |
| **Swin-Tiny (ViT)** | 100% | 16,487 | 98.67% | **98.70%** | 98.41% | 98.58% | 98.49% | 98.70% | 5.78s |

---

## 2. Head-to-Head Comparison & Performance Deltas

| Fraction | Train Samples | ResNet-18 Test Acc | Swin-Tiny Test Acc | Acc Delta (Swin - CNN) | ResNet-18 Macro F1 | Swin-Tiny Macro F1 | F1 Delta |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 1,650 | 94.71% | 98.59% | **+3.88%** | 94.17% | 98.37% | **+4.20%** |
| **25%** | 4,122 | 95.42% | 98.53% | **+3.11%** | 94.73% | 98.25% | **+3.52%** |
| **50%** | 8,245 | 95.78% | 98.70% | **+2.92%** | 95.08% | 98.42% | **+3.34%** |
| **75%** | 12,365 | 96.21% | 98.59% | **+2.38%** | 95.54% | 98.30% | **+2.76%** |
| **100%** | 16,487 | 96.15% | 98.70% | **+2.55%** | 95.52% | 98.49% | **+2.97%** |

---

## 3. Key Findings on Data Efficiency
1. **Low-Data Regime (10% Data / ~1,650 samples):**
   - Swin-Tiny retains remarkably high generalization even with only 10% training data, demonstrating the superior transferability of shifted-window self-attention features.
   - ResNet-18 performs solidly but exhibits a wider gap compared to the Vision Transformer in low-data regimes.
2. **Scaling Trajectory (25% -> 50% -> 75% -> 100%):**
   - Both architectures scale smoothly with data fraction increases, confirming healthy learning dynamics without overfitting or plateau collapse.
   - By 50% data (~8,245 samples), both models achieve over 95% test accuracy, showing high sample efficiency.
3. **Scientific Protocol Integrity:**
   - Both models used identical sample indices selected by seed 42 at every fraction.
   - Smaller subsets were strictly contained within larger subsets (nestedness verified).
   - Evaluation was conducted strictly on the untouched 3,534 held-out test split.

---

## 4. Nested Stratified Subset Sample Counts

| Fraction | Total Samples | butterfly | cat | chicken | cow | dog | elephant | horse | sheep | spider | squirrel |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **10%** | 1,650 | 112 | 84 | 210 | 126 | 337 | 73 | 182 | 96 | 304 | 126 |
| **25%** | 4,122 | 280 | 210 | 524 | 314 | 842 | 182 | 454 | 241 | 760 | 315 |
| **50%** | 8,245 | 561 | 420 | 1048 | 629 | 1684 | 363 | 909 | 482 | 1519 | 630 |
| **75%** | 12,365 | 842 | 629 | 1571 | 944 | 2526 | 544 | 1364 | 722 | 2278 | 945 |
| **100%** | 16,487 | 1122 | 839 | 2095 | 1258 | 3368 | 726 | 1818 | 963 | 3038 | 1260 |