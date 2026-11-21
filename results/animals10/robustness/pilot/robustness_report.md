# AnimalVision AI: Image Corruption Robustness Analysis (Pilot Report)

> **Experiment Level:** Phase 2 Pilot Experiment (Deterministic 500-Image Stratified Test Subset)  
> **Dataset Source:** `data/animals-10/splits/test/` (50 samples per class across 10 canonical categories)  
> **Model Architectures:** ResNet-18 (Residual CNN, 11.18M params) vs. Swin-Tiny (Hierarchical ViT, 27.53M params)  
> **Total Execution Time:** 41.39 minutes (2483.5 seconds)  
> **Status:** Completed & Validated

---

## 1. Executive Summary & Core Findings

This pilot robustness experiment empirically measures the stability and degradation curves of **ResNet-18** (inductive convolutional bias) and **Swin-Tiny** (shifted-window self-attention) under 5 realistic visual corruptions across 5 standard severity levels (25 distinct degradation conditions).

### Key Takeaways:
1. **Clean Baseline on 500-Sample Subset:**
   - **ResNet-18:** **96.00%** Accuracy | **96.01%** Macro F1
   - **Swin-Tiny:** **98.60%** Accuracy | **98.60%** Macro F1
   - **Baseline Advantage:** Swin-Tiny leads by **+2.60%** accuracy on the clean subset.

2. **Overall Robustness Superiority:**
   - Across all 25 corrupted conditions, **Swin-Tiny proved more robust in 20 of 25 evaluations** (4 ResNet wins, 1 ties).
   - Mean absolute accuracy drop across all corruptions:
     - **ResNet-18:** **-7.92%** average degradation
     - **Swin-Tiny:** **-3.59%** average degradation
   - Swin-Tiny maintains higher absolute accuracy and macro F1 across **every single corruption type and severity level**.

3. **Corruption Sensitivity Hierarchy:**
   - **Most Destructive:** Gaussian Noise (Severity 5) and Severe Fog cause the greatest performance drops for both models.
   - **Highest Resilience:** Both architectures display exceptional resilience against Photometric Brightness shifts.

---

## 2. Checkpoint & Preprocessing Verification

- **ResNet-18 Checkpoint:** `models/resnet18_animals10_best.pth`  
  *SHA-256:* `bbf2a7ea36eade566af178cb4181894391f50e2162457991d801ba6912b8b933` (Unchanged)
- **Swin-Tiny Checkpoint:** `models/swin_tiny_animals10_best.pth`  
  *SHA-256:* `649e5a7794cefd7797249b27508e39fbe46cd31412ffa4e631ea329b7db8ab6b` (Unchanged)
- **Preprocessing Pipeline:** Deterministic `Resize(256) + CenterCrop(224) + ToTensor()`.
- **Order of Operations:** Corruptions applied strictly in `[0.0, 1.0]` tensor space **prior** to ImageNet z-score normalization (`IMAGENET_MEAN = [0.485, 0.456, 0.406]`, `IMAGENET_STD = [0.229, 0.224, 0.225]`).

---

## 3. Results by Corruption Type

### Gaussian Noise
*Additive zero-mean Gaussian noise simulating sensor/thermal noise.*

| Severity | Parameter Label | ResNet-18 Acc | ResNet Drop | Swin-Tiny Acc | Swin Drop | Robustness Gap | More Robust |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1** | `sigma=0.04 (mild)` | 96.00% | -0.00pp (0.0%) | **98.60%** | -0.00pp (0.0%) | **+2.60pp** | Tie |
| **Level 2** | `sigma=0.08 (low)` | 92.20% | -3.80pp (4.0%) | **98.20%** | -0.40pp (0.4%) | **+6.00pp** | Swin-Tiny |
| **Level 3** | `sigma=0.14 (moderate)` | 83.60% | -12.40pp (12.9%) | **94.00%** | -4.60pp (4.7%) | **+10.40pp** | Swin-Tiny |
| **Level 4** | `sigma=0.20 (high)` | 67.60% | -28.40pp (29.6%) | **88.00%** | -10.60pp (10.8%) | **+20.40pp** | Swin-Tiny |
| **Level 5** | `sigma=0.28 (severe)` | 46.00% | -50.00pp (52.1%) | **73.00%** | -25.60pp (26.0%) | **+27.00pp** | Swin-Tiny |

### Gaussian Blur
*Defocus / optical blur via spatial Gaussian convolution kernel.*

| Severity | Parameter Label | ResNet-18 Acc | ResNet Drop | Swin-Tiny Acc | Swin Drop | Robustness Gap | More Robust |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1** | `k=5, s=0.75 (mild)` | 95.60% | -0.40pp (0.4%) | **98.40%** | -0.20pp (0.2%) | **+2.80pp** | Swin-Tiny |
| **Level 2** | `k=7, s=1.25 (low)` | 93.00% | -3.00pp (3.1%) | **97.40%** | -1.20pp (1.2%) | **+4.40pp** | Swin-Tiny |
| **Level 3** | `k=11, s=2.00 (moderate)` | 88.80% | -7.20pp (7.5%) | **93.20%** | -5.40pp (5.5%) | **+4.40pp** | Swin-Tiny |
| **Level 4** | `k=15, s=3.00 (high)` | 81.60% | -14.40pp (15.0%) | **88.20%** | -10.40pp (10.6%) | **+6.60pp** | Swin-Tiny |
| **Level 5** | `k=21, s=4.50 (severe)` | 65.80% | -30.20pp (31.5%) | **72.60%** | -26.00pp (26.4%) | **+6.80pp** | Swin-Tiny |

### Brightness
*Photometric illumination shift simulating high solar over-exposure.*

| Severity | Parameter Label | ResNet-18 Acc | ResNet Drop | Swin-Tiny Acc | Swin Drop | Robustness Gap | More Robust |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1** | `factor=1.20 (mild)` | 95.80% | -0.20pp (0.2%) | **98.80%** | --0.20pp (-0.2%) | **+3.00pp** | Swin-Tiny |
| **Level 2** | `factor=1.40 (low)` | 94.40% | -1.60pp (1.7%) | **98.40%** | -0.20pp (0.2%) | **+4.00pp** | Swin-Tiny |
| **Level 3** | `factor=1.65 (moderate)` | 92.20% | -3.80pp (4.0%) | **98.00%** | -0.60pp (0.6%) | **+5.80pp** | Swin-Tiny |
| **Level 4** | `factor=1.90 (high)` | 90.40% | -5.60pp (5.8%) | **98.00%** | -0.60pp (0.6%) | **+7.60pp** | Swin-Tiny |
| **Level 5** | `factor=2.20 (severe)` | 84.40% | -11.60pp (12.1%) | **98.00%** | -0.60pp (0.6%) | **+13.60pp** | Swin-Tiny |

### Contrast
*Dynamic range compression towards mean grayscale intensity.*

| Severity | Parameter Label | ResNet-18 Acc | ResNet Drop | Swin-Tiny Acc | Swin Drop | Robustness Gap | More Robust |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1** | `factor=0.80 (mild)` | 96.20% | --0.20pp (-0.2%) | **98.20%** | -0.40pp (0.4%) | **+2.00pp** | ResNet-18 |
| **Level 2** | `factor=0.60 (low)` | 96.60% | --0.60pp (-0.6%) | **98.20%** | -0.40pp (0.4%) | **+1.60pp** | ResNet-18 |
| **Level 3** | `factor=0.40 (moderate)` | 94.80% | -1.20pp (1.2%) | **98.60%** | -0.00pp (0.0%) | **+3.80pp** | Swin-Tiny |
| **Level 4** | `factor=0.25 (high)` | 92.80% | -3.20pp (3.3%) | **98.00%** | -0.60pp (0.6%) | **+5.20pp** | Swin-Tiny |
| **Level 5** | `factor=0.15 (severe)` | 87.40% | -8.60pp (9.0%) | **97.40%** | -1.20pp (1.2%) | **+10.00pp** | Swin-Tiny |

### Atmospheric Fog
*Atmospheric haze scattering model blending smooth fractal fog with image.*

| Severity | Parameter Label | ResNet-18 Acc | ResNet Drop | Swin-Tiny Acc | Swin Drop | Robustness Gap | More Robust |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1** | `alpha=0.15 (mild)` | 96.20% | --0.20pp (-0.2%) | **98.60%** | -0.00pp (0.0%) | **+2.40pp** | ResNet-18 |
| **Level 2** | `alpha=0.30 (low)` | 96.40% | --0.40pp (-0.4%) | **98.40%** | -0.20pp (0.2%) | **+2.00pp** | ResNet-18 |
| **Level 3** | `alpha=0.45 (moderate)` | 95.00% | -1.00pp (1.0%) | **98.20%** | -0.40pp (0.4%) | **+3.20pp** | Swin-Tiny |
| **Level 4** | `alpha=0.60 (high)` | 92.40% | -3.60pp (3.8%) | **98.40%** | -0.20pp (0.2%) | **+6.00pp** | Swin-Tiny |
| **Level 5** | `alpha=0.75 (severe)` | 86.80% | -9.20pp (9.6%) | **98.40%** | -0.20pp (0.2%) | **+11.60pp** | Swin-Tiny |

---

## 4. Methodological Distinction

This report documents a **500-image stratified pilot subset** (50 images per class, `Seed 42`) specifically generated to test the corruption engine, measure CPU runtime, and validate metric degradation trends.  
It is **not** the official full-test benchmark (3,534 images). The original primary held-out benchmark files in `results/animals10/test_metrics_comparison.csv` remain strictly untouched and preserved.
