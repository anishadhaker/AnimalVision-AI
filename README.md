# AnimalVision AI — A Comparative Study of CNN and Swin Transformer for Multi-Class Animal Image Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61dafb.svg)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38bdf8.svg)](https://tailwindcss.com/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646cff.svg)](https://vitejs.dev/)
[![Device](https://img.shields.io/badge/Compute-CPU--Optimized-blue.svg)]()

**AnimalVision AI** is an interactive computer vision application and empirical benchmark comparing two deep learning paradigms for animal image classification:
1. **CNN v2 (AnimalResNet)**: A 4-stage residual convolutional network trained entirely from scratch.
2. **Swin-Tiny Transformer**: A hierarchical vision transformer with shifted-window self-attention adapting an ImageNet-1K pretrained backbone.

The project pairs verified PyTorch model checkpoints with a full-stack web application built on **FastAPI** (Python backend) and a modern **React + Vite + Tailwind CSS** frontend.

---

## 1. Project Overview & Objectives

Convolutional Neural Networks (CNNs) have long formed the bedrock of computer vision due to strong inductive biases: **local receptive fields** and **translation equivariance**. Conversely, **Vision Transformers (ViT)** and hierarchical variants like the **Swin Transformer** (Liu et al., 2021) model long-range contextual relationships using multi-head self-attention with linear computational complexity via shifted local windows.

### Primary Project Objectives
1. **Generalization Benchmark**: Evaluate and compare classification accuracy, macro F1-scores, and confusion matrices across 5 animal categories from CIFAR-10.
2. **Edge CPU Latency Assessment**: Quantify single-image forward-pass latency (ms) and throughput (FPS) on consumer CPU hardware.
3. **Visual Feature Explainability**: Apply Gradient-weighted Class Activation Mapping (**Grad-CAM**) on the CNN's feature representations and analyze Swin-Tiny's window self-attention architecture.
4. **Interactive Dual-Model Web App**: Deliver an intuitive web application where users can upload any animal image, select quick test samples, and simultaneously observe both models' predictions and consensus analysis in real time.

---

## 2. Premium AI Dashboard Architecture & The Seven Sections

The user interface follows a modern white SaaS AI dashboard design with a persistent left sidebar on desktop, responsive slide-over drawer on mobile, subtle shadows, rounded cards, and elegant typography:

```
AnimalVision AI Dashboard Navigation
├── 1. Dashboard          -> Executive KPI metrics, held-out test split, quick dual prediction hub
├── 2. Animal Prediction  -> Dedicated side-by-side inference workspace with drag-and-drop upload
├── 3. Model Comparison   -> Quantitative benchmark table, accuracy/F1 comparative charts
├── 4. Data Efficiency    -> Empirical sample scaling study (10% to 100% subsets)
├── 5. Analytics          -> Interactive 10×10 confusion matrices, class reports, grouped bar charts
└── 6. About Project      -> Research methodology, inductive bias, dataset specs & tech stack
```

### 1. Dashboard (Central Research Hub)
* **Executive Overview**: High-level contrast of CNN v2 vs Swin-Tiny with key architecture summaries.
* **Top Metric KPI Cards**: Quick glance at test accuracy, parameter counts, and CPU inference latency.
* **The 5 Target Classes**: Visual cards for Bird, Cat, Deer, Dog, and Frog with per-class F1 performance and class indices.
* **Integrated Dual-Model Prediction Hub**: Directly test custom uploads or 1-click test samples without leaving the dashboard.
* **Architecture Comparison**: In-depth theoretical contrast between convolutional inductive bias and shifted-window self-attention.

### 2. Image Recognition (Dedicated Inference Workspace)
* **Drag-and-Drop Upload Area**: Large drop zone supporting PNG, JPG, JPEG, and WEBP formats with an explicit **Upload Image** button.
* **1-Click Test Samples**: Pre-loaded CIFAR-10 test animal images for instant evaluation.
* **Live Image Preview**: Displays image metadata and standardized resolution badge (`96×96`).
* **Side-by-Side Model Cards**: Animal emojis, predicted class name, confidence progress bars, top-3 candidate rankings, and exact CPU inference latency.
* **Consensus & Speed Comparison**: Agreement/divergence verdict, confidence margin, and CPU speedup ratio.
* **Inline Grad-CAM Trigger**: Expandable heatmap inspection right under prediction results.

### 3. Model Comparison (Quantitative Benchmark)
* **Connected to Backend Data**: Consumes official benchmark metrics from `results/test_metrics_comparison.csv`.
* **Head-to-Head KPI Cards**: Highlights test accuracy, macro F1, CPU latency, and throughput.
* **Visual Benchmark Bars**: Graphical comparative bars for test accuracy, macro F1, and continuous CPU throughput (FPS).
* **Complete Metrics Table**: Exhaustive quantitative breakdown covering accuracy, precision, recall, F1, latency percentiles (Mean, Median, P95), and parameter counts.

### 4. Analytics & Convergence (Deep Error Analysis)
* **Interactive 5×5 Confusion Matrices**: Shaded visual grids for CNN v2 and Swin-Tiny showing exact misclassifications and diagonal accuracy percentages. Direct link to the generated matplotlib plot (`/api/confusion-matrix-plot`).
* **Per-Class Classification Report**: Detailed precision, recall, and F1-score breakdown across all 1,000 test images per animal class.
* **Empirical Convergence Curves**: Interactive SVG line charts plotting real Epoch 1–10 loss and accuracy curves for CNN v2 and Epoch 1–5 for Swin-Tiny directly from `results/cnn_v2_history.csv` and `results/swin_tiny_history.csv`.

### 5. Explainability (Feature Attribution)
* **Target Class Selection**: Select automatic top prediction or force visual attribution for any of the 5 animal classes.
* **3-Panel Synchronized View**:
  1. *Input Image*: Preprocessed 96×96 image with ImageNet channel normalization.
  2. *Grad-CAM Heatmap*: Normalized Jet colormap activations from residual layer `stage4.conv2`.
  3. *Visual Overlay*: 55% alpha blend showing neural attention over the original photo.
* **Prediction Explanation**: Detailed anatomical interpretation of why the CNN model focused on specific visual features (beaks, wings, pointed ears, snouts, etc.).
* **Swin Attention Architecture Disclosure**: Transparent documentation explaining why Swin-Tiny's shifted-window attention is encapsulated inside PyTorch C++ kernels.

### 6. About Project (Research Documentation)
* Research motivation, methodology, and software architecture.
* Supported animal categories and core takeaways (Inductive Bias vs Representation Capacity vs Data Efficiency).
* Formal academic citations (Swin Transformer, ResNet, Animals-10).
* Expandable technical details drawer.

---

## 3. Dataset & The Five Animal Classes

The benchmark uses a controlled 5-class animal subset extracted from the official **CIFAR-10** dataset:

| Target Index | Animal Class Name | Original CIFAR-10 Index | Training Split | Validation Split | Official Test Split |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `bird` | 2 | 1,000 images | 200 images | 1,000 images |
| **1** | `cat` | 3 | 1,000 images | 200 images | 1,000 images |
| **2** | `deer` | 4 | 1,000 images | 200 images | 1,000 images |
| **3** | `dog` | 5 | 1,000 images | 200 images | 1,000 images |
| **4** | `frog` | 6 | 1,000 images | 200 images | 1,000 images |
| **Total** | **5 Classes** | — | **5,000 images** | **1,000 images** | **5,000 images** |

### Preprocessing & Normalization
* **Resolution**: 96 × 96 pixels (3-channel RGB), striking the balance between spatial feature resolution and CPU inference latency.
* **Standardization**: ImageNet Mean `[0.485, 0.456, 0.406]` and Std `[0.229, 0.224, 0.225]`.
* **Evaluation Integrity**: The official 5,000-image CIFAR-10 test set was kept strictly untouched during training and used solely for final benchmarking.

---

## 4. Model Architectures & Checkpoints

### Model 1: CNN v2 (AnimalResNet)
* **File**: `src/cnn_model.py` (`build_cnn_v2_model`)
* **Checkpoint**: `models/cnn_v2_best.pth` (15.18 MB)
* **Architecture**:
  * **Stem**: Conv2D (3 $\to$ 32, 3×3, stride 1, padding 1) + BatchNorm2D + ReLU.
  * **Stage 1**: ResidualBlock (32 $\to$ 32, stride 1, identity shortcut).
  * **Stage 2**: ResidualBlock (32 $\to$ 64, stride 2, 1×1 projection shortcut).
  * **Stage 3**: ResidualBlock (64 $\to$ 128, stride 2, 1×1 projection shortcut).
  * **Stage 4**: ResidualBlock (128 $\to$ 256, stride 2, 1×1 projection shortcut).
  * **Head**: `AdaptiveAvgPool2d((1, 1))` + Dropout(0.15) + Linear(256 $\to$ 128) + ReLU + Dropout(0.075) + Linear(128 $\to$ 5).
* **Total Parameters**: 1,259,941 (100% trainable, trained from scratch).
* **Training Setup**: AdamW ($1 \times 10^{-3}$, weight decay $1 \times 10^{-4}$), Cosine Annealing LR over 10 epochs.

### Model 2: Swin-Tiny Transformer
* **File**: `src/swin_model.py` (`build_swin_model`)
* **Checkpoint**: `models/swin_tiny_best.pth` (110.42 MB)
* **Architecture**:
  * Base: Torchvision `models.swin_t` (Swin Transformer Tiny).
  * Feature Representation: 768-dimensional latent vector via shifted-window self-attention.
  * Head: Replaced top classification head with `Linear(in_features=768, out_features=5, bias=True)`.
* **Total Parameters**: 27,523,199.
* **Trainable Parameters**: 3,845 (classification head adapted; backbone frozen with ImageNet-1K pretrained weights).
* **Training Setup**: AdamW ($1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$) over 5 epochs.

> **Methodological Disclosure**: Swin-Tiny benefits from 1.28 million ImageNet-1K representations, while CNN v2 was trained entirely from scratch on 5,000 CIFAR-10 images. This comparison reflects real-world engineering constraints: lightweight from-scratch CNNs versus fine-tuning pretrained vision transformers for CPU deployments.

---

## 5. Official Test Benchmark Results

Evaluated on all 5,000 official CIFAR-10 test set images (`src/evaluate.py`):

| Evaluation Metric | CNN v2 (ResNet-lite) | Swin-Tiny Transformer | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Training Regime** | Scratch (5k samples, 10 ep) | Pretrained Backbone (Frozen, 5 ep) | — |
| **Total Model Parameters** | 1,259,941 | 27,523,199 | +26.26M |
| **Trainable Parameters** | 1,259,941 | 3,845 | -1.25M |
| **Official Test Accuracy** | **55.94%** | **84.04%** | **+28.10% (Swin leads)** |
| **Macro Precision** | 55.83% | 84.11% | +28.28% |
| **Macro Recall** | 55.94% | 84.04% | +28.10% |
| **Macro F1-Score** | 55.44% | 84.06% | +28.62% |
| **Weighted F1-Score** | 55.44% | 84.06% | +28.62% |
| **Mean Single-Image CPU Latency** | **14.08 ms** | **47.91 ms** | **-33.83 ms (CNN 3.4x faster)** |
| **Median CPU Latency** | 13.39 ms | 46.90 ms | -33.51 ms |
| **P95 CPU Latency** | 17.79 ms | 56.31 ms | -38.52 ms |
| **Single-Image Throughput** | **71.0 FPS** | **20.9 FPS** | **CNN 3.4x higher throughput** |
| **Batch Evaluation Time (5k images)** | 83.23 s | 181.09 s | CNN 2.2x faster |

### Per-Class F1-Score Breakdown
| Class | Support | CNN v2 F1-Score | Swin-Tiny F1-Score | Lead |
| :--- | :---: | :---: | :---: | :---: |
| **Bird** | 1,000 | 51.0% | **85.4%** | Swin (+34.4%) |
| **Cat** | 1,000 | 43.4% | **77.8%** | Swin (+34.4%) |
| **Deer** | 1,000 | 59.6% | **86.0%** | Swin (+26.4%) |
| **Dog** | 1,000 | 55.2% | **83.4%** | Swin (+28.2%) |
| **Frog** | 1,000 | 68.0% | **87.8%** | Swin (+19.8%) |

---

## 6. Installation & Execution Guide

### Prerequisites
* **Operating System**: Windows 10/11, macOS, or Linux.
* **Python**: Python 3.10 or 3.11.
* **Node.js**: Node.js 18+ and npm (only needed if rebuilding frontend assets).

### Quick Setup

1. **Navigate to the project root directory**:
   ```powershell
   cd "c:\Users\Laptop\OneDrive\Desktop\Swin vs CNN"
   ```

2. **Install Python dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **(Optional) Rebuild the React frontend**:
   *The production bundle is already compiled in `frontend/dist/`. If you modify frontend code, recompile via:*
   ```powershell
   cd frontend
   npm.cmd install
   npm.cmd run build
   cd ..
   ```

---

### Running the Application

#### Option A: 1-Click Launch (Recommended)
Runs the FastAPI server and automatically opens your web browser:
```powershell
python start_app.py
```
*Access the application at **`http://127.0.0.1:8000`**.*

#### Option B: Dedicated Development Servers
For live hot-reloading during code edits:

* **Terminal 1 (Backend API)**:
  ```powershell
  uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
  ```
* **Terminal 2 (Frontend Dev Server)**:
  ```powershell
  cd frontend
  npm.cmd run dev
  ```
  *Open `http://localhost:5173` (requests automatically proxy to `/api` on port 8000).*

#### Option C: Re-running the Test Set Evaluation Benchmark
To re-evaluate both model checkpoints on the complete 5,000-image CIFAR-10 test split:
```powershell
python -m src.evaluate --batch-size 64
```

---

## 7. Backend API Documentation

Interactive OpenAPI / Swagger documentation is available at **`http://127.0.0.1:8000/docs`**.

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Health check and model loading readiness |
| `GET` | `/api/models-info` | Architectural parameter counts and training setups |
| `GET` | `/api/sample-images` | Pre-cached CIFAR-10 test animal samples with base64 data URLs |
| `POST` | `/api/predict` | Multipart image upload for dual-model side-by-side inference |
| `POST` | `/api/gradcam` | Computes genuine Grad-CAM on CNN v2 (`stage4.conv2`) |
| `GET` | `/api/swin-attention-status` | Technical report on Swin attention functional encapsulation |
| `GET` | `/api/evaluation-metrics` | Official test metrics, per-class reports, and confusion data |
| `GET` | `/api/confusion-matrix-plot` | Serves the generated confusion matrix PNG |

---

## 8. Project Structure

```
Swin vs CNN/
├── backend/
│   ├── app.py                     # FastAPI application & route definitions
│   └── inference.py               # ModelManager, Grad-CAM engine & feature extraction
├── frontend/
│   ├── dist/                      # Pre-compiled production React distribution
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx        # Sidebar navigation (6 active pages)
│   │   │   ├── Header.jsx         # Header status & breadcrumb bar
│   │   │   ├── PredictionSection.jsx # Main image upload & recognition view
│   │   │   ├── ImageUploader.jsx  # Drag-and-drop upload zone & sample picker
│   │   │   ├── PredictionCard.jsx # Model prediction card (emoji, conf, top-3)
│   │   │   ├── ComparisonSection.jsx # Primary held-out test benchmarks
│   │   │   ├── DataEfficiencySection.jsx # 10% to 100% sample scaling study
│   │   │   ├── AnalyticsSection.jsx # 10-class grouped bar charts & confusion matrices
│   │   │   └── AboutSection.jsx   # Project takeaways & technical specs
│   │   ├── App.jsx                # Main application state & tab router
│   │   └── index.css              # Tailwind styling & animations
│   ├── index.html                 # HTML entry point
│   ├── package.json               # Node.js dependencies
│   └── vite.config.js             # Vite build & proxy configuration
├── models/
│   ├── resnet18_animals10_best.pth # ResNet-18 trained checkpoint (44.8 MB)
│   ├── swin_tiny_animals10_best.pth # Swin-Tiny trained checkpoint (110.4 MB)
│   ├── cnn_v2_best.pth            # CNN v2 trained checkpoint (15.2 MB)
│   └── swin_tiny_best.pth         # Swin-Tiny trained checkpoint (110.4 MB)
├── results/
│   ├── test_evaluation_summary.md # Comprehensive test set markdown report
│   ├── test_metrics_comparison.csv # Test metrics table (accuracy, latency, F1)
│   ├── test_classification_reports.json # Per-class precision, recall, F1
│   ├── test_confusion_matrices.png # Generated confusion matrix plot
│   └── prediction_history.json    # Persistent user prediction history log
├── src/
│   ├── dataset.py                 # CIFAR-10 5-class animal filtering & loader
│   ├── cnn_model.py               # AnimalCNN & AnimalResNet (CNN v2) architectures
│   ├── swin_model.py              # Swin-Tiny Transformer adaptation
│   ├── train.py                   # Model training script
│   └── evaluate.py                # Test set evaluation & benchmark suite
├── requirements.txt               # Pinned Python dependencies
├── start_app.py                   # 1-click full-stack application launcher
└── README.md                      # Project documentation
```

---

## 9. Academic References & Citations

1. **Swin Transformer**:
   Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*. IEEE/CVF International Conference on Computer Vision (ICCV), 10012–10022.
2. **Deep Residual Learning (ResNet)**:
   He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770–778.
3. **Grad-CAM**:
   Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE International Conference on Computer Vision (ICCV), 618–626.
4. **CIFAR-10 Dataset**:
   Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. University of Toronto Technical Report.
