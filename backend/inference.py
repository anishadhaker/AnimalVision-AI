"""
Inference engine and explainability services for CNN v2 and Swin-Tiny Animal Classification.

Provides:
- Loading model checkpoints (cnn_v2_best.pth and swin_tiny_best.pth) on CPU once at startup.
- Dual-model side-by-side inference with top-3 predictions, confidence %, latency, and consensus.
- Genuine Grad-CAM for CNN v2 (ResNet-lite) using target layer stage4.conv2.
- Swin-Tiny attention status disclosure (genuine research reporting).
- Robustness testing suite (original, rotation 15°, brightness +35%, Gaussian blur).
- Prediction history persistence and retrieval.
- Official test-set metrics and comparison data serving.
"""

import os
import io
import time
import json
import csv
import base64
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from src.cnn_model import build_cnn_v2_model, count_parameters as count_cnn_params
from src.swin_model import build_swin_model, count_parameters as count_swin_params
from src.resnet_model import build_resnet18_model, count_parameters as count_resnet_params
from src.dataset import ANIMAL_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from src.animals10_dataset import ANIMALS10_CLASSES
try:
    from backend.ood_detector import OODDetector
except ImportError:
    from ood_detector import OODDetector

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")


def pil_to_base64_data_url(image: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Converts a PIL Image to a base64 data URL string."""
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format.upper() == "JPEG" else f"image/{format.lower()}"
    return f"data:{mime};base64,{b64}"


class ModelManager:
    """
    Singleton manager that loads and caches CNN v2 and Swin-Tiny in memory for instant API inference.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.cnn_model: Optional[nn.Module] = None
        self.swin_model: Optional[nn.Module] = None
        self.cnn_meta: Dict[str, Any] = {}
        self.swin_meta: Dict[str, Any] = {}
        self.img_size: int = 96
        self.transform: Optional[transforms.Compose] = None
        self.is_loaded: bool = False

    def load_models(self):
        if self.is_loaded:
            return

        print(f"[ModelManager] Loading model checkpoints onto {self.device}...", flush=True)

        cnn_path = os.path.join(MODELS_DIR, "cnn_v2_best.pth")
        swin_path = os.path.join(MODELS_DIR, "swin_tiny_best.pth")

        if not os.path.exists(cnn_path):
            raise FileNotFoundError(f"CNN v2 checkpoint not found at: {cnn_path}")
        if not os.path.exists(swin_path):
            raise FileNotFoundError(f"Swin-Tiny checkpoint not found at: {swin_path}")

        # 1. Load CNN v2 (AnimalResNet) Checkpoint
        cnn_ckpt = torch.load(cnn_path, map_location=self.device)
        self.cnn_model = build_cnn_v2_model(num_classes=5, dropout_rate=0.15)
        self.cnn_model.load_state_dict(cnn_ckpt["model_state_dict"])
        self.cnn_model.to(self.device)
        self.cnn_model.eval()

        cnn_total_p, cnn_train_p = count_cnn_params(self.cnn_model)
        self.img_size = cnn_ckpt.get("img_size", 96)
        self.cnn_meta = {
            "epoch": cnn_ckpt.get("epoch", 9),
            "val_acc": cnn_ckpt.get("val_acc", 57.80),
            "val_loss": cnn_ckpt.get("val_loss", 1.082),
            "total_params": cnn_total_p,
            "trainable_params": cnn_train_p,
            "training_setup": "Trained from scratch (5,000 samples, 10 epochs)",
            "img_size": self.img_size,
        }

        # 2. Load Swin-Tiny Checkpoint
        swin_ckpt = torch.load(swin_path, map_location=self.device)
        self.swin_model = build_swin_model(num_classes=5, pretrained=False)
        self.swin_model.load_state_dict(swin_ckpt["model_state_dict"])
        self.swin_model.to(self.device)
        self.swin_model.eval()

        swin_total_p, _ = count_swin_params(self.swin_model)
        swin_head_p = sum(p.numel() for p in self.swin_model.head.parameters())
        self.swin_meta = {
            "epoch": swin_ckpt.get("epoch", 5),
            "val_acc": swin_ckpt.get("val_acc", 84.50),
            "val_loss": swin_ckpt.get("val_loss", 0.4653),
            "total_params": swin_total_p,
            "trainable_params": swin_head_p,
            "training_setup": "Frozen ImageNet-1K pretrained backbone (5,000 samples, 5 epochs)",
            "img_size": self.img_size,
        }

        # 3. Setup Preprocessing Pipeline (Identical 96x96 ImageNet normalization for both)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.is_loaded = True
        print(f"[ModelManager] CNN v2 and Swin-Tiny successfully loaded! Target resolution: {self.img_size}x{self.img_size}", flush=True)

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts PIL image to preprocessed tensor batch [1, 3, 96, 96]."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def _infer_single(self, model: nn.Module, tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        """Runs forward pass with precise timing."""
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return probs, latency_ms

    def predict(self, image_bytes: bytes, log_history: bool = True) -> Dict[str, Any]:
        """
        Runs side-by-side inference for CNN v2 and Swin-Tiny.
        Computes top-3 predictions, confidence %, latency, and consensus.
        """
        if not self.is_loaded:
            self.load_models()

        raw_pil = Image.open(io.BytesIO(image_bytes))
        if raw_pil.mode != "RGB":
            raw_pil = raw_pil.convert("RGB")

        tensor = self.preprocess_image(raw_pil)

        # 1. CNN v2 Inference
        cnn_probs, cnn_lat_ms = self._infer_single(self.cnn_model, tensor)
        cnn_pred_idx = int(cnn_probs.argmax())
        cnn_pred_class = ANIMAL_CLASSES[cnn_pred_idx]
        cnn_confidence = round(float(cnn_probs[cnn_pred_idx]) * 100, 2)

        # 2. Swin-Tiny Inference
        swin_probs, swin_lat_ms = self._infer_single(self.swin_model, tensor)
        swin_pred_idx = int(swin_probs.argmax())
        swin_pred_class = ANIMAL_CLASSES[swin_pred_idx]
        swin_confidence = round(float(swin_probs[swin_pred_idx]) * 100, 2)

        # Top-3 predictions helper
        def get_top3(probs: np.ndarray) -> List[Dict[str, Any]]:
            indices = np.argsort(probs)[::-1][:3]
            return [
                {
                    "rank": rank + 1,
                    "class_name": ANIMAL_CLASSES[idx],
                    "class_idx": int(idx),
                    "probability": round(float(probs[idx]) * 100, 2),
                }
                for rank, idx in enumerate(indices)
            ]

        cnn_top3 = get_top3(cnn_probs)
        swin_top3 = get_top3(swin_probs)

        # All class probabilities
        cnn_all_probs = {
            ANIMAL_CLASSES[i]: round(float(cnn_probs[i]) * 100, 2) for i in range(len(ANIMAL_CLASSES))
        }
        swin_all_probs = {
            ANIMAL_CLASSES[i]: round(float(swin_probs[i]) * 100, 2) for i in range(len(ANIMAL_CLASSES))
        }

        # Consensus and agreement analysis
        agreement = (cnn_pred_class == swin_pred_class)
        conf_diff = round(abs(swin_confidence - cnn_confidence), 2)

        if agreement:
            verdict = (
                f"Consensus: Both models agree on '{swin_pred_class.upper()}' "
                f"(Swin: {swin_confidence}%, CNN v2: {cnn_confidence}%)."
            )
        else:
            verdict = (
                f"Divergence: Swin-Tiny predicts '{swin_pred_class.upper()}' ({swin_confidence}%), "
                f"while CNN v2 predicts '{cnn_pred_class.upper()}' ({cnn_confidence}%)."
            )

        # Prepare base64 thumbnail
        thumbnail = raw_pil.copy()
        thumbnail.thumbnail((120, 120))
        thumbnail_b64 = pil_to_base64_data_url(thumbnail, quality=80)

        response = {
            "agreement": agreement,
            "verdict": verdict,
            "confidence_margin": conf_diff,
            "image_preview": thumbnail_b64,
            "cnn": {
                "name": "CNN v2 (ResNet-lite)",
                "predicted_class": cnn_pred_class,
                "confidence": cnn_confidence,
                "latency_ms": cnn_lat_ms,
                "top3": cnn_top3,
                "probabilities": cnn_all_probs,
                "training_setup": self.cnn_meta["training_setup"],
                "total_params": self.cnn_meta["total_params"],
            },
            "swin": {
                "name": "Swin-Tiny Transformer",
                "predicted_class": swin_pred_class,
                "confidence": swin_confidence,
                "latency_ms": swin_lat_ms,
                "top3": swin_top3,
                "probabilities": swin_all_probs,
                "training_setup": self.swin_meta["training_setup"],
                "total_params": self.swin_meta["total_params"],
            },
            "image_meta": {
                "input_resolution": f"{self.img_size}x{self.img_size}",
                "classes": ANIMAL_CLASSES,
            }
        }

        return response

    def generate_gradcam(
        self,
        image_bytes: bytes,
        target_class_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates genuine Grad-CAM on CNN v2 (ResNet-lite) using target layer stage4.conv2.
        Extracts actual forward activations and backward gradients without fabricated maps.
        """
        if not self.is_loaded:
            self.load_models()

        raw_pil = Image.open(io.BytesIO(image_bytes))
        if raw_pil.mode != "RGB":
            raw_pil = raw_pil.convert("RGB")

        # Resize display copy to standard 224x224 for crisp visual display
        display_pil = raw_pil.resize((224, 224), Image.Resampling.BILINEAR)

        # Preprocess input tensor with gradient tracking
        tensor = self.preprocess_image(raw_pil)
        tensor.requires_grad = True

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        target_layer = self.cnn_model.stage4.conv2
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            self.cnn_model.zero_grad()
            with torch.enable_grad():
                logits = self.cnn_model(tensor)
                probs = F.softmax(logits, dim=-1)[0]

                if target_class_idx is None or not isinstance(target_class_idx, int) or not (0 <= target_class_idx < len(ANIMAL_CLASSES)):
                    target_class_idx = int(logits.argmax(dim=-1).item())

                predicted_class = ANIMAL_CLASSES[target_class_idx]
                confidence = round(float(probs[target_class_idx].item()) * 100, 2)

                score = logits[0, target_class_idx]
                score.backward()

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks failed to record activations or gradients.")

            act = activations[0].detach()  # Shape: [1, 256, H, W]
            grad = gradients[0].detach()   # Shape: [1, 256, H, W]

            # Global average pooling of gradients over spatial dimensions
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, 256, 1, 1]

            # Weighted linear combination of feature maps
            cam = torch.sum(weights * act, dim=1, keepdim=True)  # [1, 1, H, W]
            cam = F.relu(cam)

            # Upsample heatmap to match display size (224x224)
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam_np = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            c_min, c_max = cam_np.min(), cam_np.max()
            if c_max > c_min:
                cam_norm = (cam_np - c_min) / (c_max - c_min)
            else:
                cam_norm = np.zeros_like(cam_np)

            # Generate colored heatmap using colormap 'jet'
            cmap = matplotlib.colormaps["jet"]
            colored_heatmap = cmap(cam_norm)[:, :, :3]  # RGB in [0, 1]
            colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(colored_heatmap_uint8)

            # Blend with original image (50% original, 50% heatmap)
            overlay_pil = Image.blend(display_pil, heatmap_pil, alpha=0.55)

            # Convert images to base64 data URLs
            original_b64 = pil_to_base64_data_url(display_pil)
            heatmap_b64 = pil_to_base64_data_url(heatmap_pil)
            overlay_b64 = pil_to_base64_data_url(overlay_pil)

            return {
                "success": True,
                "target_class": predicted_class,
                "target_class_idx": target_class_idx,
                "confidence": confidence,
                "layer_name": "stage4.conv2 (Final Residual Stage)",
                "original_image": original_b64,
                "heatmap_image": heatmap_b64,
                "overlay_image": overlay_b64,
                "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
            }
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    def test_robustness(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs robustness perturbations on the uploaded image:
        1. Original image (baseline)
        2. Rotation (+15°)
        3. Brightness adjustment (+35%)
        4. Gaussian Blur (radius=2.0)
        Computes predictions, confidences, and top-3 for both models on each variation.
        """
        if not self.is_loaded:
            self.load_models()

        raw_pil = Image.open(io.BytesIO(image_bytes))
        if raw_pil.mode != "RGB":
            raw_pil = raw_pil.convert("RGB")

        # 4 Variations
        variations = [
            {
                "id": "original",
                "name": "Original Image",
                "description": "Unperturbed input image as baseline reference.",
                "image": raw_pil,
            },
            {
                "id": "rotation",
                "name": "Rotation (+15°)",
                "description": "Affine rotation by 15° with bilinear interpolation.",
                "image": raw_pil.rotate(15, expand=False, resample=Image.Resampling.BILINEAR),
            },
            {
                "id": "brightness",
                "name": "Brightness (+35%)",
                "description": "Photometric illumination shift (brightness factor 1.35).",
                "image": ImageEnhance.Brightness(raw_pil).enhance(1.35),
            },
            {
                "id": "blur",
                "name": "Gaussian Blur (r=2.0)",
                "description": "High-frequency attenuation via Gaussian kernel (radius 2.0).",
                "image": raw_pil.filter(ImageFilter.GaussianBlur(radius=2.0)),
            },
        ]

        def get_top3(probs: np.ndarray) -> List[Dict[str, Any]]:
            indices = np.argsort(probs)[::-1][:3]
            return [
                {
                    "class_name": ANIMAL_CLASSES[idx],
                    "probability": round(float(probs[idx]) * 100, 2),
                }
                for idx in indices
            ]

        results = []
        for var in variations:
            img = var["image"]
            tensor = self.preprocess_image(img)

            cnn_probs, cnn_lat = self._infer_single(self.cnn_model, tensor)
            swin_probs, swin_lat = self._infer_single(self.swin_model, tensor)

            cnn_idx = int(cnn_probs.argmax())
            swin_idx = int(swin_probs.argmax())

            cnn_pred = ANIMAL_CLASSES[cnn_idx]
            swin_pred = ANIMAL_CLASSES[swin_idx]

            cnn_conf = round(float(cnn_probs[cnn_idx]) * 100, 2)
            swin_conf = round(float(swin_probs[swin_idx]) * 100, 2)

            agreement = (cnn_pred == swin_pred)

            preview_img = img.copy()
            preview_img.thumbnail((160, 160))
            preview_b64 = pil_to_base64_data_url(preview_img)

            results.append({
                "id": var["id"],
                "name": var["name"],
                "description": var["description"],
                "image_preview": preview_b64,
                "agreement": agreement,
                "cnn": {
                    "predicted_class": cnn_pred,
                    "confidence": cnn_conf,
                    "top3": get_top3(cnn_probs),
                    "latency_ms": cnn_lat,
                },
                "swin": {
                    "predicted_class": swin_pred,
                    "confidence": swin_conf,
                    "top3": get_top3(swin_probs),
                    "latency_ms": swin_lat,
                },
            })

        # Summary of stability
        orig_cnn = results[0]["cnn"]["predicted_class"]
        orig_swin = results[0]["swin"]["predicted_class"]

        cnn_stable_count = sum(1 for r in results if r["cnn"]["predicted_class"] == orig_cnn)
        swin_stable_count = sum(1 for r in results if r["swin"]["predicted_class"] == orig_swin)

        return {
            "variations": results,
            "stability_summary": {
                "cnn_stability_rate": round((cnn_stable_count / len(results)) * 100, 1),
                "swin_stability_rate": round((swin_stable_count / len(results)) * 100, 1),
                "cnn_baseline_class": orig_cnn,
                "swin_baseline_class": orig_swin,
            }
        }


# Global model manager singleton
model_manager = ModelManager()


def get_models_info() -> Dict[str, Any]:
    """
    Returns architecture specs, parameters, training setup, and validation stats.
    """
    cnn_cfg_path = os.path.join(RESULTS_DIR, "cnn_v2_config.json")
    swin_cfg_path = os.path.join(RESULTS_DIR, "swin_tiny_config.json")

    cnn_cfg = {}
    swin_cfg = {}

    if os.path.exists(cnn_cfg_path):
        with open(cnn_cfg_path, "r", encoding="utf-8") as f:
            cnn_cfg = json.load(f)

    if os.path.exists(swin_cfg_path):
        with open(swin_cfg_path, "r", encoding="utf-8") as f:
            swin_cfg = json.load(f)

    return {
        "classes": ANIMAL_CLASSES,
        "device": "CPU",
        "resolution": "96x96",
        "cnn": {
            "name": "CNN v2 (ResNet-lite)",
            "type": "Residual Convolutional Network",
            "total_params": 1259941,
            "trainable_params": 1259941,
            "frozen_params": 0,
            "best_val_accuracy": cnn_cfg.get("best_val_accuracy", 57.80),
            "epochs_completed": cnn_cfg.get("epochs_completed", 10),
            "training_samples": cnn_cfg.get("total_train_samples", 5000),
            "validation_samples": cnn_cfg.get("total_val_samples", 1000),
            "learning_rate": cnn_cfg.get("learning_rate", 0.001),
            "backbone_status": "Trained from scratch with residual skip connections",
        },
        "swin": {
            "name": "Swin-Tiny Transformer",
            "type": "Hierarchical Vision Transformer",
            "total_params": 27523199,
            "trainable_params": 3845,
            "frozen_params": 27519354,
            "best_val_accuracy": swin_cfg.get("best_val_accuracy", 84.50),
            "epochs_completed": swin_cfg.get("epochs_completed", 5),
            "training_samples": swin_cfg.get("total_train_samples", 5000),
            "validation_samples": swin_cfg.get("total_val_samples", 1000),
            "learning_rate": swin_cfg.get("learning_rate", 0.0001),
            "backbone_status": "Frozen Pretrained (ImageNet-1K), head trained on 5 classes",
        },
        "scientific_notice": (
            "Comparative note: Swin-Tiny leverages a frozen ImageNet-1K pretrained backbone (1.28M images) "
            "with a 3,845-param trained head. CNN v2 (ResNet-lite, 1.26M params) was trained from scratch "
            "on 5,000 CIFAR-10 animal images. Both evaluate on the official CIFAR-10 test set at 96x96 resolution."
        )
    }


def get_training_history() -> Dict[str, Any]:
    """
    Parses cnn_v2_history.csv and swin_tiny_history.csv into structured chart data.
    """
    cnn_csv_path = os.path.join(RESULTS_DIR, "cnn_v2_history.csv")
    swin_csv_path = os.path.join(RESULTS_DIR, "swin_tiny_history.csv")

    def read_csv(path: str) -> List[Dict[str, Any]]:
        rows = []
        if os.path.exists(path):
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append({
                        "epoch": int(r["epoch"]),
                        "train_loss": round(float(r["train_loss"]), 4),
                        "train_acc": round(float(r["train_acc"]), 2),
                        "val_loss": round(float(r["val_loss"]), 4),
                        "val_acc": round(float(r["val_acc"]), 2),
                        "duration_seconds": round(float(r["duration_seconds"]), 1),
                    })
        return rows

    return {
        "cnn": read_csv(cnn_csv_path),
        "swin": read_csv(swin_csv_path),
        "classes": ANIMAL_CLASSES,
    }


def get_evaluation_metrics() -> Dict[str, Any]:
    """
    Loads official test-set benchmark results from results/test_metrics_comparison.csv
    and results/test_classification_reports.json.
    """
    csv_path = os.path.join(RESULTS_DIR, "test_metrics_comparison.csv")
    json_path = os.path.join(RESULTS_DIR, "test_classification_reports.json")

    comparison_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                comparison_rows.append({
                    "model": r["model"],
                    "training_setup": r["training_setup"],
                    "total_params": int(r["total_params"]),
                    "trainable_params_training": int(r["trainable_params_training"]),
                    "frozen_params_training": int(r["frozen_params_training"]),
                    "test_accuracy": float(r["test_accuracy"]),
                    "macro_precision": float(r["macro_precision"]),
                    "macro_recall": float(r["macro_recall"]),
                    "macro_f1": float(r["macro_f1"]),
                    "weighted_f1": float(r["weighted_f1"]),
                    "mean_latency_ms": float(r["mean_latency_ms"]),
                    "median_latency_ms": float(r["median_latency_ms"]),
                    "p95_latency_ms": float(r["p95_latency_ms"]),
                    "throughput_fps": float(r["throughput_fps"]),
                    "eval_duration_seconds": float(r["eval_duration_seconds"]),
                })

    reports_data = {}
    if os.path.exists(json_path):
        with open(json_path, mode="r", encoding="utf-8") as f:
            reports_data = json.load(f)

    return {
        "available": bool(comparison_rows),
        "comparison_table": comparison_rows,
        "classification_reports": reports_data,
        "classes": ANIMAL_CLASSES,
        "test_dataset_size": 5000,
        "test_samples_per_class": 1000,
    }


def get_swin_attention_status() -> Dict[str, Any]:
    """
    Returns authentic research disclosure on Swin-Tiny attention map availability.
    """
    return {
        "available": False,
        "feature_name": "Swin-Tiny Window Attention Visualization",
        "status_badge": "Unavailable by Architecture Design",
        "investigation_summary": (
            "Torchvision's official Swin Transformer implementation (torchvision.models.swin_t) computes "
            "multi-head self-attention inside the functional shifted_window_attention() routine in local functional/C++ scope. "
            "These intermediate attention matrices are not cached or returned in module state, nor exposed by PyTorch forward hooks. "
            "Furthermore, Swin operates on shifted local 7x7 windows rather than global token arrays."
        ),
        "research_integrity_disclosure": (
            "To adhere strictly to AI research integrity standards, we do not fabricate random or simulated attention heatmaps. "
            "Genuine Grad-CAM is provided for CNN v2, which directly computes gradients with respect to convolutional feature maps."
        )
    }


# =============================================================================
# Animals-10 (10-Class) Model Manager & Helper Functions
# =============================================================================
# This section is entirely additive. The existing 5-class ModelManager,
# singleton, and helper functions above are NOT modified.
# =============================================================================

ANIMALS10_RESULTS_DIR = os.path.join(BASE_DIR, "results", "animals10")
ANIMALS10_DATA_DIR = os.path.join(BASE_DIR, "data", "animals-10", "splits")
ANIMALS10_DATA_EFFICIENCY_CSV = os.path.join(ANIMALS10_RESULTS_DIR, "data_efficiency_metrics.csv")
ANIMALS10_DATA_EFFICIENCY_PLOT = os.path.join(ANIMALS10_RESULTS_DIR, "data_efficiency_comparison.png")
ANIMALS10_DATA_EFFICIENCY_INDICES = os.path.join(ANIMALS10_RESULTS_DIR, "data_efficiency_indices.json")


class Animals10ModelManager:
    """
    Singleton manager for 10-class Animals-10 models (ResNet-18 + Swin-Tiny).
    Completely separate from the 5-class ModelManager to preserve backward compatibility.

    Key differences from the 5-class ModelManager:
    - Input resolution: 224x224 (Resize(256) + CenterCrop(224))
    - CNN architecture: torchvision.models.resnet18 (not AnimalResNet)
    - Checkpoint key schema: best_epoch/best_val_acc (not epoch/val_acc)
    - Class list: 10 classes from ANIMALS10_CLASSES
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.resnet_model: Optional[nn.Module] = None
        self.swin_model: Optional[nn.Module] = None
        self.resnet_meta: Dict[str, Any] = {}
        self.swin_meta: Dict[str, Any] = {}
        self.img_size: int = 224
        self.transform: Optional[transforms.Compose] = None
        self.is_loaded: bool = False

    def load_models(self):
        """Load both 10-class checkpoints into memory for instant API inference."""
        if self.is_loaded:
            return

        print("[Animals10ModelManager] Loading 10-class model checkpoints...", flush=True)

        resnet_path = os.path.join(MODELS_DIR, "resnet18_animals10_best.pth")
        swin_path = os.path.join(MODELS_DIR, "swin_tiny_animals10_best.pth")

        if not os.path.exists(resnet_path):
            raise FileNotFoundError(f"ResNet-18 Animals-10 checkpoint not found at: {resnet_path}")
        if not os.path.exists(swin_path):
            raise FileNotFoundError(f"Swin-Tiny Animals-10 checkpoint not found at: {swin_path}")

        # 1. Load ResNet-18 (10-class)
        resnet_ckpt = torch.load(resnet_path, map_location=self.device)
        self.resnet_model = build_resnet18_model(num_classes=10, pretrained=False)
        self.resnet_model.load_state_dict(resnet_ckpt["model_state_dict"])
        self.resnet_model.to(self.device)
        self.resnet_model.eval()

        resnet_total, resnet_train = count_resnet_params(self.resnet_model)
        self.resnet_meta = {
            "best_epoch": resnet_ckpt.get("best_epoch"),
            "best_val_acc": resnet_ckpt.get("best_val_acc"),
            "best_val_loss": resnet_ckpt.get("best_val_loss"),
            "total_params": resnet_total,
            "trainable_params": resnet_train,
            "training_protocol": resnet_ckpt.get("training_protocol", "Frozen-Backbone Transfer Learning"),
            "img_size": resnet_ckpt.get("img_size", 224),
        }

        # 2. Load Swin-Tiny (10-class)
        swin_ckpt = torch.load(swin_path, map_location=self.device)
        self.swin_model = build_swin_model(num_classes=10, pretrained=False)
        self.swin_model.load_state_dict(swin_ckpt["model_state_dict"])
        self.swin_model.to(self.device)
        self.swin_model.eval()

        swin_total, swin_train = count_swin_params(self.swin_model)
        self.swin_meta = {
            "best_epoch": swin_ckpt.get("best_epoch"),
            "best_val_acc": swin_ckpt.get("best_val_acc"),
            "best_val_loss": swin_ckpt.get("best_val_loss"),
            "total_params": swin_total,
            "trainable_params": swin_train,
            "training_protocol": swin_ckpt.get("training_protocol", "Frozen-Backbone Transfer Learning"),
            "img_size": swin_ckpt.get("img_size", 224),
        }

        # 3. Setup preprocessing: Resize(256) + CenterCrop(224) + ImageNet normalization
        #    Must match training-time eval transform from src/animals10_dataset.py
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.is_loaded = True
        print(
            f"[Animals10ModelManager] ResNet-18 and Swin-Tiny (10-class) loaded! "
            f"Resolution: {self.img_size}x{self.img_size}",
            flush=True,
        )

    def preprocess_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts PIL image to preprocessed tensor batch [1, 3, 224, 224]."""
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def _infer_single(
        self,
        model: nn.Module,
        tensor: torch.Tensor,
        return_logits: bool = False,
    ):
        """Runs forward pass with precise timing. Optionally returns raw logits."""
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            logits_np = logits[0].cpu().numpy()
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        if return_logits:
            return probs, latency_ms, logits_np
        return probs, latency_ms

    def predict(self, image_bytes: bytes, log_history: bool = True) -> Dict[str, Any]:
        """
        Runs side-by-side inference for ResNet-18 and Swin-Tiny on 10 animal classes.
        Computes top-3 predictions, confidence %, latency, and consensus.
        """
        if not self.is_loaded:
            self.load_models()

        raw_pil = Image.open(io.BytesIO(image_bytes))
        if raw_pil.mode != "RGB":
            raw_pil = raw_pil.convert("RGB")

        tensor = self.preprocess_image(raw_pil)

        # 1. ResNet-18 Inference
        resnet_probs, resnet_lat_ms, resnet_logits = self._infer_single(
            self.resnet_model, tensor, return_logits=True
        )
        resnet_pred_idx = int(resnet_probs.argmax())
        resnet_pred_class = ANIMALS10_CLASSES[resnet_pred_idx]
        resnet_confidence = round(float(resnet_probs[resnet_pred_idx]) * 100, 2)

        # 2. Swin-Tiny Inference
        swin_probs, swin_lat_ms, swin_logits = self._infer_single(
            self.swin_model, tensor, return_logits=True
        )
        swin_pred_idx = int(swin_probs.argmax())
        swin_pred_class = ANIMALS10_CLASSES[swin_pred_idx]
        swin_confidence = round(float(swin_probs[swin_pred_idx]) * 100, 2)

        # 3. Out-of-Distribution (OOD) & Unsupported Image Detection
        ood_result = OODDetector.evaluate(
            resnet_logits=resnet_logits,
            swin_logits=swin_logits,
            resnet_probs=resnet_probs,
            swin_probs=swin_probs,
            resnet_pred_class=resnet_pred_class,
            swin_pred_class=swin_pred_class,
        )

        # Top-3 predictions helper
        def get_top3(probs: np.ndarray) -> List[Dict[str, Any]]:
            indices = np.argsort(probs)[::-1][:3]
            return [
                {
                    "rank": rank + 1,
                    "class_name": ANIMALS10_CLASSES[idx],
                    "class_idx": int(idx),
                    "probability": round(float(probs[idx]) * 100, 2),
                }
                for rank, idx in enumerate(indices)
            ]

        resnet_top3 = get_top3(resnet_probs)
        swin_top3 = get_top3(swin_probs)

        # All class probabilities
        resnet_all_probs = {
            ANIMALS10_CLASSES[i]: round(float(resnet_probs[i]) * 100, 2)
            for i in range(len(ANIMALS10_CLASSES))
        }
        swin_all_probs = {
            ANIMALS10_CLASSES[i]: round(float(swin_probs[i]) * 100, 2)
            for i in range(len(ANIMALS10_CLASSES))
        }

        # Consensus and agreement analysis
        agreement = (resnet_pred_class == swin_pred_class)
        conf_diff = round(abs(swin_confidence - resnet_confidence), 2)

        if agreement:
            verdict = (
                f"Consensus: Both models agree on '{swin_pred_class.upper()}' "
                f"(Swin: {swin_confidence}%, ResNet-18: {resnet_confidence}%)."
            )
        else:
            verdict = (
                f"Divergence: Swin-Tiny predicts '{swin_pred_class.upper()}' ({swin_confidence}%), "
                f"while ResNet-18 predicts '{resnet_pred_class.upper()}' ({resnet_confidence}%)."
            )

        # Prepare base64 thumbnail
        thumbnail = raw_pil.copy()
        thumbnail.thumbnail((120, 120))
        thumbnail_b64 = pil_to_base64_data_url(thumbnail, quality=80)

        response = {
            "agreement": agreement,
            "verdict": verdict,
            "confidence_margin": conf_diff,
            "confidence_difference": conf_diff,
            "confidence_difference_pp": conf_diff,
            "image_preview": thumbnail_b64,
            "dataset": "Animals-10",
            "num_classes": 10,
            "ood_detection": ood_result,
            "cnn": {
                "name": "ResNet-18 (ImageNet Pretrained)",
                "predicted_class": resnet_pred_class,
                "confidence": resnet_confidence,
                "latency_ms": resnet_lat_ms,
                "top3": resnet_top3,
                "probabilities": resnet_all_probs,
                "training_setup": self.resnet_meta.get("training_protocol", ""),
                "total_params": self.resnet_meta["total_params"],
            },
            "swin": {
                "name": "Swin-Tiny Transformer",
                "predicted_class": swin_pred_class,
                "confidence": swin_confidence,
                "latency_ms": swin_lat_ms,
                "top3": swin_top3,
                "probabilities": swin_all_probs,
                "training_setup": self.swin_meta.get("training_protocol", ""),
                "total_params": self.swin_meta["total_params"],
            },
            "image_meta": {
                "input_resolution": f"{self.img_size}x{self.img_size}",
                "preprocessing": "Resize(256) + CenterCrop(224)",
                "classes": ANIMALS10_CLASSES,
            },
        }

        return response

    def generate_gradcam(
        self,
        image_bytes: bytes,
        target_class_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates genuine Grad-CAM on ResNet-18 using target layer layer4[1].conv2.
        Extracts actual forward activations and backward gradients.
        """
        if not self.is_loaded:
            self.load_models()

        raw_pil = Image.open(io.BytesIO(image_bytes))
        if raw_pil.mode != "RGB":
            raw_pil = raw_pil.convert("RGB")

        # Resize display copy to standard 224x224 for crisp visual display
        display_pil = raw_pil.resize((224, 224), Image.Resampling.BILINEAR)

        # Preprocess input tensor with gradient tracking
        tensor = self.preprocess_image(raw_pil)
        tensor.requires_grad = True

        activations: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        # Target: last BasicBlock's second conv in ResNet-18's layer4
        target_layer = self.resnet_model.layer4[1].conv2
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            self.resnet_model.zero_grad()
            with torch.enable_grad():
                logits = self.resnet_model(tensor)
                probs = F.softmax(logits, dim=-1)[0]

                if (
                    target_class_idx is None
                    or not isinstance(target_class_idx, int)
                    or not (0 <= target_class_idx < len(ANIMALS10_CLASSES))
                ):
                    target_class_idx = int(logits.argmax(dim=-1).item())

                predicted_class = ANIMALS10_CLASSES[target_class_idx]
                confidence = round(float(probs[target_class_idx].item()) * 100, 2)

                score = logits[0, target_class_idx]
                score.backward()

            if not activations or not gradients:
                raise RuntimeError("Grad-CAM hooks failed to record activations or gradients.")

            act = activations[0].detach()   # Shape: [1, 512, 7, 7]
            grad = gradients[0].detach()    # Shape: [1, 512, 7, 7]

            # Global average pooling of gradients over spatial dimensions
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, 512, 1, 1]

            # Weighted linear combination of feature maps
            cam = torch.sum(weights * act, dim=1, keepdim=True)   # [1, 1, 7, 7]
            cam = F.relu(cam)

            # Upsample heatmap to match display size (224x224)
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam_np = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            c_min, c_max = cam_np.min(), cam_np.max()
            if c_max > c_min:
                cam_norm = (cam_np - c_min) / (c_max - c_min)
            else:
                cam_norm = np.zeros_like(cam_np)

            # Generate colored heatmap using colormap 'jet'
            cmap = matplotlib.colormaps["jet"]
            colored_heatmap = cmap(cam_norm)[:, :, :3]   # RGB in [0, 1]
            colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)
            heatmap_pil = Image.fromarray(colored_heatmap_uint8)

            # Blend with original image (45% original, 55% heatmap)
            overlay_pil = Image.blend(display_pil, heatmap_pil, alpha=0.55)

            # Convert images to base64 data URLs
            original_b64 = pil_to_base64_data_url(display_pil)
            heatmap_b64 = pil_to_base64_data_url(heatmap_pil)
            overlay_b64 = pil_to_base64_data_url(overlay_pil)

            return {
                "success": True,
                "target_class": predicted_class,
                "target_class_idx": target_class_idx,
                "confidence": confidence,
                "layer_name": "layer4[1].conv2 (Final ResNet-18 BasicBlock)",
                "original_image": original_b64,
                "heatmap_image": heatmap_b64,
                "overlay_image": overlay_b64,
                "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
                "model": "ResNet-18 (10-class Animals-10)",
            }
        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    def test_robustness(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs robustness perturbations on the uploaded image for 10-class models:
        1. Original image (baseline)
        2. Rotation (+15°)
        3. Brightness adjustment (+35%)
        4. Gaussian Blur (radius=2.0)
        """
        if not self.is_loaded:
            self.load_models()

        raw_pil = Image.open(io.BytesIO(image_bytes))
        if raw_pil.mode != "RGB":
            raw_pil = raw_pil.convert("RGB")

        variations = [
            {
                "id": "original",
                "name": "Original Image",
                "description": "Unperturbed input image as baseline reference.",
                "image": raw_pil,
            },
            {
                "id": "rotation",
                "name": "Rotation (+15°)",
                "description": "Affine rotation by 15° with bilinear interpolation.",
                "image": raw_pil.rotate(15, expand=False, resample=Image.Resampling.BILINEAR),
            },
            {
                "id": "brightness",
                "name": "Brightness (+35%)",
                "description": "Photometric illumination shift (brightness factor 1.35).",
                "image": ImageEnhance.Brightness(raw_pil).enhance(1.35),
            },
            {
                "id": "blur",
                "name": "Gaussian Blur (r=2.0)",
                "description": "High-frequency attenuation via Gaussian kernel (radius 2.0).",
                "image": raw_pil.filter(ImageFilter.GaussianBlur(radius=2.0)),
            },
        ]

        def get_top3(probs: np.ndarray) -> List[Dict[str, Any]]:
            indices = np.argsort(probs)[::-1][:3]
            return [
                {
                    "class_name": ANIMALS10_CLASSES[idx],
                    "probability": round(float(probs[idx]) * 100, 2),
                }
                for idx in indices
            ]

        results = []
        for var in variations:
            img = var["image"]
            tensor = self.preprocess_image(img)

            resnet_probs, resnet_lat = self._infer_single(self.resnet_model, tensor)
            swin_probs, swin_lat = self._infer_single(self.swin_model, tensor)

            resnet_idx = int(resnet_probs.argmax())
            swin_idx = int(swin_probs.argmax())

            resnet_pred = ANIMALS10_CLASSES[resnet_idx]
            swin_pred = ANIMALS10_CLASSES[swin_idx]

            resnet_conf = round(float(resnet_probs[resnet_idx]) * 100, 2)
            swin_conf = round(float(swin_probs[swin_idx]) * 100, 2)

            preview_img = img.copy()
            preview_img.thumbnail((160, 160))
            preview_b64 = pil_to_base64_data_url(preview_img)

            results.append({
                "id": var["id"],
                "name": var["name"],
                "description": var["description"],
                "image_preview": preview_b64,
                "agreement": (resnet_pred == swin_pred),
                "cnn": {
                    "predicted_class": resnet_pred,
                    "confidence": resnet_conf,
                    "top3": get_top3(resnet_probs),
                    "latency_ms": resnet_lat,
                },
                "swin": {
                    "predicted_class": swin_pred,
                    "confidence": swin_conf,
                    "top3": get_top3(swin_probs),
                    "latency_ms": swin_lat,
                },
            })

        # Summary of stability
        orig_resnet = results[0]["cnn"]["predicted_class"]
        orig_swin = results[0]["swin"]["predicted_class"]

        resnet_stable_count = sum(
            1 for r in results if r["cnn"]["predicted_class"] == orig_resnet
        )
        swin_stable_count = sum(
            1 for r in results if r["swin"]["predicted_class"] == orig_swin
        )

        return {
            "dataset": "Animals-10",
            "num_classes": 10,
            "variations": results,
            "stability_summary": {
                "cnn_stability_rate": round((resnet_stable_count / len(results)) * 100, 1),
                "swin_stability_rate": round((swin_stable_count / len(results)) * 100, 1),
                "cnn_baseline_class": orig_resnet,
                "swin_baseline_class": orig_swin,
            },
        }


# Global Animals-10 model manager singleton
animals10_model_manager = Animals10ModelManager()


# =============================================================================
# Animals-10 Helper Functions (Models Info, Training History, Evaluation Metrics)
# =============================================================================

def get_animals10_models_info() -> Dict[str, Any]:
    """Returns architecture specs, parameters, and training config for 10-class models."""
    resnet_cfg_path = os.path.join(ANIMALS10_RESULTS_DIR, "resnet18_config.json")
    swin_cfg_path = os.path.join(ANIMALS10_RESULTS_DIR, "swin_tiny_config.json")

    resnet_cfg: Dict[str, Any] = {}
    swin_cfg: Dict[str, Any] = {}

    if os.path.exists(resnet_cfg_path):
        with open(resnet_cfg_path, "r", encoding="utf-8") as f:
            resnet_cfg = json.load(f)
    if os.path.exists(swin_cfg_path):
        with open(swin_cfg_path, "r", encoding="utf-8") as f:
            swin_cfg = json.load(f)

    return {
        "dataset": "Rapidata/Animals-10",
        "classes": ANIMALS10_CLASSES,
        "num_classes": 10,
        "device": "CPU",
        "resolution": "224x224",
        "preprocessing": "Resize(256) + CenterCrop(224) + ImageNet Normalization",
        "cnn": {
            "name": "ResNet-18 (ImageNet Pretrained)",
            "type": "Residual Convolutional Network (torchvision.models.resnet18)",
            "total_params": resnet_cfg.get("total_parameters", 11181642),
            "trainable_params": resnet_cfg.get("trainable_parameters", 5130),
            "frozen_params": resnet_cfg.get("frozen_parameters", 11176512),
            "best_val_accuracy": resnet_cfg.get("best_val_accuracy"),
            "best_epoch": resnet_cfg.get("best_epoch"),
            "epochs_completed": resnet_cfg.get("epochs", 10),
            "batch_size": resnet_cfg.get("batch_size", 64),
            "learning_rate": resnet_cfg.get("learning_rate", 0.001),
            "weight_decay": resnet_cfg.get("weight_decay", 0.0001),
            "backbone_status": "Frozen ImageNet-1K Pretrained Backbone, Linear Head Tuned",
            "training_protocol": resnet_cfg.get("training_protocol", ""),
        },
        "swin": {
            "name": "Swin-Tiny Transformer",
            "type": "Hierarchical Vision Transformer (torchvision.models.swin_t)",
            "total_params": swin_cfg.get("total_parameters", 27527044),
            "trainable_params": swin_cfg.get("trainable_parameters", 7690),
            "frozen_params": swin_cfg.get("frozen_parameters", 27519354),
            "best_val_accuracy": swin_cfg.get("best_val_accuracy"),
            "best_epoch": swin_cfg.get("best_epoch"),
            "epochs_completed": swin_cfg.get("epochs", 10),
            "batch_size": swin_cfg.get("batch_size", 64),
            "learning_rate": swin_cfg.get("learning_rate", 0.001),
            "weight_decay": swin_cfg.get("weight_decay", 0.0001),
            "backbone_status": "Frozen ImageNet-1K Pretrained Backbone, Linear Head Tuned",
            "training_protocol": swin_cfg.get("training_protocol", ""),
        },
        "scientific_notice": (
            "Both models use frozen ImageNet-1K pretrained backbones with only the linear classification "
            "head trained on the Animals-10 dataset (16,487 training images across 10 classes). "
            "Evaluation uses the held-out test set (3,534 images) with deterministic "
            "Resize(256) + CenterCrop(224) preprocessing."
        ),
    }


def get_animals10_training_history() -> Dict[str, Any]:
    """Parses resnet18_history.csv and swin_tiny_history.csv from results/animals10/."""
    resnet_csv_path = os.path.join(ANIMALS10_RESULTS_DIR, "resnet18_history.csv")
    swin_csv_path = os.path.join(ANIMALS10_RESULTS_DIR, "swin_tiny_history.csv")

    def read_csv(path: str) -> List[Dict[str, Any]]:
        rows = []
        if os.path.exists(path):
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    row = {
                        "epoch": int(r["epoch"]),
                        "train_loss": round(float(r["train_loss"]), 4),
                        "train_acc": round(float(r["train_acc"]), 2),
                        "val_loss": round(float(r["val_loss"]), 4),
                        "val_acc": round(float(r["val_acc"]), 2),
                        "duration_seconds": round(float(r.get("duration_seconds", 0)), 1),
                    }
                    # Animals-10 history CSVs also include val_macro_f1
                    if "val_macro_f1" in r:
                        row["val_macro_f1"] = round(float(r["val_macro_f1"]), 2)
                    rows.append(row)
        return rows

    return {
        "dataset": "Animals-10",
        "cnn": read_csv(resnet_csv_path),
        "swin": read_csv(swin_csv_path),
        "classes": ANIMALS10_CLASSES,
    }


def get_animals10_evaluation_metrics() -> Dict[str, Any]:
    """
    Loads official held-out test-set benchmark results from
    results/animals10/test_metrics_comparison.csv and
    results/animals10/test_classification_reports.json.
    """
    csv_path = os.path.join(ANIMALS10_RESULTS_DIR, "test_metrics_comparison.csv")
    json_path = os.path.join(ANIMALS10_RESULTS_DIR, "test_classification_reports.json")

    comparison_rows: List[Dict[str, Any]] = []
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                comparison_rows.append({
                    "model": r["model"],
                    "training_setup": r["training_setup"],
                    "total_params": int(r["total_params"]),
                    "trainable_params_training": int(r["trainable_params_training"]),
                    "frozen_params_training": int(r["frozen_params_training"]),
                    "test_accuracy": float(r["test_accuracy"]),
                    "macro_precision": float(r["macro_precision"]),
                    "macro_recall": float(r["macro_recall"]),
                    "macro_f1": float(r["macro_f1"]),
                    "weighted_f1": float(r["weighted_f1"]),
                    "mean_latency_ms": float(r["mean_latency_ms"]),
                    "median_latency_ms": float(r["median_latency_ms"]),
                    "p95_latency_ms": float(r["p95_latency_ms"]),
                    "throughput_fps": float(r["throughput_fps"]),
                    "eval_duration_seconds": float(r.get("batch_eval_duration_seconds", r.get("eval_duration_seconds", 0))),
                })

    reports_data: Dict[str, Any] = {}
    if os.path.exists(json_path):
        with open(json_path, mode="r", encoding="utf-8") as f:
            reports_data = json.load(f)

    return {
        "available": bool(comparison_rows),
        "dataset": "Animals-10",
        "comparison_table": comparison_rows,
        "classification_reports": reports_data,
        "classes": ANIMALS10_CLASSES,
        "test_dataset_size": 3534,
        "data_efficiency": get_animals10_data_efficiency(),
    }


def get_animals10_data_efficiency() -> Dict[str, Any]:
    """Loads Phase A Data Efficiency benchmark metrics, comparison curves, and protocol metadata."""
    if not os.path.exists(ANIMALS10_DATA_EFFICIENCY_CSV):
        return {"available": False}

    rows = []
    with open(ANIMALS10_DATA_EFFICIENCY_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "model": r["model"],
                "model_key": r["model_key"],
                "fraction": r["fraction"],
                "train_samples": int(r["train_samples"]),
                "val_samples": int(r["val_samples"]),
                "test_samples": int(r["test_samples"]),
                "best_epoch": int(r["best_epoch"]),
                "best_val_acc": float(r["best_val_acc"]),
                "best_val_loss": float(r["best_val_loss"]),
                "test_accuracy": float(r["test_accuracy"]),
                "macro_precision": float(r["macro_precision"]),
                "macro_recall": float(r["macro_recall"]),
                "macro_f1": float(r["macro_f1"]),
                "weighted_precision": float(r["weighted_precision"]),
                "weighted_recall": float(r["weighted_recall"]),
                "weighted_f1": float(r["weighted_f1"]),
                "train_duration_seconds": float(r["train_duration_seconds"]),
                "checkpoint_file": r["checkpoint_file"],
            })

    # Group by fraction for head-to-head comparison
    fractions = ["10%", "25%", "50%", "75%", "100%"]
    comparison = []
    for frac in fractions:
        r18 = next((r for r in rows if r["model_key"] == "resnet18" and r["fraction"] == frac), None)
        swin = next((r for r in rows if r["model_key"] == "swin_tiny" and r["fraction"] == frac), None)
        if r18 and swin:
            acc_delta = round(swin["test_accuracy"] - r18["test_accuracy"], 2)
            f1_delta = round(swin["macro_f1"] - r18["macro_f1"], 2)
            comparison.append({
                "fraction": frac,
                "fraction_pct": int(frac.replace("%", "")),
                "train_samples": r18["train_samples"],
                "resnet_test_acc": r18["test_accuracy"],
                "swin_test_acc": swin["test_accuracy"],
                "acc_delta": acc_delta,
                "winner_acc": "Swin-Tiny" if acc_delta > 0 else "ResNet-18",
                "resnet_macro_f1": r18["macro_f1"],
                "swin_macro_f1": swin["macro_f1"],
                "f1_delta": f1_delta,
                "winner_f1": "Swin-Tiny" if f1_delta > 0 else "ResNet-18",
                "resnet_val_acc": r18["best_val_acc"],
                "swin_val_acc": swin["best_val_acc"],
                "resnet_train_time": r18["train_duration_seconds"],
                "swin_train_time": swin["train_duration_seconds"],
            })

    return {
        "available": True,
        "dataset": "Animals-10",
        "fractions": fractions,
        "metrics": rows,
        "comparison_by_fraction": comparison,
        "has_plot_image": os.path.exists(ANIMALS10_DATA_EFFICIENCY_PLOT),
        "protocol": {
            "strategy": "Frozen ImageNet-1K Pretrained Backbone + Tuned Linear Classification Head",
            "seed": 42,
            "nested_stratified_subsets": "10% ⊂ 25% ⊂ 50% ⊂ 75% ⊂ 100% (Identical image indices across models)",
            "test_split_size": 3534,
            "val_split_size": 3533,
            "full_train_size": 16487,
        },
    }


def load_animals10_sample_images() -> List[Dict[str, Any]]:
    """
    Loads representative sample images for all 10 Animals-10 classes.
    For 'chicken', selects a clear, representative sample from the validation split
    (per project requirements, avoiding data leakage from the held-out test split).
    Returns:
      - data_url: Full native-resolution base64 JPEG for model inference.
      - thumbnail_url: Lightweight 128x128-bounded base64 thumbnail for UI preview.
      - original_dimensions: [width, height] of the full native image.
      - thumbnail_dimensions: [width, height] of the UI preview thumbnail.
    """
    samples: List[Dict[str, Any]] = []
    test_dir = os.path.join(ANIMALS10_DATA_DIR, "test")
    val_dir = os.path.join(ANIMALS10_DATA_DIR, "val")

    for idx, class_name in enumerate(ANIMALS10_CLASSES):
        # Chicken sample is loaded from validation split; other classes from test split
        if class_name == "chicken" and os.path.isdir(os.path.join(val_dir, class_name)):
            class_dir = os.path.join(val_dir, class_name)
            split_source = "val"
        elif os.path.isdir(os.path.join(test_dir, class_name)):
            class_dir = os.path.join(test_dir, class_name)
            split_source = "test"
        elif os.path.isdir(os.path.join(val_dir, class_name)):
            class_dir = os.path.join(val_dir, class_name)
            split_source = "val"
        else:
            continue

        # Pick the first valid image file (sorted for determinism)
        for fname in sorted(os.listdir(class_dir)):
            fpath = os.path.join(class_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                try:
                    # 1. Read original full image at native resolution (NO resizing/distortion)
                    with open(fpath, "rb") as f:
                        raw_bytes = f.read()

                    raw_pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    orig_w, orig_h = raw_pil.size

                    # Base64 data_url for original uncompressed/unresized image (for model inference)
                    orig_buf = io.BytesIO()
                    raw_pil.save(orig_buf, format="JPEG", quality=95)
                    orig_b64 = base64.b64encode(orig_buf.getvalue()).decode("utf-8")

                    # 2. Generate separate thumbnail for UI preview display ONLY
                    thumb_pil = raw_pil.copy()
                    thumb_pil.thumbnail((128, 128), Image.Resampling.LANCZOS)
                    thumb_w, thumb_h = thumb_pil.size

                    thumb_buf = io.BytesIO()
                    thumb_pil.save(thumb_buf, format="JPEG", quality=85)
                    thumb_b64 = base64.b64encode(thumb_buf.getvalue()).decode("utf-8")

                    samples.append({
                        "class_name": class_name,
                        "target_idx": idx,
                        "filename": fname,
                        "split": split_source,
                        "data_url": f"data:image/jpeg;base64,{orig_b64}",
                        "thumbnail_url": f"data:image/jpeg;base64,{thumb_b64}",
                        "original_dimensions": [orig_w, orig_h],
                        "thumbnail_dimensions": [thumb_w, thumb_h],
                    })
                except Exception as ex:
                    print(f"[Warning] Failed to load sample {class_name}/{fname}: {ex}")
                break  # Only one sample per class

    return samples
