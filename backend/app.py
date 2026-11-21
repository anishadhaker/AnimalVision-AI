"""
FastAPI Application for Comparative Study: CNN vs Swin Transformer.

Endpoints (CIFAR-10, 5 Classes):
- GET  /api/health                 -> Health check and model readiness
- GET  /api/models-info            -> Parameter counts, architectures, training setups
- GET  /api/history                -> Training history curves (loss and accuracy)
- GET  /api/sample-images          -> 5 real CIFAR-10 animal test images for 1-click test
- POST /api/predict                -> Side-by-side inference (CNN v2 vs Swin-Tiny, top-3, consensus)
- POST /api/gradcam                -> Genuine Grad-CAM for CNN v2 (stage4.conv2)
- GET  /api/swin-attention-status  -> Authentic Swin attention availability disclosure
- POST /api/robustness-test        -> 4 image perturbations (Original, 15° Rotation, Brightness +35%, Gaussian Blur)
- GET  /api/evaluation-metrics     -> Official CIFAR-10 test set metrics, reports, confusion matrices
- GET  /api/confusion-matrix-plot  -> Serves the generated confusion matrix image file

Endpoints (Animals-10, 10 Classes):
- GET  /api/animals10/health                 -> Health check and model readiness for 10-class models
- GET  /api/animals10/models-info            -> Parameter counts, architectures, training setups (ResNet-18 + Swin-Tiny)
- GET  /api/animals10/history                -> Training history curves (Animals-10)
- GET  /api/animals10/sample-images          -> 10 real Animals-10 test images for 1-click test
- POST /api/animals10/predict                -> Side-by-side inference (ResNet-18 vs Swin-Tiny, 10 classes)
- POST /api/animals10/gradcam                -> Genuine Grad-CAM for ResNet-18 (layer4[1].conv2)
- GET  /api/animals10/swin-attention-status  -> Authentic Swin attention availability disclosure
- POST /api/animals10/robustness-test        -> 4 image perturbations on 10-class models
- GET  /api/animals10/evaluation-metrics     -> Held-out test set metrics, reports, confusion matrices (Animals-10)
- GET  /api/animals10/confusion-matrix-plot  -> Serves the Animals-10 confusion matrix image file
"""

import os
import io
import base64
from typing import Dict, Any, List, Optional
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.inference import (
    model_manager,
    get_models_info,
    get_training_history,
    get_evaluation_metrics,
    get_swin_attention_status,
    ANIMAL_CLASSES,
    animals10_model_manager,
    get_animals10_models_info,
    get_animals10_training_history,
    get_animals10_evaluation_metrics,
    get_animals10_data_efficiency,
    load_animals10_sample_images,
    ANIMALS10_CLASSES,
)
from src.dataset import filter_animal_subset
import torchvision

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "test_confusion_matrices.png")
ANIMALS10_RESULTS_DIR = os.path.join(RESULTS_DIR, "animals10")
ANIMALS10_CONFUSION_MATRIX_PATH = os.path.join(ANIMALS10_RESULTS_DIR, "test_confusion_matrices.png")
ANIMALS10_DATA_EFFICIENCY_PLOT_PATH = os.path.join(ANIMALS10_RESULTS_DIR, "data_efficiency_comparison.png")

app = FastAPI(
    title="AnimalVision AI API (Animals-10)",
    description="Comparative AI/ML API benchmarking ResNet-18 Residual CNN and Swin-Tiny Vision Transformer on the 10-class Animals-10 dataset.",
    version="2.1.0",
)

# Enable CORS for React frontend (Vite default is 5173, allow all local origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for sample images
_CACHED_SAMPLES: List[Dict[str, Any]] = []
_CACHED_ANIMALS10_SAMPLES: List[Dict[str, Any]] = []


def _load_animals10_sample_images() -> List[Dict[str, Any]]:
    global _CACHED_ANIMALS10_SAMPLES
    if _CACHED_ANIMALS10_SAMPLES:
        return _CACHED_ANIMALS10_SAMPLES

    try:
        _CACHED_ANIMALS10_SAMPLES = load_animals10_sample_images()
    except Exception as err:
        print(f"[Warning] Could not preload Animals-10 sample images: {err}")
        _CACHED_ANIMALS10_SAMPLES = []

    return _CACHED_ANIMALS10_SAMPLES


def _load_sample_images() -> List[Dict[str, Any]]:
    global _CACHED_SAMPLES
    if _CACHED_SAMPLES:
        return _CACHED_SAMPLES

    try:
        data_cifar_dir = os.path.join(BASE_DIR, "data", "cifar-10-batches-py")
        need_download = not os.path.exists(data_cifar_dir)
        raw_test = torchvision.datasets.CIFAR10(
            root=os.path.join(BASE_DIR, "data"),
            train=False,
            download=need_download,
        )
        images, targets = filter_animal_subset(raw_test)
        samples = []
        found_classes = set()

        for idx in range(len(images)):
            target = targets[idx]
            if target not in found_classes:
                raw_img_arr = images[idx]
                pil_img = Image.fromarray(raw_img_arr)
                buffered = io.BytesIO()
                pil_img.save(buffered, format="JPEG", quality=90)
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                class_name = ANIMAL_CLASSES[target]
                samples.append({
                    "class_name": class_name,
                    "target_idx": target,
                    "data_url": f"data:image/jpeg;base64,{img_b64}",
                })
                found_classes.add(target)
                if len(found_classes) == 5:
                    break

        _CACHED_SAMPLES = sorted(samples, key=lambda s: s["target_idx"])
    except Exception as err:
        print(f"[Warning] Could not preload sample images: {err}")
        _CACHED_SAMPLES = []

    return _CACHED_SAMPLES


@app.on_event("startup")
def startup_event():
    """Load Animals-10 model checkpoints into memory once on startup."""
    try:
        animals10_model_manager.load_models()
        _load_animals10_sample_images()
        print("[Startup] Animals-10 models (ResNet-18 & Swin-Tiny) loaded successfully.")
    except Exception as err:
        print(f"[Startup Warning] Could not load 10-class models immediately: {err}")


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_manager.is_loaded,
        "device": str(model_manager.device),
        "target_resolution": f"{model_manager.img_size}x{model_manager.img_size}",
    }


@app.get("/api/models-info")
def models_info():
    return get_models_info()


@app.get("/api/history")
def history():
    return get_training_history()


@app.get("/api/sample-images")
def sample_images():
    samples = _load_sample_images()
    return {"samples": samples}


@app.post("/api/predict")
def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.content_type}'. Please upload an image file (PNG, JPG, WEBP).",
        )

    try:
        contents = file.file.read() if hasattr(file, "file") else file.read()
        results = model_manager.predict(contents)
        return results
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(err)}")


@app.post("/api/gradcam")
def gradcam_visualization(
    file: UploadFile = File(...),
    target_class: Optional[int] = Query(None, description="Optional target class index (0..4)"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.content_type}'. Please upload an image file.",
        )

    try:
        contents = file.file.read() if hasattr(file, "file") else file.read()
        target_idx = target_class if isinstance(target_class, int) else None
        gradcam_result = model_manager.generate_gradcam(contents, target_class_idx=target_idx)
        return gradcam_result
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation error: {str(err)}")


@app.get("/api/swin-attention-status")
def swin_attention_status():
    return get_swin_attention_status()


@app.post("/api/robustness-test")
def robustness_test(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.content_type}'. Please upload an image file.",
        )

    try:
        contents = file.file.read() if hasattr(file, "file") else file.read()
        results = model_manager.test_robustness(contents)
        return results
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Robustness testing error: {str(err)}")


@app.get("/api/evaluation-metrics")
def evaluation_metrics():
    return get_evaluation_metrics()


@app.get("/api/confusion-matrix-plot")
def confusion_matrix_plot():
    if os.path.exists(CONFUSION_MATRIX_PATH):
        return FileResponse(CONFUSION_MATRIX_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="Confusion matrix plot not found.")


# =============================================================================
# Animals-10 (10-Class) Endpoints
# =============================================================================

@app.get("/api/animals10/health")
def animals10_health():
    return {
        "status": "healthy",
        "dataset": "Animals-10",
        "num_classes": 10,
        "models_loaded": animals10_model_manager.is_loaded,
        "device": str(animals10_model_manager.device),
        "target_resolution": f"{animals10_model_manager.img_size}x{animals10_model_manager.img_size}",
    }


@app.get("/api/animals10/models-info")
def animals10_models_info():
    return get_animals10_models_info()


@app.get("/api/animals10/history")
def animals10_history():
    return get_animals10_training_history()


@app.get("/api/animals10/sample-images")
def animals10_sample_images():
    samples = _load_animals10_sample_images()
    return {"samples": samples, "dataset": "Animals-10", "num_classes": len(samples)}


@app.post("/api/animals10/predict")
def animals10_predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.content_type}'. Please upload an image file (PNG, JPG, WEBP).",
        )

    try:
        contents = file.file.read() if hasattr(file, "file") else file.read()
        results = animals10_model_manager.predict(contents)
        return results
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(err)}")


@app.post("/api/animals10/gradcam")
def animals10_gradcam_visualization(
    file: UploadFile = File(...),
    target_class: Optional[int] = Query(None, description="Optional target class index (0..9)"),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.content_type}'. Please upload an image file.",
        )

    try:
        contents = file.file.read() if hasattr(file, "file") else file.read()
        target_idx = target_class if isinstance(target_class, int) else None
        gradcam_result = animals10_model_manager.generate_gradcam(contents, target_class_idx=target_idx)
        return gradcam_result
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation error: {str(err)}")


@app.get("/api/animals10/swin-attention-status")
def animals10_swin_attention_status():
    return get_swin_attention_status()


@app.post("/api/animals10/robustness-test")
def animals10_robustness_test(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.content_type}'. Please upload an image file.",
        )

    try:
        contents = file.file.read() if hasattr(file, "file") else file.read()
        results = animals10_model_manager.test_robustness(contents)
        return results
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Robustness testing error: {str(err)}")


@app.get("/api/animals10/evaluation-metrics")
def animals10_evaluation_metrics():
    return get_animals10_evaluation_metrics()


@app.get("/api/animals10/confusion-matrix-plot")
def animals10_confusion_matrix_plot():
    if os.path.exists(ANIMALS10_CONFUSION_MATRIX_PATH):
        return FileResponse(ANIMALS10_CONFUSION_MATRIX_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="Animals-10 confusion matrix plot not found.")


@app.get("/api/animals10/data-efficiency")
def animals10_data_efficiency():
    return get_animals10_data_efficiency()


@app.get("/api/animals10/data-efficiency-plot")
def animals10_data_efficiency_plot():
    if os.path.exists(ANIMALS10_DATA_EFFICIENCY_PLOT_PATH):
        return FileResponse(ANIMALS10_DATA_EFFICIENCY_PLOT_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="Animals-10 data efficiency comparison plot not found.")


# Mount built frontend static files (Single-server deployment support)
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
