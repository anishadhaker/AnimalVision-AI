"""
Integration test for FastAPI endpoint functions in backend.app:
1. animals10_health()
2. animals10_models_info()
3. animals10_evaluation_metrics()
4. animals10_predict_image()
"""

import os
import io
import sys
from fastapi import UploadFile
from starlette.datastructures import Headers

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import backend.app as ba
from backend.inference import animals10_model_manager

def test_endpoints():
    print("[API Test] Initializing backend and verifying endpoint handlers...")
    if not animals10_model_manager.is_loaded:
        animals10_model_manager.load_models()

    # 1. Health
    health = ba.animals10_health()
    assert health["status"] == "healthy"
    assert health["dataset"] == "Animals-10"
    assert health["num_classes"] == 10
    print("[PASS] animals10_health() -> Status: healthy, models_loaded:", health["models_loaded"])

    # 2. Models info
    info = ba.animals10_models_info()
    assert "cnn" in info and "swin" in info
    print("[PASS] animals10_models_info() -> ResNet-18:", info["cnn"]["name"], "| Swin-Tiny:", info["swin"]["name"])

    # 3. Evaluation metrics
    metrics = ba.animals10_evaluation_metrics()
    assert "comparison_table" in metrics
    print("[PASS] animals10_evaluation_metrics() -> Comparison rows:", len(metrics.get("comparison_table", [])), "| Evaluated classes:", len(metrics.get("classes", [])))

    # 4. Predict image
    cat_path = os.path.join(BASE_DIR, "data", "animals-10", "splits", "val", "cat", "cat_01604.jpg")
    with open(cat_path, "rb") as f:
        file_bytes = f.read()

    upload = UploadFile(filename="cat_test.jpg", file=io.BytesIO(file_bytes), headers=Headers({"content-type": "image/jpeg"}))
    pred = ba.animals10_predict_image(upload)

    assert pred["cnn"]["predicted_class"] == "cat"
    assert pred["swin"]["predicted_class"] == "cat"
    assert pred["ood_detection"]["is_supported"] is True
    print(f"[PASS] animals10_predict_image() -> ResNet: {pred['cnn']['predicted_class']} ({pred['cnn']['confidence']}%) | Swin: {pred['swin']['predicted_class']} ({pred['swin']['confidence']}%) | Diff: {pred['confidence_margin']} pp | OOD: {pred['ood_detection']['status']}")

    print("\nALL FASTAPI ENDPOINTS VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    test_endpoints()
