import io
import sys
import os
from PIL import Image
from fastapi import UploadFile

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath("."))

import backend.app as ba
from backend.inference import animals10_model_manager

results = {}

# 1. Check /api/animals10/health
try:
    if hasattr(ba, "startup_event"):
        ba.startup_event()
    health = ba.animals10_health()
    if health.get("status") == "healthy" and health.get("models_loaded") is True:
        results["check_1_health"] = ("PASS", f"Status: {health['status']}, models_loaded: True")
    else:
        results["check_1_health"] = ("FAIL", f"Unexpected health response: {health}")
except Exception as e:
    results["check_1_health"] = ("FAIL", str(e))

# 2. Check that both history endpoints return 404 (routes completely absent from app)
try:
    routes = [r.path for r in ba.app.routes]
    has_a10_hist = "/api/animals10/prediction-history" in routes
    has_leg_hist = "/api/prediction-history" in routes
    if not has_a10_hist and not has_leg_hist:
        results["check_2_history_404"] = ("PASS", "Both /api/animals10/prediction-history and /api/prediction-history absent (404 Not Found)")
    else:
        results["check_2_history_404"] = ("FAIL", f"History routes still present: a10={has_a10_hist}, legacy={has_leg_hist}")
except Exception as e:
    results["check_2_history_404"] = ("FAIL", str(e))

# 3 & 4. Perform ONLY ONE real image prediction on /api/animals10/predict and verify both models
try:
    # Use real test image if available or clean RGB tensor
    test_img_path = os.path.join("data", "animals-10", "splits", "test", "butterfly")
    sample_img = None
    if os.path.exists(test_img_path):
        for f in os.listdir(test_img_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_img = Image.open(os.path.join(test_img_path, f)).convert("RGB")
                break
    if sample_img is None:
        sample_img = Image.new("RGB", (224, 224), color=(100, 150, 200))

    buf = io.BytesIO()
    sample_img.save(buf, format="JPEG")
    buf.seek(0)

    upload = UploadFile(filename="smoke_test.jpg", file=buf, headers={"content-type": "image/jpeg"})
    pred = ba.animals10_predict_image(upload)

    cnn = pred.get("cnn", {})
    swin = pred.get("swin", {})

    cnn_ok = cnn.get("predicted_class") is not None and cnn.get("confidence", 0) > 0
    swin_ok = swin.get("predicted_class") is not None and swin.get("confidence", 0) > 0

    if cnn_ok and swin_ok:
        results["check_3_one_prediction"] = ("PASS", f"Successfully executed 1 image prediction (agreement={pred.get('agreement')})")
        results["check_4_model_outputs"] = ("PASS", f"ResNet-18: '{cnn['predicted_class']}' ({cnn['confidence']}%), Swin-Tiny: '{swin['predicted_class']}' ({swin['confidence']}%)")
    else:
        results["check_3_one_prediction"] = ("FAIL", f"Invalid response: cnn_ok={cnn_ok}, swin_ok={swin_ok}")
        results["check_4_model_outputs"] = ("FAIL", f"cnn: {cnn}, swin: {swin}")
except Exception as e:
    results["check_3_one_prediction"] = ("FAIL", str(e))
    results["check_4_model_outputs"] = ("FAIL", str(e))

print("\n--- SMOKE TEST RESULTS ---")
for k, (status, detail) in results.items():
    print(f"[{status}] {k}: {detail}")
