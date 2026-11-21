"""
Verification script for all 8 test cases required by the user prompt.
"""
import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.inference import animals10_model_manager

def test_cases():
    print("[Verification] Loading models if needed...")
    manager = animals10_model_manager
    if not manager.is_loaded:
        manager.load_models()

    cases = [
        ("1. Cat (Supported)", "data/animals-10/splits/val/cat/cat_01604.jpg"),
        ("2. Elephant (Supported)", "data/animals-10/splits/val/elephant/elephant_12405.jpg"),
        ("3. Lion (Unsupported Animal)", "data/validation_ood/unsupported_animals/lion.jpg"),
        ("4. Tiger (Unsupported Animal)", "data/validation_ood/unsupported_animals/tiger.jpg"),
        ("5. Monkey (Unsupported Animal)", "data/validation_ood/unsupported_animals/monkey.jpg"),
        ("6. Car (Non-Animal Object)", "data/validation_ood/non_animals/car.jpg"),
        ("7. Mobile (Non-Animal Object)", "data/validation_ood/non_animals/mobile.jpg"),
        ("8. Ambiguous (Blur / Noise)", "data/validation_ood/ambiguous/extreme_blur.jpg"),
    ]

    print("\n" + "="*95)
    print(f"{'TEST CASE':<32} | {'STATUS':<30} | {'SUPPORTED':<9} | {'CNN PREDICTION':<18} | {'SWIN PREDICTION'}")
    print("="*95)

    all_passed = True
    for name, path in cases:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            all_passed = False
            continue

        with open(path, "rb") as f:
            res = manager.predict(f.read(), log_history=False)

        ood = res["ood_detection"]
        status = ood["status"]
        is_supported = ood["is_supported"]
        cnn_pred = f"{res['cnn']['predicted_class']} ({res['cnn']['confidence']}%)"
        swin_pred = f"{res['swin']['predicted_class']} ({res['swin']['confidence']}%)"

        print(f"{name:<32} | {status:<30} | {str(is_supported):<9} | {cnn_pred:<18} | {swin_pred}")

    print("="*95)
    print("[Verification Complete]")

if __name__ == "__main__":
    test_cases()
