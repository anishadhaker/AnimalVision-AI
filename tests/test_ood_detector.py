"""
Validation script for OOD Detector in AnimalVision AI.
Tests accuracy, Unknown Detection Rate, False Rejection Rate (FRR), and False Acceptance Rate (FAR).
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.inference import animals10_model_manager, ANIMALS10_CLASSES
from backend.ood_detector import OODDetector

def run_suite():
    print("[OOD Validator] Ensuring 10-class models are loaded...")
    if not animals10_model_manager.is_loaded:
        animals10_model_manager.load_models()

    manager = animals10_model_manager

    def evaluate_img(img_path):
        raw_pil = Image.open(img_path).convert("RGB")
        tensor = manager.preprocess_image(raw_pil)
        
        with torch.no_grad():
            r_logits = manager.resnet_model(tensor)[0].cpu().numpy()
            s_logits = manager.swin_model(tensor)[0].cpu().numpy()

        r_probs = F.softmax(torch.tensor(r_logits), dim=-1).numpy()
        s_probs = F.softmax(torch.tensor(s_logits), dim=-1).numpy()

        r_pred = ANIMALS10_CLASSES[int(r_probs.argmax())]
        s_pred = ANIMALS10_CLASSES[int(s_probs.argmax())]

        verdict = OODDetector.evaluate(
            resnet_logits=r_logits,
            swin_logits=s_logits,
            resnet_probs=r_probs,
            swin_probs=s_probs,
            resnet_pred_class=r_pred,
            swin_pred_class=s_pred,
        )
        return verdict, r_pred, s_pred

    print("\n" + "="*80)
    print("1. EVALUATING SUPPORTED ANIMALS (Validation Set - In-Distribution)")
    print("="*80)
    val_dir = os.path.join(BASE_DIR, "data", "animals-10", "splits", "val")
    supported_results = []
    for cls_name in ANIMALS10_CLASSES:
        cls_dir = os.path.join(val_dir, cls_name)
        if os.path.exists(cls_dir):
            files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                p = os.path.join(cls_dir, sorted(files)[0])
                v, r_pred, s_pred = evaluate_img(p)
                supported_results.append({
                    "class": cls_name,
                    "status": v["status"],
                    "is_supported": v["is_supported"],
                    "resnet_pred": r_pred,
                    "swin_pred": s_pred,
                    "swin_energy": v["signals"]["swin_energy"],
                    "avg_energy": v["signals"]["energy_score"]
                })
                print(f"[{cls_name:10s}] -> Status: {v['status']:25s} | Supported={v['is_supported']} | SwinE={v['signals']['swin_energy']:5.2f} | AvgE={v['signals']['energy_score']:5.2f}")

    print("\n" + "="*80)
    print("2. EVALUATING UNSUPPORTED ANIMALS (OOD)")
    print("="*80)
    unsupported_dir = os.path.join(BASE_DIR, "data", "validation_ood", "unsupported_animals")
    unsupported_results = []
    for f in sorted(os.listdir(unsupported_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(unsupported_dir, f)
            v, r_pred, s_pred = evaluate_img(p)
            unsupported_results.append({
                "file": f,
                "status": v["status"],
                "is_supported": v["is_supported"],
                "r_pred": r_pred,
                "s_pred": s_pred,
                "swin_energy": v["signals"]["swin_energy"],
                "avg_energy": v["signals"]["energy_score"]
            })
            print(f"[{f:15s}] -> Status: {v['status']:28s} | Supported={v['is_supported']} | R={r_pred:10s} | S={s_pred:10s}")

    print("\n" + "="*80)
    print("3. EVALUATING NON-ANIMAL OBJECTS (OOD)")
    print("="*80)
    non_animal_dir = os.path.join(BASE_DIR, "data", "validation_ood", "non_animals")
    non_animal_results = []
    for f in sorted(os.listdir(non_animal_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(non_animal_dir, f)
            v, r_pred, s_pred = evaluate_img(p)
            non_animal_results.append({
                "file": f,
                "status": v["status"],
                "is_supported": v["is_supported"],
                "r_pred": r_pred,
                "s_pred": s_pred,
                "swin_energy": v["signals"]["swin_energy"],
                "avg_energy": v["signals"]["energy_score"]
            })
            print(f"[{f:15s}] -> Status: {v['status']:28s} | Supported={v['is_supported']} | R={r_pred:10s} | S={s_pred:10s}")

    print("\n" + "="*80)
    print("4. EVALUATING AMBIGUOUS / LOW QUALITY / NOISE (OOD)")
    print("="*80)
    ambiguous_dir = os.path.join(BASE_DIR, "data", "validation_ood", "ambiguous")
    ambiguous_results = []
    for f in sorted(os.listdir(ambiguous_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            p = os.path.join(ambiguous_dir, f)
            v, r_pred, s_pred = evaluate_img(p)
            ambiguous_results.append({
                "file": f,
                "status": v["status"],
                "is_supported": v["is_supported"],
                "r_pred": r_pred,
                "s_pred": s_pred,
                "swin_energy": v["signals"]["swin_energy"],
                "avg_energy": v["signals"]["energy_score"]
            })
            print(f"[{f:22s}] -> Status: {v['status']:25s} | Supported={v['is_supported']} | R={r_pred:10s} | S={s_pred:10s}")

    # Summary statistics
    total_supported = len(supported_results)
    correct_supported = sum(1 for r in supported_results if r["is_supported"])
    frr = (total_supported - correct_supported) / total_supported * 100.0

    all_ood = unsupported_results + non_animal_results + ambiguous_results
    total_ood = len(all_ood)
    detected_ood = sum(1 for r in all_ood if not r["is_supported"])
    detection_rate = detected_ood / total_ood * 100.0
    far = (total_ood - detected_ood) / total_ood * 100.0

    print("\n" + "="*80)
    print("SUMMARY PERFORMANCE METRICS")
    print("="*80)
    print(f"Total In-Distribution (Supported) Samples: {total_supported}")
    print(f"Correctly Accepted as Supported:          {correct_supported}/{total_supported} ({100 - frr:.1f}%)")
    print(f"False Rejection Rate (FRR):               {frr:.1f}%")
    print("-" * 50)
    print(f"Total Out-of-Distribution (OOD) Samples:  {total_ood}")
    print(f"Correctly Detected as Unsupported/OOD:    {detected_ood}/{total_ood} ({detection_rate:.1f}%)")
    print(f"False Acceptance Rate (FAR):              {far:.1f}%")
    print("="*80)

if __name__ == "__main__":
    run_suite()
