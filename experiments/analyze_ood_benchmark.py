import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
from backend.inference import animals10_model_manager, ANIMALS10_CLASSES, ANIMALS10_DATA_DIR

animals10_model_manager.load_models()

resnet = animals10_model_manager.resnet_model
swin = animals10_model_manager.swin_model
transform = animals10_model_manager.transform

def analyze_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        r_logits = resnet(tensor)[0]
        s_logits = swin(tensor)[0]
        
        r_probs = F.softmax(r_logits, dim=-1)
        s_probs = F.softmax(s_logits, dim=-1)
        
        r_energy = torch.logsumexp(r_logits, dim=-1).item()
        s_energy = torch.logsumexp(s_logits, dim=-1).item()
        avg_energy = (r_energy + s_energy) / 2.0
        
        r_pred_idx = int(r_logits.argmax().item())
        s_pred_idx = int(s_logits.argmax().item())
        
        r_conf = r_probs[r_pred_idx].item() * 100.0
        s_conf = s_probs[s_pred_idx].item() * 100.0
        min_conf = min(r_conf, s_conf)
        avg_conf = (r_conf + s_conf) / 2.0
        
        r_sorted, _ = torch.sort(r_logits, descending=True)
        s_sorted, _ = torch.sort(s_logits, descending=True)
        r_margin = (r_sorted[0] - r_sorted[1]).item()
        s_margin = (s_sorted[0] - s_sorted[1]).item()
        avg_margin = (r_margin + s_margin) / 2.0
        
        agreement = (r_pred_idx == s_pred_idx)
        
        r_entropy = -(r_probs * torch.log(r_probs + 1e-12)).sum().item()
        s_entropy = -(s_probs * torch.log(s_probs + 1e-12)).sum().item()
        avg_entropy = (r_entropy + s_entropy) / 2.0
        
    return {
        "r_pred": ANIMALS10_CLASSES[r_pred_idx],
        "r_conf": r_conf,
        "s_pred": ANIMALS10_CLASSES[s_pred_idx],
        "s_conf": s_conf,
        "agreement": agreement,
        "min_conf": min_conf,
        "avg_conf": avg_conf,
        "r_energy": r_energy,
        "s_energy": s_energy,
        "avg_energy": avg_energy,
        "r_margin": r_margin,
        "s_margin": s_margin,
        "avg_margin": avg_margin,
        "avg_entropy": avg_entropy,
    }

print("=" * 100)
print("COMPREHENSIVE OOD & SUPPORTED VALIDATION SUITE ANALYSIS")
print("=" * 100)

print("\n--- 1. SUPPORTED ANIMALS (Validation Split) ---")
supported_stats = []
for c in ANIMALS10_CLASSES:
    cdir = os.path.join(ANIMALS10_DATA_DIR, "val", c)
    fname = sorted(os.listdir(cdir))[0]
    res = analyze_image(os.path.join(cdir, fname))
    supported_stats.append(res)
    print(f"[{c:10s}]: Agree={str(res['agreement']):5s} | R={res['r_pred']} ({res['r_conf']:5.1f}%), S={res['s_pred']} ({res['s_conf']:5.1f}%) | Energy: R={res['r_energy']:5.2f}, S={res['s_energy']:5.2f}, Avg={res['avg_energy']:5.2f} | Margin={res['avg_margin']:5.2f}")

print("\n--- 2. UNSUPPORTED ANIMALS ---")
unsupported_stats = []
unsupported_dir = os.path.join("data", "validation_ood", "unsupported_animals")
for fname in sorted(os.listdir(unsupported_dir)):
    res = analyze_image(os.path.join(unsupported_dir, fname))
    unsupported_stats.append(res)
    print(f"[{fname:15s}]: Agree={str(res['agreement']):5s} | R={res['r_pred']} ({res['r_conf']:5.1f}%), S={res['s_pred']} ({res['s_conf']:5.1f}%) | Energy: R={res['r_energy']:5.2f}, S={res['s_energy']:5.2f}, Avg={res['avg_energy']:5.2f} | Margin={res['avg_margin']:5.2f}")

print("\n--- 3. NON-ANIMAL OBJECTS ---")
non_animal_stats = []
non_animal_dir = os.path.join("data", "validation_ood", "non_animals")
for fname in sorted(os.listdir(non_animal_dir)):
    res = analyze_image(os.path.join(non_animal_dir, fname))
    non_animal_stats.append(res)
    print(f"[{fname:15s}]: Agree={str(res['agreement']):5s} | R={res['r_pred']} ({res['r_conf']:5.1f}%), S={res['s_pred']} ({res['s_conf']:5.1f}%) | Energy: R={res['r_energy']:5.2f}, S={res['s_energy']:5.2f}, Avg={res['avg_energy']:5.2f} | Margin={res['avg_margin']:5.2f}")

print("\n--- 4. AMBIGUOUS / LOW-QUALITY IMAGES ---")
ambiguous_stats = []
amb_dir = os.path.join("data", "validation_ood", "ambiguous")
for fname in sorted(os.listdir(amb_dir)):
    res = analyze_image(os.path.join(amb_dir, fname))
    ambiguous_stats.append(res)
    print(f"[{fname:22s}]: Agree={str(res['agreement']):5s} | R={res['r_pred']} ({res['r_conf']:5.1f}%), S={res['s_pred']} ({res['s_conf']:5.1f}%) | Energy: R={res['r_energy']:5.2f}, S={res['s_energy']:5.2f}, Avg={res['avg_energy']:5.2f} | Margin={res['avg_margin']:5.2f}")
