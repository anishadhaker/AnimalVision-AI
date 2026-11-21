"""
Verification script for Model Architectures (Step 2).

This script:
1. Detects compute device (CUDA GPU vs CPU).
2. Instantiates the custom AnimalCNN baseline and pretrained Swin-Tiny Transformer.
3. Computes and compares total and trainable parameters.
4. Performs dummy forward passes using a batch of two [2, 3, 224, 224] images.
5. Rigorously verifies output shape [2, 5] and finite numerical stability (no NaN/Inf).
"""

import sys
import torch

from src.cnn_model import build_cnn_model, count_parameters as count_cnn_params
from src.swin_model import (
    build_swin_model,
    count_parameters as count_swin_params,
    PretrainedWeightDownloadError,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using CUDA GPU: {device_name}")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU (No CUDA GPU detected). All operations will run on CPU.")
    return device


def test_model_forward(model: torch.nn.Module, model_name: str, device: torch.device):
    print(f"\n--- Testing {model_name} ---")
    model = model.to(device)
    model.eval()

    # Create dummy batch of 2 RGB images at 224x224
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    print(f"  Dummy input tensor shape: {list(dummy_input.shape)}")

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"  Output logits shape: {list(logits.shape)}")

    # 1. Check shape
    expected_shape = [batch_size, 5]
    if list(logits.shape) != expected_shape:
        raise ValueError(
            f"Shape mismatch for {model_name}! Expected {expected_shape}, got {list(logits.shape)}"
        )
    print("  [OK] Output shape verified as [2, 5].")

    # 2. Check for NaN or Inf
    if torch.isnan(logits).any():
        raise ValueError(f"Numerical error: NaN detected in {model_name} logits!")
    if torch.isinf(logits).any():
        raise ValueError(f"Numerical error: Inf detected in {model_name} logits!")
    print("  [OK] Numerical stability verified (all values finite, no NaN/Inf).")

    # 3. Softmax check for valid probabilities
    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5):
        raise ValueError(f"Probabilities do not sum to 1.0 in {model_name}!")
    print(f"  Sample predicted class indices: {torch.argmax(probs, dim=-1).tolist()}")
    print("  [OK] Softmax probability distribution is valid.")


def main():
    print("=" * 65)
    print("Step 2: Model Architectures Verification (CNN vs Swin-Tiny)")
    print("=" * 65)

    device = get_device()

    # -------------------------------------------------------------
    # 1. AnimalCNN Baseline
    # -------------------------------------------------------------
    print("\n[1/2] Building CNN Baseline (AnimalCNN)...")
    try:
        cnn_model = build_cnn_model(num_classes=5)
        cnn_total, cnn_trainable = count_cnn_params(cnn_model)
        print(f"  Total parameters:     {cnn_total:,}")
        print(f"  Trainable parameters: {cnn_trainable:,}")
        test_model_forward(cnn_model, "AnimalCNN", device)
    except Exception as err:
        print(f"[FAIL] AnimalCNN verification failed: {err}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------
    # 2. Pretrained Swin-Tiny Transformer
    # -------------------------------------------------------------
    print("\n[2/2] Building Pretrained Swin-Tiny Transformer...")
    print("  Loading official torchvision Swin-Tiny weights (Swin_T_Weights.DEFAULT)...")
    try:
        swin_model = build_swin_model(num_classes=5, pretrained=True)
        swin_total, swin_trainable = count_swin_params(swin_model)
        print(f"  Total parameters:     {swin_total:,}")
        print(f"  Trainable parameters: {swin_trainable:,}")
        test_model_forward(swin_model, "Swin-Tiny", device)
    except PretrainedWeightDownloadError as pwe:
        print(f"\n{pwe}", file=sys.stderr)
        sys.exit(1)
    except Exception as err:
        print(f"[FAIL] Swin-Tiny verification failed: {err}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------
    # Summary Comparison
    # -------------------------------------------------------------
    print("\n" + "=" * 65)
    print("Architecture Summary Comparison:")
    print(f"{'Model':<15} | {'Total Params':<15} | {'Trainable Params':<18} | {'Output Logits'}")
    print("-" * 65)
    print(f"{'AnimalCNN':<15} | {cnn_total:<15,} | {cnn_trainable:<18,} | [batch_size, 5]")
    print(f"{'Swin-Tiny':<15} | {swin_total:<15,} | {swin_trainable:<18,} | [batch_size, 5]")
    print("=" * 65)
    print("Both architectures successfully verified and ready for training!")


if __name__ == "__main__":
    main()
