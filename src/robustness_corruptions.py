"""
Image Corruption Engine for AnimalVision AI.
Provides deterministic, physics-grounded synthetic corruptions applied to PyTorch
tensors in the [0.0, 1.0] range BEFORE ImageNet normalization.

Supported Corruptions (5 types, 5 severity levels each):
1. Gaussian Noise (additive sensor noise)
2. Gaussian Blur (optical defocus / high-frequency attenuation)
3. Brightness (photometric illumination shift / saturation)
4. Contrast (dynamic range attenuation towards mean grayscale)
5. Atmospheric Fog (Koschmieder atmospheric scattering model with smooth spatial haze)
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter


# Canonical severity parameters
CORRUPTION_CONFIG: Dict[str, Dict[str, Any]] = {
    "gaussian_noise": {
        "name": "Gaussian Noise",
        "description": "Additive zero-mean Gaussian noise simulating sensor/thermal noise.",
        "params": {
            1: {"sigma": 0.04, "label": "sigma=0.04 (mild)"},
            2: {"sigma": 0.08, "label": "sigma=0.08 (low)"},
            3: {"sigma": 0.14, "label": "sigma=0.14 (moderate)"},
            4: {"sigma": 0.20, "label": "sigma=0.20 (high)"},
            5: {"sigma": 0.28, "label": "sigma=0.28 (severe)"},
        },
    },
    "gaussian_blur": {
        "name": "Gaussian Blur",
        "description": "Defocus / optical blur via spatial Gaussian convolution kernel.",
        "params": {
            1: {"kernel_size": 5, "sigma": 0.75, "label": "k=5, s=0.75 (mild)"},
            2: {"kernel_size": 7, "sigma": 1.25, "label": "k=7, s=1.25 (low)"},
            3: {"kernel_size": 11, "sigma": 2.00, "label": "k=11, s=2.00 (moderate)"},
            4: {"kernel_size": 15, "sigma": 3.00, "label": "k=15, s=3.00 (high)"},
            5: {"kernel_size": 21, "sigma": 4.50, "label": "k=21, s=4.50 (severe)"},
        },
    },
    "brightness": {
        "name": "Brightness",
        "description": "Photometric illumination shift simulating high solar over-exposure.",
        "params": {
            1: {"factor": 1.20, "label": "factor=1.20 (mild)"},
            2: {"factor": 1.40, "label": "factor=1.40 (low)"},
            3: {"factor": 1.65, "label": "factor=1.65 (moderate)"},
            4: {"factor": 1.90, "label": "factor=1.90 (high)"},
            5: {"factor": 2.20, "label": "factor=2.20 (severe)"},
        },
    },
    "contrast": {
        "name": "Contrast",
        "description": "Dynamic range compression towards mean grayscale intensity.",
        "params": {
            1: {"factor": 0.80, "label": "factor=0.80 (mild)"},
            2: {"factor": 0.60, "label": "factor=0.60 (low)"},
            3: {"factor": 0.40, "label": "factor=0.40 (moderate)"},
            4: {"factor": 0.25, "label": "factor=0.25 (high)"},
            5: {"factor": 0.15, "label": "factor=0.15 (severe)"},
        },
    },
    "atmospheric_fog": {
        "name": "Atmospheric Fog",
        "description": "Atmospheric haze scattering model blending smooth fractal fog with image.",
        "params": {
            1: {"alpha": 0.15, "label": "alpha=0.15 (mild)"},
            2: {"alpha": 0.30, "label": "alpha=0.30 (low)"},
            3: {"alpha": 0.45, "label": "alpha=0.45 (moderate)"},
            4: {"alpha": 0.60, "label": "alpha=0.60 (high)"},
            5: {"alpha": 0.75, "label": "alpha=0.75 (severe)"},
        },
    },
}

SUPPORTED_CORRUPTIONS = list(CORRUPTION_CONFIG.keys())


def apply_gaussian_noise(
    tensor: torch.Tensor,
    severity: int,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Applies additive Gaussian noise to a tensor in [0.0, 1.0].
    """
    assert 1 <= severity <= 5, f"Severity must be in 1..5, got {severity}"
    sigma = CORRUPTION_CONFIG["gaussian_noise"]["params"][severity]["sigma"]

    if seed is not None:
        generator = torch.Generator(device=tensor.device).manual_seed(seed)
        noise = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)
    else:
        noise = torch.randn_like(tensor)

    return torch.clamp(tensor + noise * sigma, 0.0, 1.0)


def apply_gaussian_blur(
    tensor: torch.Tensor,
    severity: int,
) -> torch.Tensor:
    """
    Applies Gaussian blur kernel to a tensor in [0.0, 1.0].
    """
    assert 1 <= severity <= 5, f"Severity must be in 1..5, got {severity}"
    k = CORRUPTION_CONFIG["gaussian_blur"]["params"][severity]["kernel_size"]
    sigma = CORRUPTION_CONFIG["gaussian_blur"]["params"][severity]["sigma"]
    return TF.gaussian_blur(tensor, kernel_size=[k, k], sigma=[sigma, sigma])


def apply_brightness(
    tensor: torch.Tensor,
    severity: int,
) -> torch.Tensor:
    """
    Applies brightness scaling factor to a tensor in [0.0, 1.0].
    """
    assert 1 <= severity <= 5, f"Severity must be in 1..5, got {severity}"
    factor = CORRUPTION_CONFIG["brightness"]["params"][severity]["factor"]
    return TF.adjust_brightness(tensor, factor)


def apply_contrast(
    tensor: torch.Tensor,
    severity: int,
) -> torch.Tensor:
    """
    Applies contrast adjustment factor to a tensor in [0.0, 1.0].
    """
    assert 1 <= severity <= 5, f"Severity must be in 1..5, got {severity}"
    factor = CORRUPTION_CONFIG["contrast"]["params"][severity]["factor"]
    return TF.adjust_contrast(tensor, factor)


def _generate_smooth_fog_mask(batch_size: int, height: int, width: int, seed: int = 42) -> torch.Tensor:
    """
    Generates deterministic low-frequency haze masks simulating spatial fog depth variations.
    """
    rng = np.random.RandomState(seed)
    masks = []
    for _ in range(batch_size):
        noise = rng.randn(height, width)
        smooth = gaussian_filter(noise, sigma=height // 6)
        smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-8)
        # Airlight haze centered in [0.85, 1.0]
        haze = 0.85 + 0.15 * smooth
        masks.append(haze)

    arr = np.stack(masks, axis=0)[:, np.newaxis, :, :]  # [B, 1, H, W]
    arr = np.repeat(arr, 3, axis=1)  # [B, 3, H, W]
    return torch.from_numpy(arr).float()


def apply_atmospheric_fog(
    tensor: torch.Tensor,
    severity: int,
    seed: int = 42,
) -> torch.Tensor:
    """
    Applies atmospheric scattering fog degradation to a tensor in [0.0, 1.0].
    """
    assert 1 <= severity <= 5, f"Severity must be in 1..5, got {severity}"
    alpha = CORRUPTION_CONFIG["atmospheric_fog"]["params"][severity]["alpha"]

    # Ensure 4D shape [B, C, H, W]
    is_3d = tensor.ndim == 3
    if is_3d:
        tensor = tensor.unsqueeze(0)

    B, C, H, W = tensor.shape
    fog_mask = _generate_smooth_fog_mask(B, H, W, seed=seed).to(tensor.device, dtype=tensor.dtype)
    corrupted = torch.clamp((1.0 - alpha) * tensor + alpha * fog_mask, 0.0, 1.0)

    if is_3d:
        corrupted = corrupted.squeeze(0)

    return corrupted


def apply_corruption(
    tensor: torch.Tensor,
    corruption_type: str,
    severity: int,
    seed: Optional[int] = 42,
) -> torch.Tensor:
    """
    Unified dispatcher to apply any of the 5 supported corruptions.

    Args:
        tensor: Float tensor in range [0.0, 1.0] of shape [B, 3, H, W] or [3, H, W].
        corruption_type: One of 'gaussian_noise', 'gaussian_blur', 'brightness',
                         'contrast', 'atmospheric_fog'.
        severity: Integer in 1..5.
        seed: Random seed for stochastic corruptions (noise, fog).

    Returns:
        Corrupted tensor with identical shape and dtype, clamped to [0.0, 1.0].
    """
    if corruption_type == "gaussian_noise":
        return apply_gaussian_noise(tensor, severity, seed=seed)
    elif corruption_type == "gaussian_blur":
        return apply_gaussian_blur(tensor, severity)
    elif corruption_type == "brightness":
        return apply_brightness(tensor, severity)
    elif corruption_type == "contrast":
        return apply_contrast(tensor, severity)
    elif corruption_type == "atmospheric_fog":
        return apply_atmospheric_fog(tensor, severity, seed=seed if seed is not None else 42)
    else:
        raise ValueError(
            f"Unsupported corruption_type: '{corruption_type}'. Supported: {SUPPORTED_CORRUPTIONS}"
        )
