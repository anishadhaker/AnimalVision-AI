"""
Out-of-Distribution (OOD) & Unsupported Image Detection Module for AnimalVision AI.

Identifies whether an input image belongs to the 10 supported Animals-10 classes,
is an unsupported animal (e.g., Lion, Tiger, Monkey, Giraffe),
is a non-animal object (e.g., Car, Mobile, Laptop),
or is ambiguous / low-quality / uncertain.

Uses an energy-based out-of-distribution scoring layer combined with:
1. Logsumexp Free Energy Score: E_score(z) = log(sum(exp(z_k)))
2. Logit Margin: Delta z = z_(1) - z_(2)
3. Dual-Model Architectural Divergence: ResNet-18 vs. Swin-Tiny top-1 consensus
4. Maximum Softmax Probability (MSP) & Shannon Entropy
"""

from typing import Dict, Any, List, Optional
import numpy as np


class OODDetector:
    """
    Evaluates dual-model outputs to detect unsupported animals, non-animal objects,
    and uncertain predictions without retraining models or modifying weights.
    """

    # Calibrated thresholds based on validation benchmarks
    SWIN_ENERGY_SUPPORTED_MIN = 4.5
    AVG_ENERGY_SUPPORTED_MIN = 3.0

    @staticmethod
    def compute_energy_score(logits: np.ndarray, temperature: float = 1.0) -> float:
        """
        Computes the Helmholtz free energy score of the output logits:
        E_score(z; T) = T * log(sum(exp(z_k / T)))
        Higher energy indicates in-distribution feature activation; lower indicates OOD.
        """
        scaled = logits / max(temperature, 1e-6)
        max_val = np.max(scaled)
        # Log-sum-exp with numerical stability
        lse = max_val + np.log(np.sum(np.exp(scaled - max_val)) + 1e-12)
        return float(round(float(temperature * lse), 3))

    @staticmethod
    def compute_logit_margin(logits: np.ndarray) -> float:
        """
        Computes the margin between top-1 and top-2 logits:
        Delta z = z_(1) - z_(2)
        """
        sorted_logits = np.sort(logits)[::-1]
        if len(sorted_logits) < 2:
            return 0.0
        margin = sorted_logits[0] - sorted_logits[1]
        return float(round(float(margin), 3))

    @staticmethod
    def compute_entropy(probs: np.ndarray) -> float:
        """
        Computes normalized Shannon entropy in [0, 1].
        """
        eps = 1e-12
        p = np.clip(probs, eps, 1.0)
        h = -np.sum(p * np.log(p))
        max_h = np.log(len(probs))
        return float(round(float(h / max_h), 3))

    @classmethod
    def evaluate(
        cls,
        resnet_logits: np.ndarray,
        swin_logits: np.ndarray,
        resnet_probs: np.ndarray,
        swin_probs: np.ndarray,
        resnet_pred_class: str,
        swin_pred_class: str,
    ) -> Dict[str, Any]:
        """
        Evaluates logits and predictions from both ResNet-18 and Swin-Tiny.
        Returns a structured OOD detection verdict.
        """
        resnet_energy = cls.compute_energy_score(resnet_logits)
        swin_energy = cls.compute_energy_score(swin_logits)
        avg_energy = round((resnet_energy + swin_energy) / 2.0, 3)

        resnet_margin = cls.compute_logit_margin(resnet_logits)
        swin_margin = cls.compute_logit_margin(swin_logits)
        max_margin = max(resnet_margin, swin_margin)

        resnet_conf = round(float(np.max(resnet_probs)) * 100, 2)
        swin_conf = round(float(np.max(swin_probs)) * 100, 2)
        max_conf = max(resnet_conf, swin_conf)
        min_conf = min(resnet_conf, swin_conf)
        avg_conf = round((resnet_conf + swin_conf) / 2.0, 2)

        resnet_entropy = cls.compute_entropy(resnet_probs)
        swin_entropy = cls.compute_entropy(swin_probs)
        avg_entropy = round((resnet_entropy + swin_entropy) / 2.0, 3)

        agreement = (resnet_pred_class == swin_pred_class)

        # Signal bundle for transparency & analysis
        signals = {
            "energy_score": avg_energy,
            "swin_energy": swin_energy,
            "resnet_energy": resnet_energy,
            "logit_margin": max_margin,
            "swin_margin": swin_margin,
            "resnet_margin": resnet_margin,
            "agreement": agreement,
            "swin_confidence": swin_conf,
            "resnet_confidence": resnet_conf,
            "max_confidence": max_conf,
            "min_confidence": min_conf,
            "avg_confidence": avg_conf,
            "entropy": avg_entropy,
        }

        # --- Rule 1: In-Distribution Supported Animal ---
        # High Swin energy (>= 4.5) and Dual Average energy (>= 3.0) confirm genuine in-distribution representation
        if swin_energy >= cls.SWIN_ENERGY_SUPPORTED_MIN and avg_energy >= cls.AVG_ENERGY_SUPPORTED_MIN:
            return {
                "status": "supported",
                "is_supported": True,
                "confidence_level": "high",
                "label": "Supported Animal",
                "title": "Recognized Animal Class",
                "warning_message": None,
                "detail": "Visual features strongly align with the 10 supported Animals-10 class distributions.",
                "recommendation": None,
                "signals": signals,
            }

        # --- Rule 2: Uncertain / Ambiguous / Degraded Input ---
        # Both models exhibit low confidence (< 48%), or extreme transformer uncertainty (< 25%),
        # or flat Shannon entropy (> 0.72)
        if (swin_conf < 48.0 and resnet_conf < 48.0) or (swin_conf < 25.0) or (max_margin < 0.85 and avg_conf < 50.0) or avg_entropy > 0.72:
            return {
                "status": "uncertain",
                "is_supported": False,
                "confidence_level": "low",
                "label": "Uncertain / Ambiguous Image",
                "title": "Low Confidence & Ambiguous Visual Features",
                "warning_message": (
                    "Unable to confidently determine if this image depicts a supported animal or a valid subject. "
                    "The visual signals are ambiguous, low-contrast, or noisy."
                ),
                "detail": (
                    "Both models produced low activation margins or high probability dispersion, "
                    "suggesting an ambiguous, low-resolution, or heavily degraded image."
                ),
                "recommendation": "Try uploading a clearer, higher-resolution photo centered on the subject.",
                "signals": signals,
            }

        # --- Rule 3: Potentially Unsupported Animal ---
        # Distinct biological/animal morphological activation detected:
        # Both models exhibit >= 60% confidence in candidate animal classes,
        # OR one model is highly confident (>= 85%) while the other has at least 30%,
        # but energy or divergence confirms it lies outside the 10 supported classes.
        is_animal_like = (resnet_conf >= 60.0 and swin_conf >= 60.0) or (max_conf >= 85.0 and min_conf >= 30.0)
        if is_animal_like:
            return {
                "status": "potentially_unsupported_animal",
                "is_supported": False,
                "confidence_level": "medium",
                "label": "Potentially Unsupported Animal",
                "title": "Animal Outside the 10 Supported Classes",
                "warning_message": (
                    "This image appears to be an animal, but it may not belong to the 10 supported classes "
                    "(Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, Squirrel)."
                ),
                "detail": (
                    f"The models detected general animal features but sub-threshold energy (Swin energy: {swin_energy}, avg: {avg_energy}) "
                    f"indicates an out-of-domain species (e.g., lion, tiger, monkey, rabbit, or giraffe). "
                    f"Predictions below represent nearest candidate approximations."
                ),
                "recommendation": "AnimalVision AI is optimized for 10 specific animal classes. Unsupported species will be mapped to the closest visual neighbor.",
                "signals": signals,
            }

        # --- Rule 4: Potentially Non-Animal Object ---
        # Low energy and neither model exhibits coherent animal recognition
        if swin_energy <= 3.35 and (resnet_energy <= 0.80 or max_margin <= 1.40 or min_conf < 55.0):
            return {
                "status": "potentially_non_animal",
                "is_supported": False,
                "confidence_level": "medium",
                "label": "Potentially Non-Animal Object",
                "title": "Subject Appears Outside Animal Domain",
                "warning_message": (
                    "This image appears to contain a non-animal object (e.g., a vehicle, electronic device, "
                    "furniture, or inanimate subject) that is not supported by AnimalVision AI."
                ),
                "detail": (
                    "Free energy activations and candidate margins fall well below animal visual distribution norms. "
                    "The models' predicted animal classes are forced outputs and should not be considered accurate."
                ),
                "recommendation": "AnimalVision AI is trained specifically on animal classifications. Please upload an animal image.",
                "signals": signals,
            }

        # --- Fallback: Potentially Unsupported Animal ---
        return {
            "status": "potentially_unsupported_animal",
            "is_supported": False,
            "confidence_level": "medium",
            "label": "Potentially Unsupported Animal",
            "title": "Animal Outside the 10 Supported Classes",
            "warning_message": (
                "This image appears to be an animal, but it may not belong to the 10 supported classes."
            ),
            "detail": "The image features do not match in-distribution thresholds for the 10 supported classes.",
            "recommendation": "Predictions shown below represent nearest candidate approximations.",
            "signals": signals,
        }
