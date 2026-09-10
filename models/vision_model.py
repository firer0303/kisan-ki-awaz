"""
Kisan Ki Awaz - Vision Model Interface
========================================
Abstract interface and demo implementation for agricultural
computer-vision (crop disease / pest detection).

Designed so a fine-tuned model can replace the demo model
without changing any calling code.
"""
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


class VisionModelInterface(ABC):
    """Abstract interface for all crop-disease vision models."""

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict:
        """
        Run inference on a single image.

        Returns:
            Dict with keys:
            - prediction: str (top predicted class)
            - confidence: float (0-100)
            - all_predictions: List[Tuple[str, float]]
            - risk_level: str ("low" | "medium" | "high" | "critical")
            - crop_detected: str
            - disease_detected: str or None
        """
        ...

    @abstractmethod
    def get_class_labels(self) -> List[str]:
        """Return all class labels the model can predict."""
        ...


# ──────────────────────────────────────────────────────────────
# Demo model (used when no real model is configured)
# ──────────────────────────────────────────────────────────────

# Realistic crop-disease class labels (matching a fine-tuned model output)
CROP_DISEASE_LABELS = [
    "Wheat - Leaf Rust",
    "Wheat - Stripe Rust",
    "Wheat - Healthy",
    "Rice - Blast",
    "Rice - Brown Spot",
    "Rice - Healthy",
    "Cotton - Bollworm Damage",
    "Cotton - Leaf Curl Virus",
    "Cotton - Healthy",
    "Maize - Fall Armyworm",
    "Maize - Northern Leaf Blight",
    "Maize - Healthy",
    "Tomato - Late Blight",
    "Tomato - Early Blight",
    "Tomato - Leaf Curl",
    "Tomato - Healthy",
    "Potato - Late Blight",
    "Potato - Early Blight",
    "Potato - Healthy",
    "Citrus - Canker",
    "Citrus - Greening (HLB)",
    "Citrus - Healthy",
    "Mango - Anthracnose",
    "Mango - Sudden Death",
    "Mango - Healthy",
    "Sugarcane - Red Rot",
    "Sugarcane - Smut",
    "Sugarcane - Healthy",
    "Chili - Leaf Curl Virus",
    "Chili - Anthracnose",
    "Chili - Healthy",
]

# Mapping from class label -> crop, disease, risk
_CLASS_METADATA: Dict[str, Dict] = {}
for _label in CROP_DISEASE_LABELS:
    _parts = _label.split(" - ", 1)
    _crop = _parts[0].strip()
    _disease = _parts[1].strip() if len(_parts) > 1 else "Unknown"
    _is_healthy = "Healthy" in _disease
    _CLASS_METADATA[_label] = {
        "crop": _crop,
        "disease": _disease if not _is_healthy else None,
        "risk_level": "low" if _is_healthy else random.choice(["medium", "high"]),
    }
# Fix risk levels deterministically for the demo
for _label, _meta in _CLASS_METADATA.items():
    if _meta["disease"] is None:
        _meta["risk_level"] = "low"
    elif any(
        kw in _label
        for kw in ["Blast", "Red Rot", "Sudden Death", "Greening", "Armyworm"]
    ):
        _meta["risk_level"] = "high"
    elif any(kw in _label for kw in ["Rust", "Blight", "Canker", "Curl", "Rot"]):
        _meta["risk_level"] = "medium"
    else:
        _meta["risk_level"] = "medium"


class DemoVisionModel(VisionModelInterface):
    """
    Demo vision model that simulates crop-disease classification.

    In production, replace this with:
    - A fine-tuned ResNet/EfficientNet model via torch
    - Alibaba Cloud Vision API
    - Hugging Face pipeline with a crop-disease model

    The demo model analyzes basic image properties (color histograms,
    texture) to produce plausible (but NOT real) predictions.
    """

    def __init__(self):
        self.labels = CROP_DISEASE_LABELS
        self.metadata = _CLASS_METADATA

    def predict(self, image: Image.Image) -> Dict:
        """
        Simulate prediction based on image characteristics.
        Uses color analysis to pick plausible classes.
        """
        # Validate image
        if image is None:
            return self._empty_result("No image provided")

        # Resize for analysis
        img_small = image.resize((64, 64))
        pixels = np.array(img_small)

        # Analyze color distribution to pick a plausible class
        avg_r = pixels[:, :, 0].mean() if len(pixels.shape) == 3 else 128
        avg_g = pixels[:, :, 1].mean() if len(pixels.shape) == 3 else 128
        avg_b = pixels[:, :, 2].mean() if len(pixels.shape) == 3 else 128

        # Simple heuristic: green-heavy = healthy, brown/yellow = diseased
        green_ratio = avg_g / (avg_r + avg_g + avg_b + 1)
        brown_indicator = (avg_r > 120 and avg_g < 100 and avg_b < 80)

        # Use image hash as seed for reproducible "predictions"
        img_hash = hash(pixels.tobytes()[:100]) % 1000
        random.seed(img_hash)

        if green_ratio > 0.38 and not brown_indicator:
            # Likely healthy or mild issue
            healthy_labels = [l for l in self.labels if "Healthy" in l]
            top_label = random.choice(healthy_labels)
            confidence = random.uniform(65, 92)
        elif brown_indicator:
            # Likely disease
            disease_labels = [
                l
                for l in self.labels
                if "Healthy" not in l
                and any(kw in l for kw in ["Rust", "Blight", "Rot", "Spot"])
            ]
            top_label = random.choice(disease_labels) if disease_labels else random.choice(self.labels)
            confidence = random.uniform(45, 78)
        else:
            # Mixed - could be pest or disease
            issue_labels = [l for l in self.labels if "Healthy" not in l]
            top_label = random.choice(issue_labels)
            confidence = random.uniform(40, 75)

        # Build full prediction list
        all_preds = [(top_label, confidence)]
        remaining = [l for l in self.labels if l != top_label]
        random.shuffle(remaining)
        remaining_conf = 100 - confidence
        for label in remaining[:4]:
            c = remaining_conf * random.uniform(0.05, 0.3)
            all_preds.append((label, round(c, 1)))
            remaining_conf -= c

        meta = self.metadata.get(top_label, {})

        return {
            "prediction": top_label,
            "confidence": round(confidence, 1),
            "all_predictions": all_preds,
            "risk_level": meta.get("risk_level", "medium"),
            "crop_detected": meta.get("crop", "Unknown"),
            "disease_detected": meta.get("disease"),
            "is_demo": True,
        }

    def _empty_result(self, reason: str) -> Dict:
        return {
            "prediction": "Unable to analyze",
            "confidence": 0,
            "all_predictions": [],
            "risk_level": "unknown",
            "crop_detected": "Unknown",
            "disease_detected": None,
            "error": reason,
            "is_demo": True,
        }

    def get_class_labels(self) -> List[str]:
        return list(self.labels)
