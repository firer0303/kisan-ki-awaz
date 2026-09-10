"""
Kisan Ki Awaz - Vision Service
================================
High-level service wrapping the vision model with preprocessing,
validation, confidence handling, and low-confidence warnings.
"""
from typing import Dict, Optional

from loguru import logger
from PIL import Image

from models.vision_model import DemoVisionModel, VisionModelInterface


class VisionService:
    """
    Orchestrates image preprocessing, model inference, and result
    post-processing for crop/disease detection.
    """

    # Minimum confidence to consider a detection reliable
    LOW_CONFIDENCE_THRESHOLD = 50.0
    MEDIUM_CONFIDENCE_THRESHOLD = 70.0

    def __init__(self, model: Optional[VisionModelInterface] = None):
        """
        Initialize with a vision model. Defaults to DemoVisionModel
        if no model is provided.
        """
        self.model = model or DemoVisionModel()
        self._is_demo = isinstance(self.model, DemoVisionModel)
        logger.info(
            f"Vision service initialized (model: "
            f"{'DEMO' if self._is_demo else type(self.model).__name__})"
        )

    def analyze_image(self, image: Image.Image) -> Dict:
        """
        Full image analysis pipeline:
        1. Validate image
        2. Preprocess
        3. Run model inference
        4. Post-process and add warnings
        """
        # Step 1: Validate
        validation = self._validate_image(image)
        if not validation["valid"]:
            return {
                "success": False,
                "error": validation["error"],
                "prediction": None,
                "confidence": 0,
                "risk_level": "unknown",
                "warnings": [validation["error"]],
                "is_demo": self._is_demo,
            }

        # Step 2: Preprocess
        processed = self._preprocess(image)

        # Step 3: Model inference
        try:
            result = self.model.predict(processed)
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "prediction": None,
                "confidence": 0,
                "risk_level": "unknown",
                "warnings": ["Model inference error"],
                "is_demo": self._is_demo,
            }

        # Step 4: Post-process
        result["success"] = True
        result["warnings"] = self._generate_warnings(result)
        result["image_quality"] = self._assess_image_quality(image)
        result["is_demo"] = self._is_demo

        return result

    def _validate_image(self, image: Optional[Image.Image]) -> Dict:
        """Validate the input image."""
        if image is None:
            return {"valid": False, "error": "No image provided. Please upload or capture an image."}
        if not isinstance(image, Image.Image):
            return {"valid": False, "error": "Invalid image format."}
        w, h = image.size
        if w < 50 or h < 50:
            return {
                "valid": False,
                "error": "Image too small. Please upload a clearer, larger image.",
            }
        return {"valid": True}

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for model input.
        For the demo model, just ensure RGB format.
        For a real model: resize, normalize, etc.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _generate_warnings(self, result: Dict) -> list:
        """Generate user-facing warnings based on prediction confidence."""
        warnings = []
        confidence = result.get("confidence", 0)

        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                "LOW CONFIDENCE: This is a preliminary observation only, "
                "NOT a confirmed diagnosis. Please consult an agricultural expert."
            )
            warnings.append(
                "For better results, upload a clear, well-lit close-up photo "
                "of the affected plant part."
            )
        elif confidence < self.MEDIUM_CONFIDENCE_THRESHOLD:
            warnings.append(
                "MEDIUM CONFIDENCE: The diagnosis may not be fully accurate. "
                "Consider verifying with an agricultural extension officer."
            )

        if self._is_demo:
            warnings.append(
                "DEMO MODE: Using simulated analysis. Connect a trained model "
                "for real predictions."
            )

        quality = result.get("image_quality", {})
        if quality.get("is_blurry"):
            warnings.append("Image appears blurry. A clearer image would improve accuracy.")
        if quality.get("is_dark"):
            warnings.append("Image appears dark. Better lighting would improve accuracy.")

        return warnings

    def _assess_image_quality(self, image: Image.Image) -> Dict:
        """Basic image quality assessment."""
        import numpy as np

        img_array = np.array(image.convert("RGB"))

        # Brightness check
        brightness = img_array.mean()
        is_dark = brightness < 60
        is_bright = brightness > 230

        # Contrast/blur check (variance of Laplacian approximation)
        gray = img_array.mean(axis=2)
        variance = np.var(gray)
        is_blurry = variance < 500

        return {
            "brightness": round(brightness, 1),
            "contrast": round(variance, 1),
            "is_dark": is_dark,
            "is_bright": is_bright,
            "is_blurry": is_blurry,
            "resolution": f"{image.size[0]}x{image.size[1]}",
        }

    def get_available_models(self) -> list:
        """List available model backends."""
        return [
            {"name": "demo", "description": "Simulated analysis (for demonstration)"},
            {"name": "custom", "description": "Custom trained model (configure in .env)"},
            {"name": "alibaba", "description": "Alibaba Cloud Vision API (configure in .env)"},
        ]
