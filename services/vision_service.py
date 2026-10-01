"""Kisan Ki Awaz - Vision Service."""
from typing import Dict, Optional

from loguru import logger
from PIL import Image

from config import settings
from models.vision_model import DemoVisionModel, OpenAIVisionModel, VisionModelInterface


class VisionService:
    LOW_CONFIDENCE_THRESHOLD = 50.0
    MEDIUM_CONFIDENCE_THRESHOLD = 70.0

    def __init__(self, model: Optional[VisionModelInterface] = None):
        if model is not None:
            self.model = model
        elif settings.llm.openai_api_key:
            self.model = OpenAIVisionModel()
        else:
            self.model = DemoVisionModel()
        self._is_demo = isinstance(self.model, DemoVisionModel)
        logger.info(f"Vision service initialized: {type(self.model).__name__}")

    def analyze_image(self, image: Image.Image) -> Dict:
        validation = self._validate_image(image)
        if not validation["valid"]:
            return {"success": False, "error": validation["error"], "prediction": None,
                    "confidence": 0, "risk_level": "unknown", "warnings": [validation["error"]],
                    "is_demo": self._is_demo}
        processed = self._preprocess(image)
        try:
            result = self.model.predict(processed)
        except Exception as e:
            logger.exception("Vision model inference failed")
            return {"success": False, "error": f"Image analysis failed: {str(e)}",
                    "prediction": None, "confidence": 0, "risk_level": "unknown",
                    "warnings": ["The image could not be analyzed. Please try a clear JPG or PNG image."],
                    "is_demo": self._is_demo}
        result["success"] = True
        result["warnings"] = self._generate_warnings(result)
        result["image_quality"] = self._assess_image_quality(processed)
        result["is_demo"] = self._is_demo
        return result

    def _validate_image(self, image: Optional[Image.Image]) -> Dict:
        if image is None:
            return {"valid": False, "error": "No image provided. Please upload or capture an image."}
        if not isinstance(image, Image.Image):
            return {"valid": False, "error": "Invalid image format."}
        w, h = image.size
        if w < 50 or h < 50:
            return {"valid": False, "error": "Image too small. Please upload a clearer, larger image."}
        return {"valid": True}

    def _preprocess(self, image: Image.Image) -> Image.Image:
        image = image.copy()
        try:
            from PIL import ImageOps
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass
        if image.size[0] > 1200 or image.size[1] > 1200:
            image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        return image.convert("RGB") if image.mode != "RGB" else image

    def _generate_warnings(self, result: Dict) -> list:
        warnings = []
        confidence = float(result.get("confidence", 0))
        if confidence < self.LOW_CONFIDENCE_THRESHOLD:
            warnings.append("LOW CONFIDENCE: This is a preliminary observation, not a confirmed diagnosis. Please consult an agricultural expert.")
        elif confidence < self.MEDIUM_CONFIDENCE_THRESHOLD:
            warnings.append("MEDIUM CONFIDENCE: The result may not be fully accurate. Verify with an agricultural extension officer.")
        if result.get("uncertain"):
            warnings.append("The image does not provide enough visual evidence for a reliable disease identification.")
        if self._is_demo:
            warnings.append("No real vision model is configured; this is a fallback analysis. Configure OPENAI_API_KEY for real image analysis.")
        quality = result.get("image_quality", {})
        if quality.get("is_blurry"):
            warnings.append("Image appears blurry. A clearer close-up image would improve accuracy.")
        if quality.get("is_dark"):
            warnings.append("Image appears dark. Better lighting would improve accuracy.")
        return warnings

    def _assess_image_quality(self, image: Image.Image) -> Dict:
        import numpy as np
        img_array = np.array(image.convert("RGB"))
        brightness = float(img_array.mean())
        gray = img_array.mean(axis=2)
        variance = float(np.var(gray))
        return {"brightness": round(brightness, 1), "contrast": round(variance, 1),
                "is_dark": brightness < 60, "is_bright": brightness > 230,
                "is_blurry": variance < 500, "resolution": f"{image.size[0]}x{image.size[1]}"}

    def get_available_models(self) -> list:
        return [
            {"name": "openai-vision", "description": "OpenAI multimodal crop/disease image analysis"},
            {"name": "demo", "description": "Local fallback analysis when no vision API is configured"},
        ]
