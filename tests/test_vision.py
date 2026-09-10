"""
Tests for the Vision Service and Vision Model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from models.vision_model import DemoVisionModel, CROP_DISEASE_LABELS
from services.vision_service import VisionService
from services.image_service import ImageService


class TestDemoVisionModel:
    """Test the demo vision model."""

    def setup_method(self):
        self.model = DemoVisionModel()

    def test_has_class_labels(self):
        labels = self.model.get_class_labels()
        assert len(labels) > 0
        assert len(labels) == len(CROP_DISEASE_LABELS)

    def test_predict_returns_valid_structure(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = self.model.predict(img)

        assert "prediction" in result
        assert "confidence" in result
        assert "all_predictions" in result
        assert "risk_level" in result
        assert "crop_detected" in result
        assert "disease_detected" in result

    def test_predict_confidence_in_range(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = self.model.predict(img)
        assert 0 <= result["confidence"] <= 100

    def test_predict_risk_level_valid(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = self.model.predict(img)
        assert result["risk_level"] in ("low", "medium", "high", "unknown")

    def test_predict_prediction_in_labels(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = self.model.predict(img)
        assert result["prediction"] in CROP_DISEASE_LABELS

    def test_predict_with_green_image(self):
        # Mostly green image should tend toward "healthy" predictions
        green = np.zeros((100, 100, 3), dtype=np.uint8)
        green[:, :, 1] = 200  # High green channel
        img = Image.fromarray(green)
        result = self.model.predict(img)
        assert result["prediction"] in CROP_DISEASE_LABELS

    def test_predict_with_brown_image(self):
        # Brown image should tend toward disease predictions
        brown = np.zeros((100, 100, 3), dtype=np.uint8)
        brown[:, :, 0] = 160  # Red
        brown[:, :, 1] = 80   # Green
        brown[:, :, 2] = 40   # Blue
        img = Image.fromarray(brown)
        result = self.model.predict(img)
        assert result["prediction"] in CROP_DISEASE_LABELS

    def test_none_image_returns_error(self):
        result = self.model.predict(None)
        assert result["confidence"] == 0
        assert "error" in result


class TestVisionService:
    """Test the vision service layer."""

    def setup_method(self):
        self.service = VisionService()

    def test_analyze_valid_image(self):
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        result = self.service.analyze_image(img)
        assert result["success"] is True
        assert "warnings" in result
        assert "image_quality" in result

    def test_analyze_none_image(self):
        result = self.service.analyze_image(None)
        assert result["success"] is False
        assert "error" in result

    def test_analyze_tiny_image(self):
        img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        result = self.service.analyze_image(img)
        assert result["success"] is False
        assert "too small" in result.get("error", "").lower()

    def test_low_confidence_warning(self):
        # The demo model may produce low confidence; check warnings exist
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        result = self.service.analyze_image(img)
        assert isinstance(result["warnings"], list)

    def test_image_quality_assessment(self):
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        result = self.service.analyze_image(img)
        quality = result["image_quality"]
        assert "brightness" in quality
        assert "contrast" in quality
        assert "is_blurry" in quality
        assert "is_dark" in quality
        assert "resolution" in quality

    def test_is_demo_flag(self):
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        result = self.service.analyze_image(img)
        assert result.get("is_demo") is True

    def test_get_available_models(self):
        models = self.service.get_available_models()
        assert len(models) >= 3
        names = [m["name"] for m in models]
        assert "demo" in names


class TestImageService:
    """Test the image processing service."""

    def setup_method(self):
        self.service = ImageService()

    def test_preprocess_rgb(self):
        # RGBA image should be converted to RGB
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8), "RGBA")
        result = self.service.preprocess(img)
        assert result.mode == "RGB"

    def test_preprocess_resize_large(self):
        img = Image.fromarray(np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8))
        result = self.service.preprocess(img)
        assert result.size[0] <= 800
        assert result.size[1] <= 800

    def test_preprocess_preserves_small(self):
        img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        result = self.service.preprocess(img)
        assert result.size == (100, 100)

    def test_load_pil_image(self):
        img = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
        result = self.service.load_image(img)
        assert isinstance(result, Image.Image)

    def test_load_none_returns_none(self):
        result = self.service.load_image(None)
        assert result is None
