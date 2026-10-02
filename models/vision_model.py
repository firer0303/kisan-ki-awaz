"""
Kisan Ki Awaz - Vision Model Interface
========================================
Agricultural image classification interface with an OpenAI vision backend
and a safe local fallback.
"""
import base64
import json
from abc import ABC, abstractmethod
from typing import Dict, List

import os

import numpy as np
from PIL import Image
from loguru import logger

from config import settings


class VisionModelInterface(ABC):
    @abstractmethod
    def predict(self, image: Image.Image) -> Dict:
        ...

    @abstractmethod
    def get_class_labels(self) -> List[str]:
        ...


CROP_DISEASE_LABELS = [
    "Wheat - Leaf Rust", "Wheat - Stripe Rust", "Wheat - Healthy",
    "Rice - Blast", "Rice - Brown Spot", "Rice - Healthy",
    "Cotton - Bollworm Damage", "Cotton - Leaf Curl Virus", "Cotton - Healthy",
    "Maize - Fall Armyworm", "Maize - Northern Leaf Blight", "Maize - Healthy",
    "Tomato - Late Blight", "Tomato - Early Blight", "Tomato - Leaf Curl", "Tomato - Healthy",
    "Potato - Late Blight", "Potato - Early Blight", "Potato - Healthy",
    "Citrus - Canker", "Citrus - Greening (HLB)", "Citrus - Healthy",
    "Mango - Anthracnose", "Mango - Sudden Death", "Mango - Healthy",
    "Sugarcane - Red Rot", "Sugarcane - Smut", "Sugarcane - Healthy",
    "Chili - Leaf Curl Virus", "Chili - Anthracnose", "Chili - Healthy",
]

_CLASS_METADATA = {}
for label in CROP_DISEASE_LABELS:
    crop, disease = label.split(" - ", 1)
    healthy = disease == "Healthy"
    _CLASS_METADATA[label] = {
        "crop": crop,
        "disease": None if healthy else disease,
        "risk_level": "low" if healthy else "medium",
    }


class OpenAIVisionModel(VisionModelInterface):
    """Real image analysis using the configured OpenAI multimodal model."""

    def __init__(self):
        self.model = settings.llm.openai_model or "gpt-4o"

    def predict(self, image: Image.Image) -> Dict:
        from openai import OpenAI

        image = image.convert("RGB")
        image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        import io
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        prompt = """You are an agricultural plant-disease image analyst for Pakistan.
Analyze the supplied plant/crop image carefully. Do NOT invent a disease when the image
is unclear. Identify the crop and the most likely visible disease/pest/problem if possible.
Return ONLY valid JSON with these keys:
{"crop_detected":"...","disease_detected":"... or null","prediction":"crop - disease or crop - Healthy","confidence":0-100,"risk_level":"low|medium|high|critical","reason":"brief visual evidence","uncertain":true|false}
Use confidence conservatively. If the image cannot support disease identification, set
uncertain=true, disease_detected=null, confidence<=35 and explain why in reason."""

        client = OpenAI(api_key=settings.llm.openai_api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            max_tokens=500,
            temperature=0.1,
        )
        raw = response.choices[0].message.content or "{}"
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        confidence = max(0.0, min(100.0, float(result.get("confidence", 0))))
        crop = str(result.get("crop_detected") or "Unknown")
        disease = result.get("disease_detected") or None
        prediction = str(result.get("prediction") or (f"{crop} - {disease}" if disease else f"{crop} - Unknown"))
        risk = str(result.get("risk_level", "unknown")).lower()
        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "all_predictions": [(prediction, round(confidence, 1))],
            "risk_level": risk,
            "crop_detected": crop,
            "disease_detected": disease,
            "reason": result.get("reason", ""),
            "uncertain": bool(result.get("uncertain", confidence < 50)),
            "is_demo": False,
        }

    def get_class_labels(self) -> List[str]:
        return list(CROP_DISEASE_LABELS)


class LocalPlantDiseaseModel(VisionModelInterface):
    """Real local PlantVillage classifier using ONNX Runtime."""
    MODEL_ID = os.getenv(
        "PLANT_DISEASE_MODEL_ID",
        "BiernyVR/crop-disease-classifier",
    )
    MODEL_FILENAME = os.getenv(
        "PLANT_DISEASE_MODEL_FILE",
        "efficientnet_v2_s_best.onnx",
    )
    MODEL_DATA_FILENAME = os.getenv(
        "PLANT_DISEASE_MODEL_DATA_FILE",
        "efficientnet_v2_s_best.onnx.data",
    )
    CLASSES_FILENAME = os.getenv(
        "PLANT_DISEASE_CLASSES_FILE",
        "classes.json",
    )

    def __init__(self):
        self.session = None
        self.classes: List[str] = []
        self.input_name = None
        self.load_error = None
        self._load()

    def _load(self):
        try:
            import json
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            token = os.getenv("HUGGINGFACE_TOKEN") or None

            model_path = hf_hub_download(
                repo_id=self.MODEL_ID,
                filename=self.MODEL_FILENAME,
                token=token,
            )
            # The model uses ONNX external data; both files must be present
            # in the same Hugging Face cache directory.
            hf_hub_download(
                repo_id=self.MODEL_ID,
                filename=self.MODEL_DATA_FILENAME,
                token=token,
            )
            classes_path = hf_hub_download(
                repo_id=self.MODEL_ID,
                filename=self.CLASSES_FILENAME,
                token=token,
            )

            with open(classes_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            classes = payload.get("classes") if isinstance(payload, dict) else payload
            if not isinstance(classes, list) or len(classes) != 38:
                raise RuntimeError("Plant disease class mapping is missing or invalid")

            session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"],
            )

            if not session.get_inputs():
                raise RuntimeError("Plant disease ONNX model has no input tensor")

            self.session = session
            self.classes = [str(value) for value in classes]
            self.input_name = session.get_inputs()[0].name

            logger.info(
                f"Loaded ONNX plant disease model {self.MODEL_ID} "
                f"with {len(self.classes)} classes"
            )
        except Exception as exc:
            self.load_error = str(exc)
            logger.exception("Real plant disease model could not be loaded")

    @staticmethod
    def _split_label(label: str) -> Dict:
        parts = label.split("___", 1)
        raw_crop = parts[0] if parts else "Unknown"
        raw_disease = parts[1] if len(parts) > 1 else "Unknown"

        crop = (
            raw_crop.replace("_(maize)", "")
            .replace("_", " ")
            .replace(", bell", "")
            .replace("  ", " ")
            .strip()
        )
        disease = raw_disease.replace("_", " ").strip()
        healthy = disease.lower() == "healthy"

        if healthy:
            disease = None

        return {
            "crop": crop,
            "disease": disease,
            "risk": "low" if healthy else "medium",
        }

    def predict(self, image: Image.Image) -> Dict:
        if self.session is None or not self.input_name or not self.classes:
            raise RuntimeError(
                "Real plant disease model is unavailable: "
                + (self.load_error or "unknown loading error")
            )

        img = (
            image.convert("RGB")
            .resize((224, 224), Image.Resampling.BILINEAR)
        )

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )
        tensor = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        logits = np.asarray(logits, dtype=np.float32)
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs /= max(float(probs.sum()), 1e-12)

        top_indices = np.argsort(probs)[::-1][:3]
        top_predictions = [
            (self.classes[int(index)], round(float(probs[int(index)]) * 100.0, 1))
            for index in top_indices
        ]

        label = top_predictions[0][0]
        confidence = top_predictions[0][1]
        meta = self._split_label(label)

        uncertain = confidence < 70.0
        disease = meta["disease"]
        prediction = f'{meta["crop"]} - {disease or "Healthy"}'
        reason = (
            f"EfficientNetV2-S image model top result: {prediction}. "
            f"Confidence {confidence:.1f}%."
        )

        if uncertain:
            disease = None
            prediction = f'{meta["crop"]} - Uncertain'
            reason += (
                " Confidence is below the safe threshold, so no disease is confirmed."
            )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "all_predictions": top_predictions,
            "risk_level": "unknown" if uncertain else meta["risk"],
            "crop_detected": meta["crop"] if confidence >= 50 else "Unknown",
            "disease_detected": disease,
            "reason": reason,
            "uncertain": uncertain,
            "is_demo": False,
            "model_name": self.MODEL_ID,
            "model_classes": len(self.classes),
        }

    def get_class_labels(self) -> List[str]:
        return list(self.classes)


class DemoVisionModel(VisionModelInterface):
    """Legacy compatibility fallback with NO random disease claims."""

    def __init__(self):
        self.labels = CROP_DISEASE_LABELS
        self.metadata = _CLASS_METADATA

    def predict(self, image: Image.Image) -> Dict:
        return {
            "prediction": "Unable to confirm disease",
            "confidence": 0.0,
            "all_predictions": [],
            "risk_level": "unknown",
            "crop_detected": "Unknown",
            "disease_detected": None,
            "reason": (
                "No real vision model is configured. The image cannot be safely "
                "identified without a trained plant-disease model."
            ),
            "uncertain": True,
            "is_demo": True,
        }

    def get_class_labels(self) -> List[str]:
        return list(self.labels)

