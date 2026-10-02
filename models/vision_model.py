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
    """Real local 38-class PlantVillage MobileNetV2 classifier."""
    MODEL_ID = os.getenv(
        "PLANT_DISEASE_MODEL_ID",
        "Daksh159/plant-disease-mobilenetv2",
    )
    MODEL_FILENAME = os.getenv(
        "PLANT_DISEASE_MODEL_FILE",
        "mobilenetv2_plant.pth",
    )

    # Standard PlantVillage 38-class label order used by the checkpoint.
    LABELS = [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry___Powdery_mildew",
        "Cherry___healthy",
        "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites_Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]

    def __init__(self):
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.load_error = None
        self._load()

    def _load(self):
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(
                repo_id=self.MODEL_ID,
                filename=self.MODEL_FILENAME,
                token=os.getenv("HUGGINGFACE_TOKEN") or None,
            )

            model = models.mobilenet_v2(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(in_features, len(self.LABELS)),
            )

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                state = (
                    checkpoint.get("state_dict")
                    or checkpoint.get("model_state_dict")
                    or checkpoint
                )
            else:
                state = checkpoint.state_dict()

            # Handle common training wrappers such as "module." or "model.".
            clean_state = {}
            for key, value in state.items():
                clean_key = str(key)
                for prefix in ("module.", "model."):
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                clean_state[clean_key] = value

            missing, unexpected = model.load_state_dict(clean_state, strict=False)
            if missing:
                raise RuntimeError(
                    f"Plant disease checkpoint is incomplete; missing {len(missing)} weights"
                )

            model.eval()
            self.model = model
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ])
            logger.info(
                f"Loaded real plant disease model {self.MODEL_ID}; "
                f"unexpected checkpoint keys={len(unexpected)}"
            )
        except Exception as exc:
            self.load_error = str(exc)
            logger.exception("Real plant disease model could not be loaded")

    @staticmethod
    def _split_label(label: str) -> Dict:
        parts = label.split("___", 1)
        crop = parts[0].replace("_(maize)", "").replace("_", " ").strip() if parts else "Unknown"
        disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
        healthy = disease.lower() == "healthy"
        if healthy:
            disease = None
        return {
            "crop": crop,
            "disease": disease,
            "risk": "low" if healthy else "medium",
        }

    def predict(self, image: Image.Image) -> Dict:
        if self.model is None or self.transform is None:
            raise RuntimeError(
                "Real plant disease model is unavailable: "
                + (self.load_error or "unknown loading error")
            )

        import torch

        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_idx = torch.topk(probs, k=min(3, len(self.LABELS)))

        top_predictions = []
        for probability, index in zip(top_probs.tolist(), top_idx.tolist()):
            label = self.LABELS[int(index)]
            top_predictions.append((label, round(float(probability) * 100, 1)))

        label = top_predictions[0][0]
        confidence = top_predictions[0][1]
        meta = self._split_label(label)

        # Closed-set models can be confidently wrong on field photos or
        # unsupported crops. Reject weak predictions instead of inventing a diagnosis.
        uncertain = confidence < 70.0
        disease = meta["disease"]
        prediction = f'{meta["crop"]} - {disease or "Healthy"}'
        reason = (
            f"Local MobileNetV2 PlantVillage model top result: {prediction}."
            f" Top confidence {confidence:.1f}%."
        )
        if uncertain:
            disease = None
            prediction = f'{meta["crop"]} - Uncertain'
            reason += " Confidence is below the safe diagnosis threshold, so no disease is confirmed."

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
            "model_classes": len(self.LABELS),
        }

    def get_class_labels(self) -> List[str]:
        return list(self.LABELS)


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

