"""
Kisan Ki Awaz - Vision Model Interface
========================================
Agricultural image classification interface with an OpenAI vision backend
and a safe local fallback.
"""
import base64
import json
import random
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
from PIL import Image

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


class DemoVisionModel(VisionModelInterface):
    """Safe local fallback when no real vision API key is configured."""

    def __init__(self):
        self.labels = CROP_DISEASE_LABELS
        self.metadata = _CLASS_METADATA

    def predict(self, image: Image.Image) -> Dict:
        if image is None:
            return self._empty_result("No image provided")
        img_small = image.convert("RGB").resize((64, 64))
        pixels = np.array(img_small)
        avg_r, avg_g, avg_b = pixels[:, :, 0].mean(), pixels[:, :, 1].mean(), pixels[:, :, 2].mean()
        green_ratio = avg_g / (avg_r + avg_g + avg_b + 1)
        brown_indicator = avg_r > 120 and avg_g < 100 and avg_b < 80
        img_hash = hash(pixels.tobytes()[:100]) % 1000
        random.seed(img_hash)
        if green_ratio > 0.38 and not brown_indicator:
            top_label = random.choice([l for l in self.labels if "Healthy" in l])
            confidence = random.uniform(65, 92)
        elif brown_indicator:
            candidates = [l for l in self.labels if "Healthy" not in l and any(k in l for k in ["Rust", "Blight", "Rot", "Spot"])]
            top_label, confidence = random.choice(candidates), random.uniform(45, 78)
        else:
            top_label, confidence = random.choice([l for l in self.labels if "Healthy" not in l]), random.uniform(40, 75)
        meta = self.metadata[top_label]
        return {
            "prediction": top_label, "confidence": round(confidence, 1),
            "all_predictions": [(top_label, round(confidence, 1))],
            "risk_level": meta["risk_level"], "crop_detected": meta["crop"],
            "disease_detected": meta["disease"], "is_demo": True,
        }

    def _empty_result(self, reason: str) -> Dict:
        return {"prediction": "Unable to analyze", "confidence": 0, "all_predictions": [],
                "risk_level": "unknown", "crop_detected": "Unknown", "disease_detected": None,
                "error": reason, "is_demo": True}

    def get_class_labels(self) -> List[str]:
        return list(self.labels)
