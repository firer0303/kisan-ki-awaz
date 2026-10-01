"""
Kisan Ki Awaz - Recommendation Engine
========================================
Orchestrates the full analysis pipeline.
"""
import uuid
from typing import Dict, Optional

from loguru import logger
from PIL import Image

from services.database_service import DatabaseService
from services.image_service import ImageService
from services.language_service import LanguageService
from services.llm_service import LLMService
from services.market_service import MarketService
from services.narration_service import NarrationService
from services.rag_service import RAGService
from services.speech_service import SpeechService
from services.translation_service import TranslationService
from services.vision_service import VisionService
from services.weather_service import WeatherService


class RecommendationEngine:
    """Central orchestrator for the farmer query -> recommendation pipeline."""

    def __init__(self):
        self.language_service = LanguageService()
        self.rag_service = RAGService()
        self.vision_service = VisionService()
        self.image_service = ImageService()
        self.llm_service = LLMService()
        self.translation_service = TranslationService()
        self.speech_service = SpeechService(self.language_service)
        self.narration_service = NarrationService(self.speech_service, self.language_service)
        self.weather_service = WeatherService()
        self.market_service = MarketService()
        self.database_service = DatabaseService()
        self._session_id = uuid.uuid4().hex[:16]
        logger.info("Recommendation engine initialized")

    @property
    def session_id(self) -> str:
        return self._session_id

    def _localize_response(self, response_text: str, lang_code: str) -> tuple:
        """Ensure the visible response is in the selected language and has native headings."""
        if not response_text:
            return response_text, {"translated_text": response_text, "is_fallback": False}

        translation_info = {
            "translated_text": response_text,
            "is_fallback": False,
            "source": "selected-language-generation",
        }

        # If the model returned English despite the selected language, translate it.
        english_markers = (
            "Based on your question", "## Recommendations", "## Warnings",
            "## Verified Sources", "Confidence Level", "Recommendations:",
        )
        needs_translation = lang_code != "en" and any(marker in response_text for marker in english_markers)
        if needs_translation:
            translated = self.translation_service.translate(
                response_text, target_lang=lang_code, source_lang="en"
            )
            if translated.get("translated_text"):
                response_text = translated["translated_text"]
                translation_info = translated

        # Hard-enforce section headings so the UI never shows English headings for a
        # non-English selected language, even if the model/translator leaves one behind.
        headings = {
            "ur": {
                "Assessment / Answer": "تشخیص / جواب",
                "Assessment/Answer": "تشخیص / جواب",
                "Recommendations": "سفارشات",
                "Warnings": "احتیاطی تدابیر",
                "Warnings / Precautions": "احتیاطی تدابیر",
                "Verified Sources": "تصدیق شدہ ذرائع",
                "Confidence Level": "اعتماد کی سطح",
            },
            "sd": {
                "Assessment / Answer": "جائزو / جواب",
                "Recommendations": "سفارشون",
                "Warnings": "احتياطي تدبيرون",
                "Warnings / Precautions": "احتياطي تدبيرون",
                "Verified Sources": "تصديق ٿيل ذريعا",
                "Confidence Level": "اعتماد جي سطح",
            },
            "pa": {
                "Assessment / Answer": "جائزہ / جواب",
                "Recommendations": "سفارشاں",
                "Warnings": "احتیاطی تدبیراں",
                "Warnings / Precautions": "احتیاطی تدبیراں",
                "Verified Sources": "تصدیق شدہ ذرائع",
                "Confidence Level": "اعتماد دی سطح",
            },
            "ps": {
                "Assessment / Answer": "ارزونه / ځواب",
                "Recommendations": "سپارښتنې",
                "Warnings": "احتیاطي تدابیر",
                "Warnings / Precautions": "احتیاطي تدابیر",
                "Verified Sources": "تایید شوې سرچینې",
                "Confidence Level": "د باور کچه",
            },
            "bal": {
                "Assessment / Answer": "جائزگ / جواب",
                "Recommendations": "سفارشاں",
                "Warnings": "احتیاطی تدبیر",
                "Warnings / Precautions": "احتیاطی تدبیر",
                "Verified Sources": "تصدیقءَ بوتگین سراجاݔں",
                "Confidence Level": "اعتمادءِ سطح",
            },
        }
        for english, native in headings.get(lang_code, {}).items():
            response_text = response_text.replace(english, native)

        return response_text, translation_info

    def process_voice_query(self, text: str) -> Dict:
        lang_code = self.language_service.current_language.value
        logger.info(f"Processing voice query (lang={lang_code}): {text[:100]}")
        evidence = self.rag_service.retrieve_evidence(text, top_k=3)
        prompt = self.rag_service.build_rag_prompt(text, evidence, language=lang_code)
        llm_result = self.llm_service.generate(prompt)
        response_text, translation_info = self._localize_response(
            llm_result.get("text", "Unable to generate response."), lang_code
        )
        narration = self.narration_service.prepare_narration(response_text)
        narration_html = self.narration_service.get_autoplay_html(narration)
        self.database_service.save_analysis({
            "session_id": self._session_id, "language": lang_code,
            "input_type": "voice", "farmer_question": text,
            "confidence": 0, "risk_level": "info", "response_text": response_text,
            "sources_used": evidence.get("citations", []),
        })
        return {
            "success": True, "response_text": response_text,
            "sources": evidence.get("citations", []),
            "evidence_count": evidence.get("evidence_count", 0),
            "narration_html": narration_html, "narration": narration,
            "translation_info": translation_info,
            "llm_provider": llm_result.get("provider", "unknown"),
            "is_demo": llm_result.get("is_demo", False), "input_type": "voice",
        }

    def process_image(self, image: Image.Image, question: str = "", input_type: str = "image") -> Dict:
        lang_code = self.language_service.current_language.value
        logger.info(f"Processing image (lang={lang_code}, type={input_type})")
        vision_result = self.vision_service.analyze_image(image)
        if not vision_result.get("success"):
            return {"success": False, "error": vision_result.get("error", "Image analysis failed"), "warnings": vision_result.get("warnings", [])}

        crop = vision_result.get("crop_detected", "")
        disease = vision_result.get("disease_detected", "")
        rag_query = f"{crop} {disease}".strip() if disease else crop
        if question:
            rag_query = f"{rag_query} {question}"
        evidence = self.rag_service.retrieve_evidence(rag_query, crop=crop.lower() if crop else None, top_k=3)
        image_analysis = {
            "prediction": vision_result.get("prediction", "Unknown"),
            "confidence": vision_result.get("confidence", 0),
            "risk_level": vision_result.get("risk_level", "unknown"),
        }
        prompt = self.rag_service.build_rag_prompt(
            question or f"Analyze this {crop} image for disease/pest issues.",
            evidence=evidence, image_analysis=image_analysis, language=lang_code,
        )
        llm_result = self.llm_service.generate(prompt)
        response_text, translation_info = self._localize_response(
            llm_result.get("text", "Unable to generate response."), lang_code
        )
        narration = self.narration_service.prepare_narration(response_text)
        narration_html = self.narration_service.get_autoplay_html(narration)
        self.database_service.save_analysis({
            "session_id": self._session_id, "language": lang_code,
            "input_type": input_type, "farmer_question": question or f"Image analysis: {crop}",
            "crop_detected": crop, "disease_detected": disease,
            "confidence": vision_result.get("confidence", 0),
            "risk_level": vision_result.get("risk_level", "unknown"),
            "response_text": response_text, "sources_used": evidence.get("citations", []),
        })
        return {
            "success": True, "response_text": response_text,
            "vision_result": vision_result, "sources": evidence.get("citations", []),
            "evidence_count": evidence.get("evidence_count", 0),
            "narration_html": narration_html, "narration": narration,
            "translation_info": translation_info,
            "llm_provider": llm_result.get("provider", "unknown"),
            "is_demo": llm_result.get("is_demo", False) or vision_result.get("is_demo", False),
            "input_type": input_type, "image_quality": vision_result.get("image_quality", {}),
        }

    def get_explainability(self, result: Dict, question: str = "") -> Dict:
        factors = []
        vision = result.get("vision_result", {})
        if vision:
            factors.append({"category": "Image Analysis", "details": [
                f"Detected: {vision.get('prediction', 'N/A')}",
                f"Confidence: {vision.get('confidence', 0)}%",
                f"Crop: {vision.get('crop_detected', 'N/A')}",
                f"Disease: {vision.get('disease_detected', 'Not detected') or 'Not detected'}",
                f"Risk Level: {vision.get('risk_level', 'N/A')}",
            ], "source_type": "AI Model"})
        sources = result.get("sources", [])
        if sources:
            for src in sources:
                factors.append({"category": "Verified Evidence", "details": [
                    f"Source: {src.get('organization', 'N/A')}",
                    f"Document: {src.get('document_title', 'N/A')}",
                    f"Published: {src.get('publication_date', 'N/A')}",
                    f"Credibility: {src.get('credibility', 'N/A')}",
                ], "source_type": "Knowledge Base (Verified)"})
        weather = self.weather_service.get_weather()
        factors.append({"category": "Weather Context", "details": [
            f"Location: {weather.get('location', 'N/A')}",
            f"Temperature: {weather.get('temperature_c', 'N/A')}°C",
            f"Humidity: {weather.get('humidity_pct', 'N/A')}%",
            f"Condition: {weather.get('condition', 'N/A')}",
        ], "source_type": "Weather Service" + (" (Demo)" if weather.get("is_demo") else "")})
        ai_reasoning_notice = (
            "This recommendation combines AI inference (image analysis + LLM reasoning) "
            "with verified agricultural evidence from authoritative sources. "
            "AI-generated conclusions are clearly labeled and should be verified "
            "with local agricultural experts for critical decisions."
        )
        return {"factors": factors, "ai_reasoning_notice": ai_reasoning_notice,
                "total_sources": len(sources), "has_image_analysis": bool(vision),
                "is_demo": result.get("is_demo", False)}
