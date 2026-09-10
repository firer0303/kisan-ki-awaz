"""
Kisan Ki Awaz - Recommendation Engine
========================================
Orchestrates the full analysis pipeline:
1. Retrieve evidence from RAG knowledge base
2. Analyze image (if provided)
3. Generate LLM response
4. Translate to farmer's language
5. Prepare narration
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
    """
    Central orchestrator for the full farmer query -> recommendation pipeline.
    """

    def __init__(self):
        # Core services
        self.language_service = LanguageService()
        self.rag_service = RAGService()
        self.vision_service = VisionService()
        self.image_service = ImageService()
        self.llm_service = LLMService()
        self.translation_service = TranslationService()
        self.speech_service = SpeechService(self.language_service)
        self.narration_service = NarrationService(
            self.speech_service, self.language_service
        )
        self.weather_service = WeatherService()
        self.market_service = MarketService()
        self.database_service = DatabaseService()

        # Session management
        self._session_id = uuid.uuid4().hex[:16]

        logger.info("Recommendation engine initialized")

    @property
    def session_id(self) -> str:
        return self._session_id

    def process_voice_query(self, text: str) -> Dict:
        """
        Process a text/voice query from the farmer.

        Pipeline:
        1. Retrieve evidence from knowledge base
        2. Build RAG prompt
        3. Generate LLM response
        4. Translate if needed
        5. Prepare narration
        6. Save to database
        """
        lang_code = self.language_service.current_language.value
        logger.info(f"Processing voice query (lang={lang_code}): {text[:100]}")

        # Step 1: Retrieve evidence
        evidence = self.rag_service.retrieve_evidence(text, top_k=3)

        # Step 2: Build RAG prompt
        prompt = self.rag_service.build_rag_prompt(
            farmer_question=text,
            evidence=evidence,
            language=lang_code,
        )

        # Step 3: Generate LLM response
        llm_result = self.llm_service.generate(prompt)
        response_text = llm_result.get("text", "Unable to generate response.")

        # Step 4: Translate if not English
        translation_info = {"translated_text": response_text, "is_fallback": False}
        if lang_code != "en":
            translation_info = self.translation_service.translate(
                response_text, target_lang=lang_code, source_lang="en"
            )
            if translation_info.get("translated_text"):
                response_text = translation_info["translated_text"]

        # Step 5: Prepare narration
        narration = self.narration_service.prepare_narration(response_text)
        narration_html = self.narration_service.get_autoplay_html(narration)

        # Step 6: Save to database
        self.database_service.save_analysis({
            "session_id": self._session_id,
            "language": lang_code,
            "input_type": "voice",
            "farmer_question": text,
            "confidence": 0,
            "risk_level": "info",
            "response_text": response_text,
            "sources_used": evidence.get("citations", []),
        })

        return {
            "success": True,
            "response_text": response_text,
            "sources": evidence.get("citations", []),
            "evidence_count": evidence.get("evidence_count", 0),
            "narration_html": narration_html,
            "narration": narration,
            "translation_info": translation_info,
            "llm_provider": llm_result.get("provider", "unknown"),
            "is_demo": llm_result.get("is_demo", False),
            "input_type": "voice",
        }

    def process_image(
        self,
        image: Image.Image,
        question: str = "",
        input_type: str = "image",
    ) -> Dict:
        """
        Process an image (upload or camera) for crop/disease analysis.

        Pipeline:
        1. Analyze image with vision model
        2. Retrieve relevant evidence
        3. Build RAG prompt with image results
        4. Generate LLM response
        5. Translate if needed
        6. Prepare narration
        7. Save to database
        """
        lang_code = self.language_service.current_language.value
        logger.info(f"Processing image (lang={lang_code}, type={input_type})")

        # Step 1: Vision analysis
        vision_result = self.vision_service.analyze_image(image)
        if not vision_result.get("success"):
            return {
                "success": False,
                "error": vision_result.get("error", "Image analysis failed"),
                "warnings": vision_result.get("warnings", []),
            }

        # Step 2: Build query for RAG retrieval
        crop = vision_result.get("crop_detected", "")
        disease = vision_result.get("disease_detected", "")
        rag_query = f"{crop} {disease}".strip() if disease else crop
        if question:
            rag_query = f"{rag_query} {question}"

        evidence = self.rag_service.retrieve_evidence(
            rag_query, crop=crop.lower() if crop else None, top_k=3
        )

        # Step 3: Build RAG prompt
        image_analysis = {
            "prediction": vision_result.get("prediction", "Unknown"),
            "confidence": vision_result.get("confidence", 0),
            "risk_level": vision_result.get("risk_level", "unknown"),
        }
        prompt = self.rag_service.build_rag_prompt(
            farmer_question=question or f"Analyze this {crop} image for disease/pest issues.",
            evidence=evidence,
            image_analysis=image_analysis,
            language=lang_code,
        )

        # Step 4: Generate LLM response
        llm_result = self.llm_service.generate(prompt)
        response_text = llm_result.get("text", "Unable to generate response.")

        # Step 5: Translate if not English
        translation_info = {"translated_text": response_text, "is_fallback": False}
        if lang_code != "en":
            translation_info = self.translation_service.translate(
                response_text, target_lang=lang_code, source_lang="en"
            )
            if translation_info.get("translated_text"):
                response_text = translation_info["translated_text"]

        # Step 6: Prepare narration
        narration = self.narration_service.prepare_narration(response_text)
        narration_html = self.narration_service.get_autoplay_html(narration)

        # Step 7: Save to database
        self.database_service.save_analysis({
            "session_id": self._session_id,
            "language": lang_code,
            "input_type": input_type,
            "farmer_question": question or f"Image analysis: {crop}",
            "crop_detected": crop,
            "disease_detected": disease,
            "confidence": vision_result.get("confidence", 0),
            "risk_level": vision_result.get("risk_level", "unknown"),
            "response_text": response_text,
            "sources_used": evidence.get("citations", []),
        })

        return {
            "success": True,
            "response_text": response_text,
            "vision_result": vision_result,
            "sources": evidence.get("citations", []),
            "evidence_count": evidence.get("evidence_count", 0),
            "narration_html": narration_html,
            "narration": narration,
            "translation_info": translation_info,
            "llm_provider": llm_result.get("provider", "unknown"),
            "is_demo": llm_result.get("is_demo", False) or vision_result.get("is_demo", False),
            "input_type": input_type,
            "image_quality": vision_result.get("image_quality", {}),
        }

    def get_explainability(
        self, result: Dict, question: str = ""
    ) -> Dict:
        """
        Generate explainability information for the "Why this recommendation?" feature.
        """
        factors = []

        # Image analysis factors
        vision = result.get("vision_result", {})
        if vision:
            factors.append({
                "category": "Image Analysis",
                "details": [
                    f"Detected: {vision.get('prediction', 'N/A')}",
                    f"Confidence: {vision.get('confidence', 0)}%",
                    f"Crop: {vision.get('crop_detected', 'N/A')}",
                    f"Disease: {vision.get('disease_detected', 'Not detected') or 'Not detected'}",
                    f"Risk Level: {vision.get('risk_level', 'N/A')}",
                ],
                "source_type": "AI Model",
            })

            quality = vision.get("image_quality", {})
            if quality:
                factors.append({
                    "category": "Image Quality",
                    "details": [
                        f"Resolution: {quality.get('resolution', 'N/A')}",
                        f"Brightness: {quality.get('brightness', 'N/A')}",
                        f"Contrast: {quality.get('contrast', 'N/A')}",
                        f"Blur detected: {'Yes' if quality.get('is_blurry') else 'No'}",
                        f"Dark image: {'Yes' if quality.get('is_dark') else 'No'}",
                    ],
                    "source_type": "Image Processing",
                })

        # Evidence factors
        sources = result.get("sources", [])
        if sources:
            for src in sources:
                factors.append({
                    "category": "Verified Evidence",
                    "details": [
                        f"Source: {src.get('organization', 'N/A')}",
                        f"Document: {src.get('document_title', 'N/A')}",
                        f"Published: {src.get('publication_date', 'N/A')}",
                        f"Credibility: {src.get('credibility', 'N/A')}",
                    ],
                    "source_type": "Knowledge Base (Verified)",
                })

        # Weather factors
        weather = self.weather_service.get_weather()
        factors.append({
            "category": "Weather Context",
            "details": [
                f"Location: {weather.get('location', 'N/A')}",
                f"Temperature: {weather.get('temperature_c', 'N/A')}°C",
                f"Humidity: {weather.get('humidity_pct', 'N/A')}%",
                f"Condition: {weather.get('condition', 'N/A')}",
            ],
            "source_type": "Weather Service" + (" (Demo)" if weather.get("is_demo") else ""),
        })

        # AI reasoning label
        ai_reasoning_notice = (
            "This recommendation combines AI inference (image analysis + LLM reasoning) "
            "with verified agricultural evidence from authoritative sources. "
            "AI-generated conclusions are clearly labeled and should be verified "
            "with local agricultural experts for critical decisions."
        )

        return {
            "factors": factors,
            "ai_reasoning_notice": ai_reasoning_notice,
            "total_sources": len(sources),
            "has_image_analysis": bool(vision),
            "is_demo": result.get("is_demo", False),
        }
