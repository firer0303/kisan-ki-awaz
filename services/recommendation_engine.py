"""
Kisan Ki Awaz - Recommendation Engine
========================================
Orchestrates the full analysis pipeline.
"""
import uuid
from typing import Dict, Optional
from PIL import Image
from loguru import logger
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

    @property
    def session_id(self) -> str:
        return self._session_id

    def _localize_response(self, response_text: str, lang_code: str, fallback_prompt: str = "") -> tuple:
        """Always return formal output in the farmer's selected language.

        If an external translator is unavailable/rate-limited, regenerate the
        verified knowledge-base response directly in the selected language
        instead of returning English.
        """
        if not response_text or lang_code == "en":
            return response_text, {"translated_text": response_text, "is_fallback": False}

        translation_info = {"translated_text": response_text, "is_fallback": False}
        try:
            translated = self.translation_service.translate(
                response_text, target_lang=lang_code, source_lang="en"
            )
            if translated.get("translated_text") and not translated.get("is_fallback"):
                response_text = translated["translated_text"]
                translation_info = translated
            elif fallback_prompt:
                localized = self.llm_service._fallback_generate(fallback_prompt)
                if localized.get("text"):
                    response_text = localized["text"]
                    translation_info = {
                        "translated_text": response_text,
                        "source_lang": "en",
                        "target_lang": lang_code,
                        "is_fallback": True,
                        "error": translated.get("error"),
                    }
        except Exception as exc:
            logger.warning(f"Selected-language translation failed: {exc}")
            if fallback_prompt:
                try:
                    localized = self.llm_service._fallback_generate(fallback_prompt)
                    if localized.get("text"):
                        response_text = localized["text"]
                        translation_info = {
                            "translated_text": response_text,
                            "source_lang": "en",
                            "target_lang": lang_code,
                            "is_fallback": True,
                            "error": str(exc),
                        }
                except Exception as fallback_exc:
                    logger.warning(f"Localized fallback generation failed: {fallback_exc}")

        # Final UI-safe heading enforcement. This also works if the translator
        # leaves Markdown headings in English.
        headings = {
            "ur": {
                "Assessment / Answer": "تشخیص / جواب", "Assessment/Answer": "تشخیص / جواب",
                "Recommendations": "سفارشات", "Warnings": "احتیاطی تدابیر",
                "Warnings / Precautions": "احتیاطی تدابیر", "Verified Sources": "تصدیق شدہ ذرائع",
                "Confidence Level": "اعتماد کی سطح",
            },
            "sd": {
                "Assessment / Answer": "جائزو / جواب", "Recommendations": "سفارشون",
                "Warnings": "احتياطي تدبيرون", "Warnings / Precautions": "احتياطي تدبيرون",
                "Verified Sources": "تصديق ٿيل ذريعا", "Confidence Level": "اعتماد جي سطح",
            },
            "pa": {
                "Assessment / Answer": "جائزہ / جواب", "Recommendations": "سفارشاں",
                "Warnings": "احتیاطی تدبیراں", "Warnings / Precautions": "احتیاطی تدبیراں",
                "Verified Sources": "تصدیق شدہ ذرائع", "Confidence Level": "اعتماد دی سطح",
            },
            "ps": {
                "Assessment / Answer": "ارزونه / ځواب", "Recommendations": "سپارښتنې",
                "Warnings": "احتیاطي تدابیر", "Warnings / Precautions": "احتیاطي تدابیر",
                "Verified Sources": "تایید شوې سرچینې", "Confidence Level": "د باور کچه",
            },
            "bal": {
                "Assessment / Answer": "جائزگ / جواب", "Recommendations": "سفارشاں",
                "Warnings": "احتیاطی تدبیر", "Warnings / Precautions": "احتیاطی تدبیر",
                "Verified Sources": "تصدیقءَ بوتگین سراجاݔں", "Confidence Level": "اعتمادءِ سطح",
            },
        }
        for english, native in headings.get(lang_code, {}).items():
            response_text = response_text.replace(english, native)
            response_text = response_text.replace(f"## {english}", f"## {native}")
        return response_text, translation_info

    def _localize_result_metadata(self, items, lang_code: str):
        """Translate human-readable source metadata for the selected language."""
        if lang_code == "en" or not items:
            return items
        localized = []
        for item in items:
            if not isinstance(item, dict):
                localized.append(item)
                continue
            copy = dict(item)
            for key in ("document_title", "description", "credibility"):
                value = copy.get(key)
                if not value or not isinstance(value, str):
                    continue
                try:
                    result = self.translation_service.translate(value, target_lang=lang_code, source_lang="en")
                    if result.get("translated_text") and not result.get("is_fallback"):
                        copy[key] = result["translated_text"]
                except Exception as exc:
                    logger.warning(f"Metadata translation failed for {key}: {exc}")
            localized.append(copy)
        return localized

    def process_voice_query(self, text: str) -> Dict:
        lang_code = self.language_service.current_language.value
        evidence = self.rag_service.retrieve_evidence(text, top_k=3)
        localized_sources = self._localize_result_metadata(evidence.get("citations", []), lang_code)
        prompt = self.rag_service.build_rag_prompt(text, evidence, language=lang_code)
        llm_result = self.llm_service.generate(prompt)
        response_text, translation_info = self._localize_response(
            llm_result.get("text", "Unable to generate response."),
            lang_code,
            fallback_prompt=prompt,
        )
        narration = self.narration_service.prepare_narration(response_text)
        return {"success": True, "response_text": response_text, "sources": localized_sources, "evidence_count": evidence.get("evidence_count", 0), "narration_html": self.narration_service.get_autoplay_html(narration), "narration": narration, "translation_info": translation_info, "llm_provider": llm_result.get("provider", "unknown"), "is_demo": llm_result.get("is_demo", False), "input_type": "voice"}

    def process_image(self, image: Image.Image, question: str = "", input_type: str = "image") -> Dict:
        """Analyze an image and preserve the vision result if downstream services fail."""
        lang_code = self.language_service.current_language.value
        vision_result = self.vision_service.analyze_image(image)
        if not vision_result.get("success"):
            return {"success": False, "error": vision_result.get("error", "Image analysis failed"), "warnings": vision_result.get("warnings", [])}

        crop = vision_result.get("crop_detected", "") or "Unknown"
        disease = vision_result.get("disease_detected", "")
        rag_query = f"{crop} {disease}".strip() if disease else crop
        if question:
            rag_query = f"{rag_query} {question}"

        evidence = {"citations": [], "evidence_count": 0}
        llm_provider = "vision-only"
        is_demo = bool(vision_result.get("is_demo", False))
        try:
            evidence = self.rag_service.retrieve_evidence(rag_query, crop=crop.lower() if crop else None, top_k=3)
            image_analysis = {"prediction": vision_result.get("prediction", "Unknown"), "confidence": vision_result.get("confidence", 0), "risk_level": vision_result.get("risk_level", "unknown")}
            prompt = self.rag_service.build_rag_prompt(
                question or f"Analyze this {crop} image for disease/pest issues.",
                evidence=evidence, image_analysis=image_analysis, language=lang_code
            )
            llm_result = self.llm_service.generate(prompt)
            response_text, translation_info = self._localize_response(
                llm_result.get("text", ""), lang_code, fallback_prompt=prompt
            )
            localized_sources = self._localize_result_metadata(evidence.get("citations", []), lang_code)
            llm_provider = llm_result.get("provider", "unknown")
            is_demo = is_demo or bool(llm_result.get("is_demo", False))
        except Exception as exc:
            logger.exception("Image recommendation layer failed; returning vision-only result")
            response_text, translation_info = self._image_fallback_response(vision_result, lang_code, exc)

        narration = self.narration_service.prepare_narration(response_text)
        return {
            "success": True, "response_text": response_text, "vision_result": vision_result,
            "sources": localized_sources if 'localized_sources' in locals() else self._localize_result_metadata(evidence.get("citations", []), lang_code), "evidence_count": evidence.get("evidence_count", 0),
            "narration_html": self.narration_service.get_autoplay_html(narration), "narration": narration,
            "translation_info": translation_info, "llm_provider": llm_provider, "is_demo": is_demo,
            "input_type": input_type, "image_quality": vision_result.get("image_quality", {})
        }

    def _image_fallback_response(self, vision_result: Dict, lang_code: str, exc: Exception) -> tuple:
        """Localized safe answer when the RAG/LLM recommendation layer is unavailable."""
        confidence = float(vision_result.get("confidence", 0) or 0)
        prediction = vision_result.get("prediction") or "Unknown"
        disease = vision_result.get("disease_detected")
        reason = vision_result.get("reason") or ""
        templates = {
            "ur": f"تصویر کے AI تجزیے کے مطابق ممکنہ شناخت: {prediction}۔ اعتماد کی سطح {confidence:.0f} فیصد ہے۔ " + (f"ممکنہ بیماری یا مسئلہ: {disease}۔ " if disease else "بیماری کی واضح شناخت نہیں ہوئی۔ ") + (f"AI مشاہدہ: {reason}۔ " if reason else "") + "یہ ابتدائی AI تجزیہ ہے؛ بہتر نتیجے کے لیے متاثرہ پودے کی صاف اور قریب سے لی گئی تصویر استعمال کریں۔",
            "sd": f"تصوير جي AI تجزيي مطابق ممڪن سڃاڻپ: {prediction}. اعتماد جي سطح {confidence:.0f} سيڪڙو آهي. " + (f"ممڪن بيماري يا مسئلو: {disease}. " if disease else "بيماري جي واضح سڃاڻپ نه ٿي سگهي. ") + (f"AI مشاهدو: {reason}. " if reason else "") + "هي ابتدائي AI تجزيو آهي؛ بهتر نتيجي لاءِ متاثر ٿيل ٻوٽي جي صاف ويجهي تصوير استعمال ڪريو.",
            "pa": f"تصویر دے AI تجزیے مطابق ممکنہ شناخت: {prediction}۔ اعتماد دی سطح {confidence:.0f} فیصد اے۔ " + (f"ممکنہ بیماری یا مسئلہ: {disease}۔ " if disease else "بیماری دی واضح شناخت نہیں ہوئی۔ ") + (f"AI مشاہدہ: {reason}۔ " if reason else "") + "ایہہ ابتدائی AI تجزیہ اے؛ بہتر نتیجے لئی متاثرہ پودے دی صاف تے قریبوں تصویر ورتو۔",
            "ps": f"د انځور د AI تحلیل له مخې احتمالي پېژندنه: {prediction}. د باور کچه {confidence:.0f} سلنه ده. " + (f"احتمالي ناروغي یا ستونزه: {disease}. " if disease else "د ناروغۍ روښانه پېژندنه ونه شوه. ") + (f"د AI مشاهده: {reason}. " if reason else "") + "دا لومړنی AI تحلیل دی؛ د ښه پایلې لپاره د اغېزمن بوټي روښانه نږدې انځور وکاروئ.",
            "bal": f"عکسءِ AI تحلیلءِ حسابءَ ممکنہ شناخت: {prediction}. اعتمادءِ سطح {confidence:.0f} فیصد اَنت. " + (f"ممکنہ بیماری یا مسئلہ: {disease}. " if disease else "بیماریءَ واضح شناخت نہ بوتگ. ") + (f"AI مشاہدہ: {reason}. " if reason else "") + "اے ابتدائی AI تحلیل اَنت؛ بہتر نتیجےءِ وستی متاثرہ بوٹءِ صاف نزدیکین عکس استعمال کنیت.",
            "en": f"AI image analysis suggests: {prediction}. Confidence is {confidence:.0f}%. " + (f"Possible disease/problem: {disease}. " if disease else "No clear disease was identified. ") + (f"AI observation: {reason}. " if reason else "") + "This is a preliminary AI analysis; use a clear close-up image for a better result."
        }
        text = templates.get(lang_code, templates["en"])
        return text, {"translated_text": text, "is_fallback": True, "error": str(exc)}

    def get_explainability(self, result: Dict, question: str = "") -> Dict:
        factors = []
        vision = result.get("vision_result", {})
        if vision:
            factors.append({"category": "Image Analysis", "details": [f"Detected: {vision.get('prediction', 'N/A')}", f"Confidence: {vision.get('confidence', 0)}%", f"Crop: {vision.get('crop_detected', 'N/A')}", f"Disease: {vision.get('disease_detected', 'Not detected') or 'Not detected'}", f"Risk Level: {vision.get('risk_level', 'N/A')}"], "source_type": "AI Model"})
        sources = result.get("sources", [])
        for src in sources:
            factors.append({"category": "Verified Evidence", "details": [f"Source: {src.get('organization', 'N/A')}", f"Document: {src.get('document_title', 'N/A')}", f"Published: {src.get('publication_date', 'N/A')}", f"Credibility: {src.get('credibility', 'N/A')}"], "source_type": "Knowledge Base (Verified)"})
        weather = self.weather_service.get_weather()
        factors.append({"category": "Weather Context", "details": [f"Location: {weather.get('location', 'N/A')}", f"Temperature: {weather.get('temperature_c', 'N/A')}°C", f"Humidity: {weather.get('humidity_pct', 'N/A')}%", f"Condition: {weather.get('condition', 'N/A')}"], "source_type": "Weather Service"})
        return {"factors": factors, "total_sources": len(sources), "has_image_analysis": bool(vision), "is_demo": result.get("is_demo", False)}
