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

    def _repair_punjabi_shahmukhi(self, text: str) -> str:
        """Use the configured LLM to convert Punjabi/Hindi/Gurmukhi output to Pakistani Shahmukhi."""
        if not text:
            return text
        prompt = (
            "Convert the following farmer answer into Pakistani Punjabi written ONLY in Shahmukhi script. "
            "Preserve every fact, recommendation, dosage, timing, number, crop name, disease name, source name, "
            "and URL. Do not summarize or remove information. Do not use Hindi, Devanagari, or Gurmukhi. "
            "Do not write English prose. Keep necessary scientific Latin names only when they cannot be translated. "
            "Return one smooth paragraph with no headings or bullet points.\n\n"
            f"Text to convert:\n{text}"
        )
        try:
            result = self.llm_service.generate(prompt, max_tokens=2000, temperature=0.1)
            repaired = (result.get("text") or "").strip()
            if repaired and self.translation_service.is_valid_selected_language_output(repaired, "pa"):
                return repaired
        except Exception as exc:
            logger.warning(f"Punjabi Shahmukhi repair failed: {exc}")
        return ""

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
            candidate = translated.get("translated_text", "")
            valid = bool(candidate) and not translated.get("is_fallback")

            # Translation services often return Indian Punjabi/Hindi in an
            # Indic script. Pakistani Punjabi must be Shahmukhi.
            if lang_code == "pa" and valid:
                valid = self.translation_service.is_valid_selected_language_output(candidate, "pa")

            if valid:
                response_text = candidate
                translation_info = translated
            elif lang_code == "pa":
                repaired = self._repair_punjabi_shahmukhi(response_text)
                if repaired:
                    response_text = repaired
                    translation_info = {
                        "translated_text": response_text,
                        "source_lang": "en",
                        "target_lang": "pa",
                        "is_fallback": True,
                        "error": translated.get("error") or "Punjabi script repaired by LLM",
                    }
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
            if lang_code == "pa":
                repaired = self._repair_punjabi_shahmukhi(response_text)
                if repaired:
                    response_text = repaired
                    translation_info = {
                        "translated_text": response_text,
                        "source_lang": "en",
                        "target_lang": "pa",
                        "is_fallback": True,
                        "error": str(exc),
                    }
            elif fallback_prompt:
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
            "pa-hi": {
                "Assessment / Answer": "ਜਵਾਬ", "Recommendations": "ਕੀ ਕਰਨਾ ਹੈ",
                "Warnings": "ਸਾਵਧਾਨੀ", "Warnings / Precautions": "ਸਾਵਧਾਨੀ",
                "Verified Sources": "ਭਰੋਸੇਯੋਗ ਸਰੋਤ", "Confidence Level": "ਭਰੋਸੇ ਦਾ ਪੱਧਰ",
            },
            "hi": {
                "Assessment / Answer": "जवाब", "Recommendations": "क्या करें",
                "Warnings": "सावधानी", "Warnings / Precautions": "सावधानी",
                "Verified Sources": "भरोसेमंद स्रोत", "Confidence Level": "विश्वास स्तर",
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
        """Translate all source metadata in one request to minimize latency."""
        if lang_code == "en" or not items:
            return items

        localized = [dict(item) if isinstance(item, dict) else item for item in items]
        fields = []
        for item_index, item in enumerate(localized):
            if not isinstance(item, dict):
                continue
            for key in ("document_title", "description", "credibility"):
                value = item.get(key)
                if value and isinstance(value, str):
                    fields.append((item_index, key, value))

        if not fields:
            return localized

        try:
            marker_text = "\n".join(
                f"__KISAN_FIELD_{i}__ {value}" for i, (_, _, value) in enumerate(fields)
            )
            result = self.translation_service.translate(
                marker_text, target_lang=lang_code, source_lang="en"
            )
            translated = result.get("translated_text", "")
            if translated and not result.get("is_fallback"):
                for i, (item_index, key, _) in enumerate(fields):
                    marker = f"__KISAN_FIELD_{i}__"
                    if marker not in translated:
                        continue
                    value = translated.split(marker, 1)[1].split("__KISAN_FIELD_", 1)[0].strip()
                    if value:
                        localized[item_index][key] = value
        except Exception as exc:
            logger.warning(f"Metadata translation failed: {exc}")

        return localized

    def _localize_vision_result(self, vision: Dict, lang_code: str) -> Dict:
        """Translate all human-readable vision fields with one request."""
        if lang_code == "en" or not isinstance(vision, dict):
            return dict(vision)

        localized = dict(vision)
        risk_map = {
            "low": {"ur": "کم", "sd": "گهٽ", "pa": "گھٹ", "ps": "ټیټ", "bal": "کم"},
            "medium": {"ur": "درمیانہ", "sd": "وچولو", "pa": "درمیانہ", "ps": "منځنی", "bal": "درمیان"},
            "high": {"ur": "زیادہ", "sd": "وڌيڪ", "pa": "زیادہ", "ps": "لوړ", "bal": "زیادہ"},
            "unknown": {"ur": "نامعلوم", "sd": "نامعلوم", "pa": "نامعلوم", "ps": "ناڅرګند", "bal": "نامعلوم"},
        }
        risk = str(localized.get("risk_level", "")).lower()
        if risk in risk_map:
            localized["risk_level"] = risk_map[risk].get(lang_code, localized["risk_level"])

        fields = []
        for key in ("prediction", "crop_detected", "disease_detected", "reason"):
            value = localized.get(key)
            if isinstance(value, str) and value.strip() and value.lower() not in {"unknown", "none", "n/a"}:
                fields.append((key, value))

        warnings = localized.get("warnings") or []
        for idx, warning in enumerate(warnings):
            if isinstance(warning, str) and warning.strip():
                fields.append((f"warning_{idx}", warning))

        if not fields:
            return localized

        try:
            marker_text = "\n".join(
                f"__KISAN_VISION_{i}__ {value}" for i, (_, value) in enumerate(fields)
            )
            result = self.translation_service.translate(
                marker_text, target_lang=lang_code, source_lang="en"
            )
            translated = result.get("translated_text", "")
            if translated and not result.get("is_fallback"):
                for i, (key, _) in enumerate(fields):
                    marker = f"__KISAN_VISION_{i}__"
                    if marker not in translated:
                        continue
                    value = translated.split(marker, 1)[1].split("__KISAN_VISION_", 1)[0].strip()
                    if not value:
                        continue
                    if key.startswith("warning_"):
                        warnings[int(key.split("_", 1)[1])] = value
                    else:
                        localized[key] = value
                localized["warnings"] = warnings
        except Exception as exc:
            logger.warning(f"Vision field localization failed: {exc}")

        return localized

    def _detect_market_crop(self, text: str, evidence: Dict) -> str:
        """Detect a crop for live market-rate lookup."""
        q = (text or "").lower()
        aliases = {
            "wheat": ["wheat", "گندم", "गेहूं"],
            "rice": ["rice", "چاول", "چاوَل"],
            "cotton": ["cotton", "کپاس"],
            "sugarcane": ["sugarcane", "گنا"],
        }
        for crop, words in aliases.items():
            if any(word in q for word in words):
                return crop
        for doc in evidence.get("retrieved_docs", []) or []:
            crop = getattr(doc, "crop", None)
            if crop and str(crop).lower() in aliases:
                return str(crop).lower()
        return ""

    def _market_section(self, market: Dict, lang_code: str) -> str:
        """Build a simple selected-language live-price section."""
        prices = market.get("prices") or []
        if not prices:
            return ""

        headings = {
            "ur": "## تازہ منڈی ریٹ\n\n",
            "sd": "## تازو منڊي اگهه\n\n",
            "pa": "## تازہ منڈی ریٹ\n\n",
            "ps": "## د منډۍ تازه بیه\n\n",
            "bal": "## تازہ منڈی ریٹ\n\n",
            "en": "## Latest Market Rate\n\n",
        }
        lines = [headings.get(lang_code, headings["en"])]
        labels = {
            "ur": ("منڈی", "قیمت", "اوسط"),
            "sd": ("منڊي", "قيمت", "اوسط"),
            "pa": ("منڈی", "ریٹ", "اوسط"),
            "ps": ("منډۍ", "بیه", "اوسط"),
            "bal": ("منڈی", "ریٹ", "اوسط"),
            "en": ("Market", "Price", "Average"),
        }
        market_label, price_label, avg_label = labels.get(lang_code, labels["en"])
        for item in prices[:8]:
            market_name = item.get("market", "")
            price = item.get("price", "")
            avg = item.get("average", "")
            unit = item.get("unit", "")
            date = item.get("date", "")
            line = f"- {market_label}: {market_name} — {price}"
            if avg:
                line += f" ({avg_label}: {avg})"
            if unit:
                unit_text = unit
                if lang_code == "ur":
                    unit_text = unit_text.replace("100kg", "100 کلوگرام").replace("40kg", "40 کلوگرام").replace("kg", "کلوگرام")
                elif lang_code == "sd":
                    unit_text = unit_text.replace("100kg", "100 ڪلوگرام").replace("40kg", "40 ڪلوگرام").replace("kg", "ڪلوگرام")
                elif lang_code == "pa":
                    unit_text = unit_text.replace("100kg", "100 کلوگرام").replace("40kg", "40 کلوگرام").replace("kg", "کلوگرام")
                elif lang_code == "ps":
                    unit_text = unit_text.replace("100kg", "۱۰۰ کیلوګرام").replace("40kg", "۴۰ کیلوګرام").replace("kg", "کیلوګرام")
                elif lang_code == "bal":
                    unit_text = unit_text.replace("100kg", "100 کلوگرام").replace("40kg", "40 کلوگرام").replace("kg", "کلوگرام")
                line += f" / {unit_text}"
            if date:
                line += f" — {date}"
            lines.append(line + "\n")
        source = market.get("source", "")
        if source:
            if lang_code == "ur":
                lines.append(f"\nمعلومات کا ذریعہ: {source}۔ یہ تازہ دستیاب منڈی ریٹ ہے، لیکن آپ کی مقامی منڈی میں قیمت کچھ مختلف ہو سکتی ہے۔\n")
            elif lang_code == "sd":
                lines.append(f"\nذريعو: {source}. هي تازو دستياب اگهه آهي؛ مقامي منڊي ۾ ٿورو فرق ٿي سگهي ٿو.\n")
            elif lang_code == "pa":
                lines.append(f"\nماخذ: {source}۔ ایہہ تازہ دستیاب ریٹ اے؛ مقامی منڈی وچ تھوڑا فرق ہو سکدا اے۔\n")
            elif lang_code == "ps":
                lines.append(f"\nسرچینه: {source}. دا تازه موجود نرخ دی؛ په محلي منډۍ کې لږ توپیر کېدای شي.\n")
            elif lang_code == "bal":
                lines.append(f"\nماخذ: {source}. اے تازہ دسترس ءَ بوتگین ریٹ اَنت؛ مقامی منڈیءَ ءَ کم بیش فرق بوتگ.\n")
            else:
                lines.append(f"\nSource: {source}. This is the latest available rate; local mandi prices can vary.\n")
        return "".join(lines)

    def _to_single_paragraph(self, text: str) -> str:
        """Convert structured answer text into one smooth readable paragraph."""
        import re

        if not text:
            return text

        lines = [line.strip() for line in str(text).replace("\r", "").split("\n") if line.strip()]
        parts = []

        for line in lines:
            # Convert headings into inline labels.
            line = re.sub(r"^#{1,6}\s*", "", line)

            # Remove Markdown list markers while keeping the information.
            line = re.sub(r"^[-*]\s+", "", line)
            line = re.sub(r"^\d+[.)]\s+", "", line)

            # Turn Markdown emphasis into plain text.
            line = line.replace("**", "").replace("__", "")

            # Make section labels natural inside the paragraph.
            line = re.sub(r"^Assessment / Answer\s*:?\s*", "", line, flags=re.I)
            line = re.sub(r"^Recommendations\s*:?\s*", "", line, flags=re.I)
            line = re.sub(r"^Warnings / Precautions\s*:?\s*", "", line, flags=re.I)
            line = re.sub(r"^Verified Sources\s*:?\s*", "", line, flags=re.I)
            line = re.sub(r"^Confidence Level\s*:?\s*", "", line, flags=re.I)

            line = re.sub(r"\s+", " ", line).strip()
            if line:
                parts.append(line)

        result = " ".join(parts)
        # Avoid accidental duplicated punctuation from joining list items.
        result = re.sub(r"\s+([،۔,:;!?])", r"\1", result)
        result = re.sub(r"([۔!?])\s*\1+", r"\1", result)
        return result.strip()

    def process_voice_query(self, text: str) -> Dict:
        lang_code = self.language_service.current_language.value
        evidence = self.rag_service.retrieve_evidence(text, top_k=3)
        prompt = self.rag_service.build_rag_prompt(text, evidence, language=lang_code)
        market_crop = self._detect_market_crop(text, evidence)

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            source_future = executor.submit(
                self._localize_result_metadata, evidence.get("citations", []), lang_code
            )
            llm_future = executor.submit(self.llm_service.generate, prompt)
            market_future = executor.submit(
                self.market_service.get_crop_prices, market_crop
            ) if market_crop else None

            localized_sources = source_future.result()
            llm_result = llm_future.result()
            market = market_future.result() if market_future else {}

        raw_response = llm_result.get("text", "Unable to generate response.")
        already_localized = bool(
            llm_result.get("localized")
            and llm_result.get("language") == lang_code
        )
        if already_localized:
            already_localized = self.translation_service.is_valid_selected_language_output(
                raw_response, lang_code
            )

        if already_localized:
            response_text = raw_response
            translation_info = {
                "translated_text": response_text,
                "source_lang": "en",
                "target_lang": lang_code,
                "is_fallback": False,
            }
        else:
            response_text, translation_info = self._localize_response(
                raw_response,
                lang_code,
                fallback_prompt=prompt,
            )

        # Add live market information as a separate, easy-to-read section
        # after the translated answer so numeric price data is never lost.
        market_text = self._market_section(market, lang_code)
        if market_text:
            response_text = response_text.rstrip() + "\n\n" + market_text

        response_text = self._to_single_paragraph(response_text)

        narration = self.narration_service.prepare_narration(response_text)
        return {
            "success": True,
            "response_text": response_text,
            "sources": localized_sources,
            "evidence_count": evidence.get("evidence_count", 0),
            "market": market,
            "narration_html": self.narration_service.get_autoplay_html(narration),
            "narration": narration,
            "translation_info": translation_info,
            "llm_provider": llm_result.get("provider", "unknown"),
            "is_demo": bool(llm_result.get("is_demo", False)) or bool(market.get("is_demo", False)),
            "input_type": "voice",
        }

    def process_image(self, image: Image.Image, question: str = "", input_type: str = "image") -> Dict:
        """Analyze an image and preserve the vision result if downstream services fail."""
        lang_code = self.language_service.current_language.value
        vision_result = self.vision_service.analyze_image(image)
        vision_result = self._localize_vision_result(vision_result, lang_code)
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
            raw_image_response = llm_result.get("text", "")
            image_already_localized = bool(
                llm_result.get("localized")
                and llm_result.get("language") == lang_code
            )
            if image_already_localized:
                image_already_localized = self.translation_service.is_valid_selected_language_output(
                    raw_image_response, lang_code
                )
            if image_already_localized:
                response_text = raw_image_response
                translation_info = {
                    "translated_text": response_text,
                    "source_lang": "en",
                    "target_lang": lang_code,
                    "is_fallback": False,
                }
            else:
                response_text, translation_info = self._localize_response(
                    raw_image_response, lang_code, fallback_prompt=prompt
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
