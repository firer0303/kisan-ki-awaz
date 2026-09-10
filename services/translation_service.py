"""
Kisan Ki Awaz - Translation Service
=====================================
Handles translation of AI responses into the farmer's selected language.
"""
from typing import Dict, Optional

from loguru import logger


class TranslationService:
    """
    Translates text between languages.
    Uses googletrans (free) by default, with DeepL as optional upgrade.
    """

    # Language code mapping for googletrans
    LANG_MAP = {
        "ur": "ur",   # Urdu
        "sd": "sd",   # Sindhi (limited support)
        "pa": "pa",   # Punjabi
        "ps": "ps",   # Pashto
        "bal": "ur",  # Balochi -> translate via Urdu
        "en": "en",   # English
    }

    def __init__(self):
        self._translator = None
        logger.info("Translation service initialized")

    def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "en",
    ) -> Dict:
        """
        Translate text to the target language.

        Returns:
            Dict with keys:
            - translated_text: str
            - source_lang: str
            - target_lang: str
            - is_fallback: bool
            - error: str or None
        """
        if target_lang == source_lang or target_lang == "en":
            return {
                "translated_text": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "is_fallback": False,
                "error": None,
            }

        target_code = self.LANG_MAP.get(target_lang, "ur")

        # Try deep-translator first (more reliable)
        try:
            return self._deep_translate(text, target_code, source_lang)
        except Exception as e:
            logger.warning(f"deep-translator failed: {e}")

        # Fallback: googletrans
        try:
            return self._googletrans_translate(text, target_code, source_lang)
        except Exception as e:
            logger.warning(f"googletrans failed: {e}")

        # Final fallback: return English with notice
        return {
            "translated_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "is_fallback": True,
            "error": f"Translation to {target_lang} not available. Showing English response.",
        }

    def _deep_translate(
        self, text: str, target: str, source: str
    ) -> Dict:
        """Translate using deep-translator."""
        from deep_translator import GoogleTranslator

        translator = GoogleTranslator(source=source, target=target)
        result = translator.translate(text)

        return {
            "translated_text": result,
            "source_lang": source,
            "target_lang": target,
            "is_fallback": False,
            "error": None,
        }

    def _googletrans_translate(
        self, text: str, target: str, source: str
    ) -> Dict:
        """Translate using googletrans."""
        from googletrans import Translator

        if self._translator is None:
            self._translator = Translator()

        result = self._translator.translate(text, src=source, dest=target)

        return {
            "translated_text": result.text,
            "source_lang": source,
            "target_lang": target,
            "is_fallback": False,
            "error": None,
        }

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            from googletrans import Translator
            if self._translator is None:
                self._translator = Translator()
            result = self._translator.detect(text)
            return result.lang
        except Exception:
            return "unknown"
