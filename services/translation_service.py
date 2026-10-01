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

        # Last-resort public translation service. Translate line-by-line so
        # Markdown headings, bullets, URLs and source formatting remain usable.
        try:
            return self._mymemory_translate(text, target_code, source_lang)
        except Exception as e:
            logger.warning(f"MyMemory translation failed: {e}")

        return {
            "translated_text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "is_fallback": True,
            "error": f"Translation to {target_lang} is temporarily unavailable.",
        }

    def _mymemory_translate(self, text: str, target: str, source: str) -> Dict:
        """Translate the complete response in one request to avoid request timeouts."""
        import re
        import requests

        if not text or not text.strip():
            return {
                "translated_text": text,
                "source_lang": source,
                "target_lang": target,
                "is_fallback": False,
                "error": None,
            }

        # Protect URLs so the translation service cannot alter source links.
        urls = []
        def protect_url(match):
            token = f"__KISAN_URL_{len(urls)}__"
            urls.append(match.group(0))
            return token

        protected = re.sub(r"https?://\\S+", protect_url, text)

        session = requests.Session()
        session.headers.update({"User-Agent": "KisanKiAwaz/1.0"})
        resp = session.get(
            "https://api.mymemory.translated.net/get",
            params={"q": protected[:4500], "langpair": f"{source}|{target}"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        translated = (data.get("responseData") or {}).get("translatedText")
        if not translated:
            raise RuntimeError("MyMemory returned no translated text")

        for i, url in enumerate(urls):
            translated = translated.replace(f"__KISAN_URL_{i}__", url)

        return {
            "translated_text": translated,
            "source_lang": source,
            "target_lang": target,
            "is_fallback": False,
            "error": None,
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
