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
        """Translate the entire response while preserving Markdown syntax."""
        import re
        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": "KisanKiAwaz/1.0"})

        # Preserve only structural Markdown markers. Everything human-readable,
        # including headings, labels, warnings and verified-source descriptions,
        # is sent for translation.
        def translate_line(line: str) -> str:
            if not line.strip():
                return line

            # Keep heading/bullet markers, but translate their actual text.
            m = re.match(r"^(\s*)(#{1,6}\s+|[-*]\s+|\d+\.\s+)?(.*)$", line)
            if not m:
                return line
            indent, marker, body = m.groups()
            if not body.strip():
                return line

            # Keep URLs unchanged, but translate surrounding text.
            parts = re.split(r"(https?://\\S+)", body)
            result = []
            for part in parts:
                if re.match(r"^https?://", part):
                    result.append(part)
                    continue
                if not part.strip():
                    result.append(part)
                    continue

                # MyMemory has a practical query-size limit.
                chunks = [part[i:i+450] for i in range(0, len(part), 450)]
                translated_chunks = []
                for chunk in chunks:
                    resp = session.get(
                        "https://api.mymemory.translated.net/get",
                        params={"q": chunk, "langpair": f"{source}|{target}"},
                        timeout=20,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    translated = (data.get("responseData") or {}).get("translatedText")
                    if not translated:
                        raise RuntimeError("MyMemory returned no translated text")
                    translated_chunks.append(translated)
                result.append("".join(translated_chunks))

            return indent + (marker or "") + "".join(result)

        translated_lines = [translate_line(line) for line in text.splitlines()]
        return {
            "translated_text": "\n".join(translated_lines),
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
