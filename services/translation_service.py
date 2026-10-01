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
        # MyMemory enforces a hard maximum of 500 characters per query.
        # Translate each non-empty result line separately so the translated
        # response remains easy to understand sentence-by-sentence while all
        # sections (evidence, recommendations, warnings, sources, confidence)
        # are still translated.
        translated_lines = []
        for line in protected.splitlines(keepends=True):
            raw = line.rstrip("\r\n")
            newline = line[len(raw):]

            if not raw.strip():
                translated_lines.append(line)
                continue

            # Keep Markdown structure at the beginning of the line intact,
            # and translate the actual human-readable text after it.
            import re as _re
            prefix_match = _re.match(r"^(\s*(?:#{1,6}\s+|[-*]\s+|\d+[.)]\s+)?)", raw)
            prefix = prefix_match.group(1) if prefix_match else ""
            body = raw[len(prefix):]

            # A very long single line can still exceed the provider limit.
            body_parts = []
            remaining = body
            while remaining:
                if len(prefix) + len(remaining) <= 480:
                    body_parts.append(remaining)
                    break
                cut = remaining.rfind(" ", 0, 480 - len(prefix))
                if cut < 100:
                    cut = 480 - len(prefix)
                body_parts.append(remaining[:cut])
                remaining = remaining[cut:].lstrip()

            translated_parts = []
            for part in body_parts:
                request_text = prefix + part if not translated_parts else part
                resp = session.get(
                    "https://api.mymemory.translated.net/get",
                    params={"q": request_text, "langpair": f"{source}|{target}"},
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                translated_part = (data.get("responseData") or {}).get("translatedText")
                if not translated_part:
                    raise RuntimeError("MyMemory returned no translated text")
                translated_parts.append(translated_part)

            translated_line = "".join(translated_parts)
            if translated_line and not translated_line.startswith(prefix) and prefix:
                translated_line = prefix + translated_line

            # Punjabi in Kisan Ki Awaz means Pakistani Punjabi (Shahmukhi),
            # not Hindi/Devanagari. Reject Hindi output and use the normal
            # fallback chain rather than showing the wrong script.
            if target == "pa":
                devanagari_chars = sum(
                    1 for ch in translated_line
                    if "\u0900" <= ch <= "\u097f"
                )
                if devanagari_chars >= 3:
                    raise RuntimeError(
                        "Translation provider returned Hindi/Devanagari for Punjabi target"
                    )

            translated_lines.append(translated_line + newline)

        translated = "".join(translated_lines)

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
