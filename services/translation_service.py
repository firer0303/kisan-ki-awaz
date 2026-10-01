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

    def _localize_technical_terms(self, text: str, target_lang: str) -> str:
        """Convert common technical/scientific terms into farmer-friendly local wording."""
        if not text or target_lang == "en":
            return text

        terms = {
            "ur": {
                "Puccinia triticina": "گندم کے پتوں کا زنگ پیدا کرنے والی پھپھوندی",
                "Wheat leaf rust": "گندم کے پتوں کا زنگ",
                "wheat leaf rust": "گندم کے پتوں کا زنگ",
                "circular to oval": "گول یا بیضوی",
                "circular": "گول",
                "oval": "بیضوی",
                "pustules": "چھوٹے ابھرے ہوئے دانے",
                "orange-brown pustules": "نارنجی بھورے چھوٹے ابھرے ہوئے دانے",
                "spores": "جرثومی ذرات",
                "fungus": "پھپھوندی",
                "fungal disease": "پھپھوندی کی بیماری",
                "humidity": "نمی",
                "high humidity": "زیادہ نمی",
                "management": "بچاؤ اور قابو پانے کے طریقے",
                "precautions": "احتیاطی تدابیر",
                "temperature": "درجہ حرارت",
                "confidence": "اعتماد کی سطح",
                "risk level": "خطرے کی سطح",
            },
            "sd": {
                "Puccinia triticina": "ڪڻڪ جي پنن تي زنگ پيدا ڪندڙ ڦڦوند",
                "Wheat leaf rust": "ڪڻڪ جي پنن جو زنگ",
                "wheat leaf rust": "ڪڻڪ جي پنن جو زنگ",
                "circular to oval": "گول يا بيضوي",
                "circular": "گول",
                "oval": "بيضوي",
                "pustules": "ننڍڙا اڀريل داغ",
                "orange-brown pustules": "نارنگي ڀورا ننڍڙا اڀريل داغ",
                "spores": "جراثيمي ذرا",
                "fungus": "ڦڦوند",
                "fungal disease": "ڦڦوند جي بيماري",
                "humidity": "نمي",
                "high humidity": "وڌيڪ نمي",
                "management": "بچاءُ ۽ ڪنٽرول جا طريقا",
                "precautions": "احتياطي تدبيرون",
                "temperature": "گرمي پد",
                "confidence": "اعتماد جي سطح",
                "risk level": "خطري جي سطح",
            },
            "pa": {
                "Puccinia triticina": "گندم دے پتیاں دا زنگ پیدا کرن والی پھپھوندی",
                "Wheat leaf rust": "گندم دے پتیاں دا زنگ",
                "wheat leaf rust": "گندم دے پتیاں دا زنگ",
                "circular to oval": "گول یا بیضوی",
                "circular": "گول",
                "oval": "بیضوی",
                "pustules": "چھوٹے ابھرے ہوئے دانے",
                "orange-brown pustules": "نارنجی بھورے چھوٹے ابھرے ہوئے دانے",
                "spores": "جراثیمی ذرے",
                "fungus": "پھپھوندی",
                "fungal disease": "پھپھوندی دی بیماری",
                "humidity": "نمی",
                "high humidity": "زیادہ نمی",
                "management": "بچاؤ تے قابو پاؤن دے طریقے",
                "precautions": "احتیاطی تدبیراں",
                "temperature": "درجہ حرارت",
                "confidence": "اعتماد دی سطح",
                "risk level": "خطرے دی سطح",
            },
            "ps": {
                "Puccinia triticina": "هغه فنګس چې د غنمو د پاڼو زنګ رامنځته کوي",
                "Wheat leaf rust": "د غنمو د پاڼو زنګ",
                "wheat leaf rust": "د غنمو د پاڼو زنګ",
                "circular to oval": "ګرد یا بیضوي",
                "circular": "ګرد",
                "oval": "بیضوي",
                "pustules": "کوچني پورته راوتلي داغونه",
                "orange-brown pustules": "نارنجي نسواري کوچني پورته راوتلي داغونه",
                "spores": "جرثومي ذرات",
                "fungus": "فنګس",
                "fungal disease": "فنګسي ناروغي",
                "humidity": "لندبل",
                "high humidity": "لوړ لندبل",
                "management": "د مخنیوي او کنټرول لارې",
                "precautions": "احتیاطي تدابیر",
                "temperature": "تودوخه",
                "confidence": "د باور کچه",
                "risk level": "د خطر کچه",
            },
            "bal": {
                "Puccinia triticina": "گندمءِ پت ءِ زنگ پیدا کنوک پھپھوند",
                "Wheat leaf rust": "گندمءِ پت ءِ زنگ",
                "wheat leaf rust": "گندمءِ پت ءِ زنگ",
                "circular to oval": "گول یا بیضوی",
                "circular": "گول",
                "oval": "بیضوی",
                "pustules": "کوچک اُبھرین داغ",
                "orange-brown pustules": "نارنجی بھورے اُبھرین داغ",
                "spores": "جرثومی ذرات",
                "fungus": "پھپھوند",
                "fungal disease": "پھپھوندءِ بیماری",
                "humidity": "نمی",
                "high humidity": "زیادہ نمی",
                "management": "بچاؤ ءُ قابو ءِ طریقہ",
                "precautions": "احتیاطی تدبیر",
                "temperature": "درجہ حرارت",
                "confidence": "اعتمادءِ سطح",
                "risk level": "خطرہ ءِ سطح",
            },
        }

        mapping = terms.get(target_lang, {})
        for source_term in sorted(mapping, key=len, reverse=True):
            text = text.replace(source_term, mapping[source_term])
        return text

    def _validate_selected_language(self, translated: str, target_lang: str) -> None:
        """Reject clear language/script leakage without unnecessary retries."""
        import re
        if not translated or target_lang == "en":
            return

        cleaned = re.sub(r"https?://\S+", "", translated)
        cleaned = re.sub(r"__KISAN_[A-Z0-9_]+__", "", cleaned)

        # Latin-script technical/proper names may legitimately remain, but a
        # large amount of lowercase English means the translation failed.
        latin_words = re.findall(r"\b[A-Za-z][A-Za-z'-]{2,}\b", cleaned)
        lowercase_words = [w for w in latin_words if not w.isupper()]
        allowed = {"ai", "fao", "pmd", "amis", "pbc"}
        leaked = [w for w in lowercase_words if w.lower() not in allowed]

        if len(leaked) >= 12:
            raise RuntimeError(
                f"Significant English text detected in {target_lang} translation"
            )

        if target_lang == "pa":
            devanagari = sum(1 for ch in translated if "\u0900" <= ch <= "\u097f")
            if devanagari >= 3:
                raise RuntimeError(
                    "Punjabi translation returned Hindi/Devanagari script"
                )

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
            result = self._deep_translate(text, target_code, source_lang)
            result["translated_text"] = self._localize_technical_terms(result.get("translated_text", ""), target_lang)
            self._validate_selected_language(result.get("translated_text", ""), target_lang)
            return result
        except Exception as e:
            logger.warning(f"deep-translator failed: {e}")

        # Fallback: googletrans
        try:
            result = self._googletrans_translate(text, target_code, source_lang)
            result["translated_text"] = self._localize_technical_terms(result.get("translated_text", ""), target_lang)
            self._validate_selected_language(result.get("translated_text", ""), target_lang)
            return result
        except Exception as e:
            logger.warning(f"googletrans failed: {e}")

        # Last-resort public translation service. Translate line-by-line so
        # Markdown headings, bullets, URLs and source formatting remain usable.
        try:
            result = self._mymemory_translate(text, target_code, source_lang)
            result["translated_text"] = self._localize_technical_terms(result.get("translated_text", ""), target_lang)
            self._validate_selected_language(result.get("translated_text", ""), target_lang)
            return result
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
        # Translate complete paragraphs / sentences as units so the result
        # reads naturally, while splitting only when a paragraph is too long.
        # This keeps every required fact instead of translating only headings.
        import re as _re

        def translate_chunk(chunk: str) -> str:
            if not chunk or not chunk.strip():
                return chunk

            # Split only when needed, preferably at sentence boundaries.
            parts = []
            remaining = chunk.strip()
            while remaining:
                if len(remaining) <= 480:
                    parts.append(remaining)
                    break

                window = remaining[:480]
                cut = -1
                matches = list(_re.finditer(r"[.!?۔؟](?:\s+|$)", window))
                if matches:
                    cut = matches[-1].end()

                if cut < 180:
                    space = window.rfind(" ")
                    cut = space if space >= 120 else 480

                parts.append(remaining[:cut].strip())
                remaining = remaining[cut:].strip()

            out = []
            for part in parts:
                resp = session.get(
                    "https://api.mymemory.translated.net/get",
                    params={"q": part, "langpair": f"{source}|{target}"},
                    timeout=5,
                )
                resp.raise_for_status()
                data = resp.json()
                translated_part = (data.get("responseData") or {}).get("translatedText")
                if not translated_part:
                    raise RuntimeError("MyMemory returned no translated text")

                # Punjabi in Kisan Ki Awaz means Pakistani Punjabi/Shahmukhi.
                # Do not accept Hindi/Devanagari output.
                if target == "pa":
                    devanagari_chars = sum(
                        1 for ch in translated_part
                        if "\u0900" <= ch <= "\u097f"
                    )
                    if devanagari_chars >= 3:
                        raise RuntimeError(
                            "Translation provider returned Hindi/Devanagari for Punjabi target"
                        )

                out.append(translated_part.strip())

            return " ".join(out)

        # Preserve document structure, but translate actual prose in natural
        # paragraph-sized units. Headings, bullets, and numbered items retain
        # their Markdown markers; URLs were already protected above.
        blocks = _re.split(r"(\n\s*\n+)", protected)
        translated_blocks = []

        for block in blocks:
            if not block.strip():
                translated_blocks.append(block)
                continue

            lines = block.splitlines(keepends=True)
            current_paragraph = []

            def flush_paragraph():
                if not current_paragraph:
                    return
                paragraph = "".join(current_paragraph).strip()
                if paragraph:
                    translated_blocks.append(translate_chunk(paragraph))
                current_paragraph.clear()

            for line in lines:
                raw = line.rstrip("\r\n")
                if not raw.strip():
                    flush_paragraph()
                    translated_blocks.append(line)
                    continue

                # Markdown heading: translate it as a whole heading.
                heading = _re.match(r"^(\s*#{1,6}\s+)(.+?)\s*$", raw)
                if heading:
                    flush_paragraph()
                    translated_heading = translate_chunk(heading.group(2))
                    translated_blocks.append(heading.group(1) + translated_heading)
                    continue

                # Bullets/numbered recommendations are translated as complete
                # items rather than word-by-word fragments.
                item = _re.match(r"^(\s*(?:[-*]\s+|\d+[.)]\s+))(.+?)\s*$", raw)
                if item:
                    flush_paragraph()
                    translated_item = translate_chunk(item.group(2))
                    translated_blocks.append(item.group(1) + translated_item)
                    continue

                # Normal prose lines are joined into one paragraph before
                # translation, giving the provider enough context for smooth
                # grammar and terminology.
                current_paragraph.append(raw + " ")

            flush_paragraph()

        translated = "\n\n".join(
            part for part in translated_blocks if part is not None
        )

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
