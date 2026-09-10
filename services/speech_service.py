"""
Kisan Ki Awaz - Speech Service
================================
Handles speech-to-text (STT) and text-to-speech (TTS) with
multilingual support and automatic narration.

FALLBACK STRATEGY:
- If Google Cloud STT/TTS is configured, use it for best quality.
- Otherwise, use gTTS (Google Translate TTS) which supports Urdu and English.
- For languages not natively supported by gTTS (Sindhi, Punjabi, Pashto,
  Balochi), fall back to Urdu TTS with a visible disclaimer.
- pyttsx3 is used as an offline fallback for English only.
"""
import io
import os
import tempfile
from typing import Dict, Optional

from loguru import logger

from services.language_service import LanguageService, SupportedLanguage


# ──────────────────────────────────────────────────────────────
# TTS Language Support Matrix
# ──────────────────────────────────────────────────────────────
# gTTS native support
GTTS_NATIVE = {"en": "en", "ur": "ur"}

# gTTS fallback mapping (language -> closest supported gTTS code)
GTTS_FALLBACK = {
    "sd": "ur",   # Sindhi -> Urdu
    "pa": "ur",   # Punjabi -> Urdu (same script family)
    "ps": "ur",   # Pashto -> Urdu
    "bal": "ur",  # Balochi -> Urdu
}


class SpeechService:
    """Manages speech-to-text and text-to-speech operations."""

    def __init__(self, language_service: LanguageService):
        self.lang_service = language_service
        self._tts_engine = None  # Lazy-init pyttsx3

    # ──────────────────────────────────────────────────────────
    # Text-to-Speech
    # ──────────────────────────────────────────────────────────

    def text_to_speech(self, text: str) -> Dict:
        """
        Convert text to speech audio bytes.

        Returns:
            Dict with keys:
            - audio_bytes: bytes (MP3 or WAV)
            - format: str ("mp3" or "wav")
            - language_used: str (actual language code used)
            - is_fallback: bool
            - fallback_notice: str or None
        """
        lang_code = self.lang_service.current_language.value
        tts_config = self.lang_service.get_tts_config()

        # Determine the actual TTS language to use
        actual_lang, is_fallback = self._resolve_tts_language(lang_code)

        # Try gTTS first (best for Urdu/English)
        try:
            audio_bytes = self._gtts_generate(text, actual_lang)
            return {
                "audio_bytes": audio_bytes,
                "format": "mp3",
                "language_used": actual_lang,
                "is_fallback": is_fallback,
                "fallback_notice": (
                    self.lang_service.translate("narration_fallback_notice")
                    if is_fallback
                    else None
                ),
            }
        except Exception as e:
            logger.warning(f"gTTS failed for '{actual_lang}': {e}")

        # Fallback: pyttsx3 (offline, English only)
        if lang_code == "en":
            try:
                audio_bytes = self._pyttsx3_generate(text)
                return {
                    "audio_bytes": audio_bytes,
                    "format": "wav",
                    "language_used": "en",
                    "is_fallback": True,
                    "fallback_notice": "Using offline TTS engine.",
                }
            except Exception as e:
                logger.error(f"pyttsx3 fallback also failed: {e}")

        # Final fallback: return no audio
        return {
            "audio_bytes": None,
            "format": "mp3",
            "language_used": lang_code,
            "is_fallback": True,
            "fallback_notice": (
                f"Text-to-speech not available for "
                f"{self.lang_service.current_config.name_en}. "
                "Please read the written response."
            ),
        }

    def _resolve_tts_language(self, lang_code: str) -> tuple:
        """
        Determine the actual TTS language code to use.
        Returns (actual_code, is_fallback).
        """
        if lang_code in GTTS_NATIVE:
            return lang_code, False

        fallback_code = GTTS_FALLBACK.get(lang_code, "ur")
        return fallback_code, True

    def _gtts_generate(self, text: str, lang: str) -> bytes:
        """Generate speech using gTTS."""
        from gtts import gTTS

        tts = gTTS(text=text, lang=lang, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.read()

    def _pyttsx3_generate(self, text: str) -> bytes:
        """Generate speech using pyttsx3 (offline, English only)."""
        import pyttsx3

        if self._tts_engine is None:
            self._tts_engine = pyttsx3.init()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        self._tts_engine.save_to_file(text, tmp_path)
        self._tts_engine.runAndWait()

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        os.unlink(tmp_path)
        return audio_bytes

    # ──────────────────────────────────────────────────────────
    # Speech-to-Text
    # ──────────────────────────────────────────────────────────

    def speech_to_text(self, audio_data: bytes, sample_rate: int = 16000) -> Dict:
        """
        Convert speech audio to text.

        In production, this should use Google Cloud Speech API
        or Azure Speech Services for the selected language.

        Currently returns a demo message since real STT requires
        API keys and microphone access.

        Returns:
            Dict with keys:
            - text: str
            - language: str
            - confidence: float
            - is_demo: bool
            - error: str or None
        """
        stt_config = self.lang_service.get_stt_config()
        lang_code = stt_config["language_code"]

        # Try SpeechRecognition with Google Web API
        try:
            import speech_recognition as sr

            recognizer = sr.Recognizer()

            # Write audio data to a WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp_path = tmp.name

            with sr.AudioFile(tmp_path) as source:
                audio = recognizer.record(source)

            os.unlink(tmp_path)

            # Try Google Web Speech API (free tier)
            text = recognizer.recognize_google(
                audio, language=lang_code.replace("-PK", "-IN").replace("-AF", "-AF")
            )

            return {
                "text": text,
                "language": lang_code,
                "confidence": 0.85,
                "is_demo": False,
                "error": None,
            }
        except Exception as e:
            logger.warning(f"Speech recognition failed: {e}")
            return {
                "text": "",
                "language": lang_code,
                "confidence": 0,
                "is_demo": True,
                "error": (
                    f"Speech recognition not available. "
                    f"Configure Google Speech API key in .env for "
                    f"{stt_config['language_name']} speech input. "
                    f"You can type your question instead."
                ),
            }

    def get_supported_stt_languages(self) -> list:
        """Return languages supported for speech-to-text."""
        return [
            {"code": "en-US", "name": "English", "native_support": True},
            {"code": "ur-PK", "name": "Urdu", "native_support": True},
            {"code": "hi-IN", "name": "Hindi (fallback for similar languages)", "native_support": True},
        ]
