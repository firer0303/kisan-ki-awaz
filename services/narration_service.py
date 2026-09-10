"""
Kisan Ki Awaz - Narration Service
====================================
Manages automatic voice narration after AI results are generated.
Handles auto-play, pause, resume, replay, and stop controls.
"""
import base64
from typing import Dict, Optional

from loguru import logger

from services.language_service import LanguageService
from services.speech_service import SpeechService


class NarrationService:
    """
    Manages the voice narration lifecycle.
    After the AI generates a result, this service:
    1. Converts the result text to speech
    2. Prepares the audio for auto-playback in the Streamlit UI
    3. Provides controls for pause/resume/replay/stop
    """

    def __init__(
        self,
        speech_service: SpeechService,
        language_service: LanguageService,
    ):
        self.speech = speech_service
        self.lang = language_service
        self._current_audio: Optional[bytes] = None
        self._current_format: str = "mp3"
        self._is_playing: bool = False

    def prepare_narration(self, text: str) -> Dict:
        """
        Convert result text to audio for auto-narration.

        Returns:
            Dict with keys:
            - audio_b64: str (base64-encoded audio for HTML5 audio)
            - format: str ("mp3" or "wav")
            - mime_type: str
            - language: str
            - is_fallback: bool
            - fallback_notice: str or None
            - error: str or None
        """
        # Truncate very long text for TTS (keep first ~2000 chars)
        tts_text = text[:2000] if len(text) > 2000 else text

        # Remove markdown formatting for cleaner speech
        tts_text = self._clean_for_speech(tts_text)

        # Generate speech
        result = self.speech.text_to_speech(tts_text)

        if result.get("audio_bytes") is None:
            return {
                "audio_b64": None,
                "format": "mp3",
                "mime_type": "audio/mpeg",
                "language": self.lang.current_language.value,
                "is_fallback": True,
                "fallback_notice": result.get("fallback_notice", "TTS not available."),
                "error": "No audio generated",
            }

        audio_bytes = result["audio_bytes"]
        audio_format = result.get("format", "mp3")
        mime = "audio/mpeg" if audio_format == "mp3" else "audio/wav"

        # Encode for HTML5 audio embedding
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        self._current_audio = audio_bytes
        self._current_format = audio_format

        return {
            "audio_b64": audio_b64,
            "format": audio_format,
            "mime_type": mime,
            "language": self.lang.current_language.value,
            "is_fallback": result.get("is_fallback", False),
            "fallback_notice": result.get("fallback_notice"),
            "error": None,
        }

    def get_autoplay_html(self, narration: Dict) -> str:
        """
        Generate HTML for auto-playing the narration.
        This creates an HTML5 audio element that auto-plays
        and includes pause/resume/replay/stop controls.
        """
        if not narration.get("audio_b64"):
            return ""

        mime = narration["mime_type"]
        b64 = narration["audio_b64"]
        lang_name = self.lang.current_config.name_en

        fallback_notice_html = ""
        if narration.get("is_fallback") and narration.get("fallback_notice"):
            fallback_notice_html = f"""
            <div style="background:#fff3cd; padding:8px 12px; border-radius:6px;
                        margin-bottom:10px; font-size:13px; color:#856404;">
                ⚠️ {narration['fallback_notice']}
            </div>
            """

        html = f"""
        {fallback_notice_html}
        <div style="background:#f0f8f0; padding:15px; border-radius:10px;
                    border:1px solid #28a745; margin:10px 0;">
            <div style="font-size:14px; color:#28a745; margin-bottom:8px;
                        font-weight:600;">
                🔊 Auto-Narration ({lang_name})
            </div>
            <audio id="narration-audio" controls autoplay
                   style="width:100%; height:40px;">
                <source src="data:{mime};base64,{b64}" type="{mime}">
                Your browser does not support audio playback.
            </audio>
            <div style="display:flex; gap:8px; margin-top:8px; flex-wrap:wrap;">
                <button onclick="document.getElementById('narration-audio').pause()"
                        style="padding:6px 14px; border-radius:6px; border:1px solid #ccc;
                               background:#f8f9fa; cursor:pointer; font-size:13px;">
                    ⏸ Pause
                </button>
                <button onclick="document.getElementById('narration-audio').play()"
                        style="padding:6px 14px; border-radius:6px; border:1px solid #ccc;
                               background:#f8f9fa; cursor:pointer; font-size:13px;">
                    ▶ Resume
                </button>
                <button onclick="var a=document.getElementById('narration-audio');
                                 a.currentTime=0; a.play()"
                        style="padding:6px 14px; border-radius:6px; border:1px solid #ccc;
                               background:#f8f9fa; cursor:pointer; font-size:13px;">
                    🔄 Replay
                </button>
                <button onclick="var a=document.getElementById('narration-audio');
                                 a.pause(); a.currentTime=0"
                        style="padding:6px 14px; border-radius:6px; border:1px solid #ccc;
                               background:#f8f9fa; cursor:pointer; font-size:13px;">
                    ⏹ Stop
                </button>
            </div>
        </div>
        """
        return html

    def _clean_for_speech(self, text: str) -> str:
        """Clean markdown formatting for better TTS output."""
        import re
        # Remove markdown headers
        text = re.sub(r'#{1,6}\s+', '', text)
        # Remove bold/italic markers
        text = re.sub(r'\*{1,2}', '', text)
        text = re.sub(r'_{1,2}', '', text)
        # Remove bullet markers
        text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
        # Remove source reference brackets
        text = re.sub(r'\[Source:[^\]]*\]', '', text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Collapse whitespace
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
