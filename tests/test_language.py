"""
Tests for the Language Service.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.language_service import LanguageService, SupportedLanguage


class TestLanguageService:
    """Test suite for the language management service."""

    def setup_method(self):
        self.ls = LanguageService()

    def test_default_language_is_english(self):
        assert self.ls.current_language == SupportedLanguage.ENGLISH

    def test_set_language_urdu(self):
        self.ls.set_language(SupportedLanguage.URDU)
        assert self.ls.current_language == SupportedLanguage.URDU
        assert self.ls.current_config.name_en == "Urdu"
        assert self.ls.is_rtl() is True

    def test_set_language_sindhi(self):
        self.ls.set_language(SupportedLanguage.SINDHI)
        assert self.ls.current_language == SupportedLanguage.SINDHI
        assert self.ls.current_config.name_native == "سنڌي"

    def test_set_language_punjabi(self):
        self.ls.set_language(SupportedLanguage.PUNJABI)
        assert self.ls.current_language == SupportedLanguage.PUNJABI
        assert self.ls.is_rtl() is True

    def test_set_language_pashto(self):
        self.ls.set_language(SupportedLanguage.PASHTO)
        assert self.ls.current_language == SupportedLanguage.PASHTO

    def test_set_language_balochi(self):
        self.ls.set_language(SupportedLanguage.BALOCHI)
        assert self.ls.current_language == SupportedLanguage.BALOCHI

    def test_set_language_english(self):
        self.ls.set_language(SupportedLanguage.ENGLISH)
        assert self.ls.current_language == SupportedLanguage.ENGLISH
        assert self.ls.is_rtl() is False

    def test_translate_returns_current_language(self):
        self.ls.set_language(SupportedLanguage.ENGLISH)
        result = self.ls.translate("app_title")
        assert result == "Kisan Ki Awaz"

    def test_translate_urdu(self):
        self.ls.set_language(SupportedLanguage.URDU)
        result = self.ls.translate("app_title")
        assert result == "کسان کی آواز"

    def test_translate_fallback_to_english(self):
        self.ls.set_language(SupportedLanguage.BALOCHI)
        # If key exists but Balochi translation is missing, falls back to English
        result = self.ls.translate("nonexistent_key")
        assert result == "nonexistent_key"

    def test_get_available_languages_returns_six(self):
        languages = self.ls.get_available_languages()
        assert len(languages) == 6

    def test_tts_config_urdu(self):
        self.ls.set_language(SupportedLanguage.URDU)
        tts = self.ls.get_tts_config()
        assert tts["primary_code"] == "ur-PK"
        assert tts["using_fallback"] is False

    def test_tts_config_balochi_uses_fallback(self):
        self.ls.set_language(SupportedLanguage.BALOCHI)
        tts = self.ls.get_tts_config()
        assert tts["primary_code"] == "ur-PK"  # Balochi uses Urdu TTS
        assert tts["fallback_code"] == "ur-PK"

    def test_text_direction_css_rtl(self):
        self.ls.set_language(SupportedLanguage.URDU)
        css = self.ls.get_text_direction_css()
        assert "rtl" in css

    def test_text_direction_css_ltr(self):
        self.ls.set_language(SupportedLanguage.ENGLISH)
        css = self.ls.get_text_direction_css()
        assert "ltr" in css

    def test_font_family_english(self):
        self.ls.set_language(SupportedLanguage.ENGLISH)
        font = self.ls.get_font_family()
        assert "Inter" in font or "Segoe" in font

    def test_font_family_urdu(self):
        self.ls.set_language(SupportedLanguage.URDU)
        font = self.ls.get_font_family()
        assert "Nastaliq" in font or "Noto" in font

    def test_invalid_language_raises_error(self):
        import pytest
        with pytest.raises(ValueError):
            self.ls.set_language("invalid_language")
