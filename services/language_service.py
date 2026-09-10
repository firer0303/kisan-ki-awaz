"""
Kisan Ki Awaz - Language Management Service
=============================================
Manages multilingual support for 6 languages:
Urdu, Sindhi, Punjabi, Pashto, Balochi, and English.

Language selection happens BEFORE any voice/camera/image input
and persists throughout the entire session.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class SupportedLanguage(str, Enum):
    """Enumeration of all supported languages."""
    URDU = "ur"
    SINDHI = "sd"
    PUNJABI = "pa"
    PASHTO = "ps"
    BALOCHI = "bal"
    ENGLISH = "en"


@dataclass
class LanguageConfig:
    """Configuration for a single language."""
    code: str                    # ISO 639-1 or custom code
    name_en: str                 # English name
    name_native: str             # Native script name
    locale: str                  # BCP 47 locale tag
    tts_code: str                # TTS service language code
    stt_code: str                # STT service language code
    rtl: bool                    # Right-to-left script?
    font_family: str             # Preferred web font
    fallback_tts: Optional[str]  # Fallback TTS code if primary unsupported


# ──────────────────────────────────────────────────────────────
# Master language registry
# ──────────────────────────────────────────────────────────────
LANGUAGE_REGISTRY: Dict[SupportedLanguage, LanguageConfig] = {
    SupportedLanguage.URDU: LanguageConfig(
        code="ur",
        name_en="Urdu",
        name_native="اردو",
        locale="ur-PK",
        tts_code="ur-PK",
        stt_code="ur-PK",
        rtl=True,
        font_family="'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif",
        fallback_tts="hi-IN",  # Hindi as closest fallback
    ),
    SupportedLanguage.SINDHI: LanguageConfig(
        code="sd",
        name_en="Sindhi",
        name_native="سنڌي",
        locale="sd-PK",
        tts_code="sd-PK",
        stt_code="sd-PK",
        rtl=True,
        font_family="'Noto Nastaliq Urdu', 'Noto Sans Arabic', serif",
        fallback_tts="ur-PK",  # Urdu as fallback
    ),
    SupportedLanguage.PUNJABI: LanguageConfig(
        code="pa",
        name_en="Punjabi",
        name_native="پنجابی",
        locale="pa-PK",
        tts_code="pa-PK",
        stt_code="pa-PK",
        rtl=True,  # Shahmukhi script used in Pakistan
        font_family="'Noto Nastaliq Urdu', serif",
        fallback_tts="ur-PK",
    ),
    SupportedLanguage.PASHTO: LanguageConfig(
        code="ps",
        name_en="Pashto",
        name_native="پښتو",
        locale="ps-AF",
        tts_code="ps-AF",
        stt_code="ps-AF",
        rtl=True,
        font_family="'Noto Sans Arabic', 'Noto Nastaliq Urdu', serif",
        fallback_tts="ur-PK",
    ),
    SupportedLanguage.BALOCHI: LanguageConfig(
        code="bal",
        name_en="Balochi",
        name_native="بلوچی",
        locale="bal-PK",
        tts_code="ur-PK",  # Balochi not natively supported by most TTS
        stt_code="ur-PK",
        rtl=True,
        font_family="'Noto Nastaliq Urdu', serif",
        fallback_tts="ur-PK",
        # NOTE: Balochi TTS is not natively supported by major providers.
        # The system will use Urdu TTS with a disclaimer that Balochi
        # narration is approximated via Urdu pronunciation.
    ),
    SupportedLanguage.ENGLISH: LanguageConfig(
        code="en",
        name_en="English",
        name_native="English",
        locale="en-US",
        tts_code="en-US",
        stt_code="en-US",
        rtl=False,
        font_family="'Inter', 'Segoe UI', sans-serif",
        fallback_tts=None,
    ),
}


# ──────────────────────────────────────────────────────────────
# UI translation dictionary (static labels shown in the interface)
# ──────────────────────────────────────────────────────────────
UI_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "app_title": {
        "ur": "کسان کی آواز",
        "sd": "هاريءَ جو آواز",
        "pa": "کسان دی اواز",
        "ps": "د بزگر آواز",
        "bal": "کشانی آواز",
        "en": "Kisan Ki Awaz",
    },
    "app_subtitle": {
        "ur": "پاکستانی کسانوں کے لیے AI فارمنگ اسسٹنٹ",
        "sd": "پاڪستاني هارين لاءِ AI فارمنگ اسسٽنٽ",
        "pa": "پاکستانی کساناں لئی AI فارمنگ اسسٹنٹ",
        "ps": "د پاکستانی بزگرانو لپاره د AI کرهڼې مرستیال",
        "bal": "پاکستانی کشاناںءِ تئ AI دراجی معاون",
        "en": "AI Farming Assistant for Pakistani Farmers",
    },
    "select_language": {
        "ur": "اپنی زبان منتخب کریں",
        "sd": "پنھنجي ٻولي چونڊيو",
        "pa": "اپنی بولی چنو",
        "ps": "خپله ژبه وټاکئ",
        "bal": "وتی زبانءَ برچینت کنیت",
        "en": "Select Your Language",
    },
    "voice_input": {
        "ur": "آواز سے پوچھیں",
        "sd": "آواز سان پڇو",
        "pa": "آواز نال پچھو",
        "ps": "د غږ له لارې وپوښتئ",
        "bal": "آوازءَ گونءِ بپرسیت",
        "en": "Ask by Voice",
    },
    "camera_input": {
        "ur": "تصویر لیں",
        "sd": "تصوير وٺو",
        "pa": "تصویر لوو",
        "ps": "انځور واخلئ",
        "bal": "عکسءَ بگریت",
        "en": "Take Picture",
    },
    "image_upload": {
        "ur": "تصویر اپ لوڈ کریں",
        "sd": "تصوير اپ لوڊ ڪريو",
        "pa": "تصویر اپ لوڈ کرو",
        "ps": "انځور پورته کړئ",
        "bal": "عکسءَ آپلود کنیت",
        "en": "Upload Image",
    },
    "analyzing": {
        "ur": "تجزیہ ہو رہا ہے...",
        "sd": "تجزيو ٿي رهيو آهي...",
        "pa": "تجزیہ ہو رہیا اے...",
        "ps": "تحلیل روان دی...",
        "bal": "تحلیلءَ بوتگ ان...",
        "en": "Analyzing...",
    },
    "verified_sources": {
        "ur": "تصدیق شدہ ذرائع",
        "sd": "تصديق ٿيل ذريعا",
        "pa": "تصدیق شدہ ذرائع",
        "ps": "تایید شوي سرچینې",
        "bal": "تصدیقءَ بوتگین سراجاݔں",
        "en": "Verified Sources",
    },
    "confidence": {
        "ur": "اعتماد کی سطح",
        "sd": "اعتماد جي سطح",
        "pa": "اعتماد دی سطح",
        "ps": "د باور کچه",
        "bal": "اعتمادءِ سطح",
        "en": "Confidence Level",
    },
    "risk_level": {
        "ur": "خطرے کی سطح",
        "sd": "خطري جي سطح",
        "pa": "خطرے دی سطح",
        "ps": "د خطر کچه",
        "bal": "خطراءِ سطح",
        "en": "Risk Level",
    },
    "recommendations": {
        "ur": "سفارشات",
        "sd": "سفارشون",
        "pa": "سفارشاں",
        "ps": "سپارښتنې",
        "bal": "سفارشں",
        "en": "Recommendations",
    },
    "why_this": {
        "ur": "یہ سفارش کیوں؟",
        "sd": "هي سفارش ڇو؟",
        "pa": "ایہہ سفارش کیوں؟",
        "ps": "ولې دا سپارښتنه؟",
        "bal": "اے سفارشءِ چیرا؟",
        "en": "Why this recommendation?",
    },
    "no_source": {
        "ur": "کوئی تصدیق شدہ ذریعہ دستیاب نہیں",
        "sd": "ڪو به تصديق ٿيل ذريعو دستياب ناهي",
        "pa": "کوئی تصدیق شدہ ذریع دستیاب نہیں",
        "ps": "هیڅ تایید شوې سرچینه شتون نلري",
        "bal": "هیچ تصدیقءَ بوتگین سراجاءِ دست کپءَ نیست",
        "en": "No verified source available",
    },
    "low_confidence_warning": {
        "ur": "کم اعتماد - براہ کرم ماہر سے مشورہ کریں",
        "sd": "گھٽ اعتماد - مھرباني ڪري ماھر سان صلاح ڪريو",
        "pa": "گھٹ اعتماد - مہربانی کرکے ماہر نال صلاح کرو",
        "ps": "ټیټ باور - مهرباني وکړئ له پوه سره مشوره وکړئ",
        "bal": "کم اعتماد - مھربانیءَ گونءِ ماھرءَ مشورت کنیت",
        "en": "Low confidence - Please consult an expert",
    },
    "demo_mode": {
        "ur": "ڈیمو موڈ",
        "sd": "ڊيمو موڊ",
        "pa": "ڈیمو موڈ",
        "ps": "ډیمو حالت",
        "bal": "ڈیمو حالَت",
        "en": "Demo Mode",
    },
    "pause": {
        "ur": "روکیں",
        "sd": "روڪيو",
        "pa": "روکو",
        "ps": "ودرول",
        "bal": "داریت",
        "en": "Pause",
    },
    "resume": {
        "ur": "جاری رکھیں",
        "sd": "جاري رکو",
        "pa": "جاری رکھو",
        "ps": "بیا پیل",
        "bal": "جاری داریت",
        "en": "Resume",
    },
    "replay": {
        "ur": "دوبارہ سنیں",
        "sd": "ٻيهر ٻڌو",
        "pa": "دوبارہ سنو",
        "ps": "بیا واورئ",
        "bal": "دوبارہءِ بوشیت",
        "en": "Replay",
    },
    "stop": {
        "ur": "بند کریں",
        "sd": "بند ڪريو",
        "pa": "بند کرو",
        "ps": "ودرول",
        "bal": "بند کنیت",
        "en": "Stop",
    },
    "narration_fallback_notice": {
        "ur": "آواز اردو میں پڑھی جا رہی ہے (آپ کی زبان میں دستیاب نہیں)",
        "sd": "آواز اردو ۾ پڙهي پئي وڃي (توھان جي ٻوليءَ ۾ دستياب ناھي)",
        "pa": "آواز اردو وچ پڑھی جا رہی اے (تہاڈی بولی وچ دستیاب نہیں)",
        "ps": "غږ په اردو لوستل کیږي (ستاسو په ژبه کې شتون نلري)",
        "bal": "آواز اردوئیءَ وارت بوتگ اں (شمی زبانءَ دست کپءَ نیست)",
        "en": "Voice narration is in Urdu (not available in your language)",
    },
    "crop_analysis": {
        "ur": "فصل کا تجزیہ",
        "sd": "فصل جو تجزيو",
        "pa": "فصل دا تجزیہ",
        "ps": "د کرهڼې تحلیل",
        "bal": "ہربوگی تحلیل",
        "en": "Crop Analysis",
    },
    "disease_pest": {
        "ur": "بیماری / کیڑے",
        "sd": "بيماري / جيت",
        "pa": "بیماری / کیڑے",
        "ps": "ناروغي / آفتونه",
        "bal": "بیماریءِ / ہپَتءِ",
        "en": "Disease / Pest",
    },
    "market_info": {
        "ur": "منڈی کی معلومات",
        "sd": "منڊيءَ جي معلومات",
        "pa": "منڈی دی معلومات",
        "ps": "د بازار معلومات",
        "bal": "بازارءِ مالومات",
        "en": "Market Information",
    },
    "weather_alerts": {
        "ur": "موسم کی معلومات",
        "sd": "موسم جي ڄاڻ",
        "pa": "موسم دی جانکاری",
        "ps": "د هوا معلومات",
        "bal": "ھوَاءِ مالومات",
        "en": "Weather Alerts",
    },
    "history": {
        "ur": "تاریخ",
        "sd": "تاريخ",
        "pa": "تریخ",
        "ps": "تاریخ",
        "bal": "تاریخ",
        "en": "History",
    },
}


class LanguageService:
    """
    Manages language selection and session persistence.

    Usage:
        lang_service = LanguageService()
        lang_service.set_language(SupportedLanguage.URDU)
        label = lang_service.translate("app_title")
    """

    def __init__(self):
        self._current_language: SupportedLanguage = SupportedLanguage.ENGLISH
        self._translations = UI_TRANSLATIONS
        self._registry = LANGUAGE_REGISTRY

    @property
    def current_language(self) -> SupportedLanguage:
        return self._current_language

    @property
    def current_config(self) -> LanguageConfig:
        return self._registry[self._current_language]

    def set_language(self, language: SupportedLanguage) -> None:
        """Set the active session language."""
        if language not in self._registry:
            raise ValueError(f"Unsupported language: {language}")
        self._current_language = language

    def get_available_languages(self) -> Dict[SupportedLanguage, LanguageConfig]:
        """Return all available languages with their configurations."""
        return dict(self._registry)

    def translate(self, key: str) -> str:
        """
        Translate a UI label key into the current language.
        Falls back to English if the key is missing for the current language.
        """
        lang_code = self._current_language.value
        if key in self._translations:
            return self._translations[key].get(
                lang_code,
                self._translations[key].get("en", key)
            )
        return key

    def get_tts_config(self) -> Dict:
        """
        Return TTS configuration for the current language.
        Includes fallback strategy if primary TTS is not supported.
        """
        config = self.current_config
        return {
            "primary_code": config.tts_code,
            "fallback_code": config.fallback_tts,
            "using_fallback": False,  # Will be set by TTS service if primary fails
            "language_name": config.name_en,
            "rtl": config.rtl,
        }

    def get_stt_config(self) -> Dict:
        """Return speech-to-text configuration for the current language."""
        config = self.current_config
        return {
            "language_code": config.stt_code,
            "language_name": config.name_en,
        }

    def is_rtl(self) -> bool:
        """Check if the current language uses right-to-left script."""
        return self.current_config.rtl

    def get_font_family(self) -> str:
        """Return the preferred font family for the current language."""
        return self.current_config.font_family

    def get_text_direction_css(self) -> str:
        """Return CSS text-direction properties for the current language."""
        if self.is_rtl():
            return "direction: rtl; text-align: right;"
        return "direction: ltr; text-align: left;"
