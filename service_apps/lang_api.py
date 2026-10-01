from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.language_service import LanguageService, SupportedLanguage

app = FastAPI(title="Kisan Ki Awaz Language API", version="1.0.0")
language_service = LanguageService()

class LanguageSelectRequest(BaseModel):
    language: str

@app.get("/health")
def health():
    cfg = language_service.current_config
    return {
        "status": "healthy",
        "service": "language",
        "language": cfg.code,
        "name_en": cfg.name_en,
        "name_native": cfg.name_native,
    }

@app.get("/languages")
def languages():
    return {
        "languages": [
            {
                "code": lang.value,
                "name_en": cfg.name_en,
                "name_native": cfg.name_native,
                "rtl": cfg.rtl,
                "locale": cfg.locale,
                "tts_code": cfg.tts_code,
                "fallback_tts": cfg.fallback_tts,
                "font_family": cfg.font_family,
            }
            for lang, cfg in language_service.get_available_languages().items()
        ]
    }

@app.post("/select")
def select_language(req: LanguageSelectRequest):
    try:
        lang = SupportedLanguage(req.language)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {req.language}")
    language_service.set_language(lang)
    cfg = language_service.current_config
    return {
        "language": cfg.code,
        "name_en": cfg.name_en,
        "name_native": cfg.name_native,
        "rtl": cfg.rtl,
        "font_family": cfg.font_family,
        "locale": cfg.locale,
        "tts_code": cfg.tts_code,
    }

@app.get("/translations")
def translations(language: str = "en"):
    try:
        SupportedLanguage(language)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {language}")
    from services.language_service import UI_TRANSLATIONS
    return {
        "language": language,
        "translations": {
            key: values.get(language, values.get("en", key))
            for key, values in UI_TRANSLATIONS.items()
        },
    }
