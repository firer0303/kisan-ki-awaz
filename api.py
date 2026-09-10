"""
Kisan Ki Awaz - FastAPI Backend
=================================
REST API server exposing all AI services for the Android app.
Run with:  python api.py
Or:        uvicorn api:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root on path
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from PIL import Image
from pydantic import BaseModel

from services.language_service import LanguageService, SupportedLanguage
from services.recommendation_engine import RecommendationEngine

# ──────────────────────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Kisan Ki Awaz API",
    description="AI Farming Assistant API for Pakistani Farmers",
    version="1.0.0",
)

# CORS - allow Android app and web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ──────────────────────────────────────────────────────────────
# Service Singletons
# ──────────────────────────────────────────────────────────────
language_service = LanguageService()
engine: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    global engine
    if engine is None:
        engine = RecommendationEngine()
    return engine


# ──────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────
class LanguageSelectRequest(BaseModel):
    language: str  # "ur", "sd", "pa", "ps", "bal", "en"


class VoiceQueryRequest(BaseModel):
    text: str
    language: str = "en"


class ImageAnalysisRequest(BaseModel):
    image_base64: str  # base64-encoded image
    question: str = ""
    language: str = "en"
    input_type: str = "upload"


class WeatherRequest(BaseModel):
    region: str = "punjab_central"


class MarketRequest(BaseModel):
    crop: str = "wheat"


class ExplainRequest(BaseModel):
    result_id: str = ""


# ──────────────────────────────────────────────────────────────
# Health & Info Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Serve the mobile web frontend."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Kisan Ki Awaz API is running. Visit /docs for API documentation."}


@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "Kisan Ki Awaz", "version": "1.0.0"}


@app.get("/api/languages")
async def get_languages():
    """Return all supported languages."""
    ls = LanguageService()
    languages = ls.get_available_languages()
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
            }
            for lang, cfg in languages.items()
        ]
    }


# ──────────────────────────────────────────────────────────────
# Language Selection
# ──────────────────────────────────────────────────────────────
@app.post("/api/language/select")
async def select_language(req: LanguageSelectRequest):
    """Set the active session language."""
    try:
        lang_enum = SupportedLanguage(req.language)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {req.language}")

    language_service.set_language(lang_enum)
    eng = get_engine()
    eng.language_service.set_language(lang_enum)

    cfg = language_service.current_config
    return {
        "language": cfg.code,
        "name_en": cfg.name_en,
        "name_native": cfg.name_native,
        "rtl": cfg.rtl,
        "font_family": cfg.font_family,
    }


@app.get("/api/translations")
async def get_translations(language: str = "en"):
    """Return all UI translations for a language."""
    from services.language_service import UI_TRANSLATIONS

    result = {}
    for key, translations in UI_TRANSLATIONS.items():
        result[key] = translations.get(language, translations.get("en", key))
    return {"language": language, "translations": result}


# ──────────────────────────────────────────────────────────────
# Voice / Text Query
# ──────────────────────────────────────────────────────────────
@app.post("/api/query/voice")
async def process_voice_query(req: VoiceQueryRequest):
    """Process a text/voice query and return AI recommendation."""
    if not req.text.strip():
        raise HTTPException(400, "Question text cannot be empty.")

    # Set language
    try:
        lang_enum = SupportedLanguage(req.language)
        language_service.set_language(lang_enum)
        eng = get_engine()
        eng.language_service.set_language(lang_enum)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {req.language}")

    eng = get_engine()
    result = eng.process_voice_query(req.text.strip())

    return _format_result(result)


# ──────────────────────────────────────────────────────────────
# Image Analysis
# ──────────────────────────────────────────────────────────────
@app.post("/api/query/image")
async def process_image_query(req: ImageAnalysisRequest):
    """Analyze an uploaded/captured image for crop disease."""
    # Set language
    try:
        lang_enum = SupportedLanguage(req.language)
        language_service.set_language(lang_enum)
        eng = get_engine()
        eng.language_service.set_language(lang_enum)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {req.language}")

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(req.image_base64)
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(400, f"Invalid image data: {str(e)}")

    eng = get_engine()
    result = eng.process_image(img, req.question, req.input_type)

    return _format_result(result)


@app.post("/api/query/image-upload")
async def process_image_upload(
    file: UploadFile = File(...),
    question: str = Form(""),
    language: str = Form("en"),
    input_type: str = Form("upload"),
):
    """Analyze an uploaded image file (multipart form)."""
    # Set language
    try:
        lang_enum = SupportedLanguage(language)
        language_service.set_language(lang_enum)
        eng = get_engine()
        eng.language_service.set_language(lang_enum)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {language}")

    # Read image
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {str(e)}")

    eng = get_engine()
    result = eng.process_image(img, question, input_type)

    return _format_result(result)


# ──────────────────────────────────────────────────────────────
# Weather
# ──────────────────────────────────────────────────────────────
@app.post("/api/weather")
async def get_weather(req: WeatherRequest):
    """Get weather and agricultural risk alerts."""
    eng = get_engine()
    weather = eng.weather_service.get_weather(req.region)
    return weather


@app.get("/api/weather/regions")
async def get_weather_regions():
    """List available weather regions."""
    eng = get_engine()
    return eng.weather_service.get_available_regions()


# ──────────────────────────────────────────────────────────────
# Market
# ──────────────────────────────────────────────────────────────
@app.post("/api/market")
async def get_market(req: MarketRequest):
    """Get crop market prices."""
    eng = get_engine()
    return eng.market_service.get_crop_prices(req.crop)


# ──────────────────────────────────────────────────────────────
# History
# ──────────────────────────────────────────────────────────────
@app.get("/api/history")
async def get_history(limit: int = 20):
    """Get analysis history for the current session."""
    eng = get_engine()
    records = eng.database_service.get_session_history(eng.session_id, limit)
    return {"session_id": eng.session_id, "records": records}


# ──────────────────────────────────────────────────────────────
# Knowledge Base Stats
# ──────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    eng = get_engine()
    rag_stats = eng.rag_service.get_statistics()
    db_stats = eng.database_service.get_stats()

    from config import settings

    return {
        "rag": rag_stats,
        "database": db_stats,
        "services": {
            "llm": settings.llm.active_provider,
            "stt": settings.speech.active_stt_provider,
            "tts": settings.speech.active_tts_provider,
            "vision": settings.vision.active_provider,
            "weather": settings.weather.active_provider,
            "market": settings.market.active_provider,
            "translation": settings.translation.active_provider,
        },
    }


# ──────────────────────────────────────────────────────────────
# Narration (TTS)
# ──────────────────────────────────────────────────────────────
@app.post("/api/narration")
async def get_narration(text: str = "", language: str = "en"):
    """Generate TTS audio for the given text."""
    try:
        lang_enum = SupportedLanguage(language)
        language_service.set_language(lang_enum)
        eng = get_engine()
        eng.language_service.set_language(lang_enum)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {language}")

    eng = get_engine()
    narration = eng.narration_service.prepare_narration(text)

    return {
        "audio_base64": narration.get("audio_b64"),
        "format": narration.get("format", "mp3"),
        "mime_type": narration.get("mime_type", "audio/mpeg"),
        "language": narration.get("language"),
        "is_fallback": narration.get("is_fallback", False),
        "fallback_notice": narration.get("fallback_notice"),
    }


# ──────────────────────────────────────────────────────────────
# Helper: Format analysis result for API response
# ──────────────────────────────────────────────────────────────
def _format_result(result: Dict) -> Dict:
    """Format an engine result into a clean API response."""
    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Analysis failed"),
            "warnings": result.get("warnings", []),
        }

    vision = result.get("vision_result", {})
    narration = result.get("narration", {})

    response = {
        "success": True,
        "response_text": result.get("response_text", ""),
        "sources": result.get("sources", []),
        "evidence_count": result.get("evidence_count", 0),
        "is_demo": result.get("is_demo", False),
        "input_type": result.get("input_type", "unknown"),
        "llm_provider": result.get("llm_provider", "unknown"),
        "translation_info": result.get("translation_info", {}),
    }

    # Add vision details if present
    if vision:
        response["vision"] = {
            "prediction": vision.get("prediction"),
            "confidence": vision.get("confidence", 0),
            "risk_level": vision.get("risk_level", "unknown"),
            "crop_detected": vision.get("crop_detected"),
            "disease_detected": vision.get("disease_detected"),
            "all_predictions": vision.get("all_predictions", []),
            "warnings": vision.get("warnings", []),
            "image_quality": vision.get("image_quality", {}),
        }

    # Add narration audio as base64
    if narration and narration.get("audio_b64"):
        response["narration"] = {
            "audio_base64": narration["audio_b64"],
            "format": narration.get("format", "mp3"),
            "mime_type": narration.get("mime_type", "audio/mpeg"),
            "is_fallback": narration.get("is_fallback", False),
            "fallback_notice": narration.get("fallback_notice"),
        }

    return response


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Kisan Ki Awaz API server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
