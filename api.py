"""
Kisan Ki Awaz - FastAPI Backend
=================================
REST API server exposing all AI services for the Android/web app.
"""
import base64
import io
import sys
import os
import httpx
from pathlib import Path
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from loguru import logger

from services.language_service import LanguageService, SupportedLanguage
from services.recommendation_engine import RecommendationEngine

app = FastAPI(title="Kisan Ki Awaz API", description="AI Farming Assistant API for Pakistani Farmers", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

language_service = LanguageService()
engine: Optional[RecommendationEngine] = None
ANSWER_SERVICE_URL = os.getenv('ANSWER_SERVICE_URL', '').rstrip('/')
VISION_SERVICE_URL = os.getenv('VISION_SERVICE_URL', '').rstrip('/')

def get_engine() -> RecommendationEngine:
    global engine
    if engine is None:
        engine = RecommendationEngine()
    return engine

class LanguageSelectRequest(BaseModel):
    language: str
class VoiceQueryRequest(BaseModel):
    text: str
    language: str = "en"
class ImageAnalysisRequest(BaseModel):
    image_base64: str
    question: str = ""
    language: str = "en"
    input_type: str = "upload"
class WeatherRequest(BaseModel):
    region: str = "punjab_central"
class MarketRequest(BaseModel):
    crop: str = "wheat"

@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Kisan Ki Awaz API is running. Visit /docs for API documentation."}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "Kisan Ki Awaz", "version": "1.0.0"}

@app.get("/api/languages")
async def get_languages():
    ls = LanguageService()
    return {"languages": [{"code": lang.value, "name_en": cfg.name_en, "name_native": cfg.name_native, "rtl": cfg.rtl, "locale": cfg.locale, "tts_code": cfg.tts_code, "fallback_tts": cfg.fallback_tts} for lang, cfg in ls.get_available_languages().items()]}

@app.post("/api/language/select")
async def select_language(req: LanguageSelectRequest):
    try:
        lang_enum = SupportedLanguage(req.language)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {req.language}")
    language_service.set_language(lang_enum)
    eng = get_engine()
    eng.language_service.set_language(lang_enum)
    cfg = language_service.current_config
    return {"language": cfg.code, "name_en": cfg.name_en, "name_native": cfg.name_native, "rtl": cfg.rtl, "font_family": cfg.font_family}

@app.get("/api/translations")
async def get_translations(language: str = "en"):
    from services.language_service import UI_TRANSLATIONS
    return {"language": language, "translations": {k: v.get(language, v.get("en", k)) for k, v in UI_TRANSLATIONS.items()}}

@app.post("/api/query/voice")
async def process_voice_query(req: VoiceQueryRequest):
    if not req.text.strip():
        raise HTTPException(400, "Question text cannot be empty.")
    try:
        lang_enum = SupportedLanguage(req.language)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {req.language}")
    language_service.set_language(lang_enum)
    eng = get_engine()
    eng.language_service.set_language(lang_enum)
    try:
        if ANSWER_SERVICE_URL:
            async with httpx.AsyncClient(timeout=90) as client:
                r = await client.post(f"{ANSWER_SERVICE_URL}/voice", json={"text": req.text.strip(), "language": req.language})
                r.raise_for_status()
                return _format_result(r.json())
        return _format_result(eng.process_voice_query(req.text.strip()))
    except Exception as exc:
        logger.exception("Voice query failed")
        raise HTTPException(500, "AI response could not be generated. Please try again.") from exc

async def _run_image(engine_obj, image: Image.Image, question: str, input_type: str):
    try:
        return _format_result(engine_obj.process_image(image, question, input_type))
    except Exception as exc:
        logger.exception("Image analysis request failed")
        raise HTTPException(500, "Image analysis failed. Please try a clear JPG or PNG image and try again.") from exc

@app.post("/api/query/image")
async def process_image_query(req: ImageAnalysisRequest):
    try:
        lang_enum = SupportedLanguage(req.language)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {req.language}")
    try:
        raw = req.image_base64.split(",", 1)[-1]
        image_bytes = base64.b64decode(raw, validate=True)
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as exc:
        raise HTTPException(400, "Invalid image data. Please upload a valid JPG or PNG image.") from exc
    language_service.set_language(lang_enum)
    eng = get_engine()
    eng.language_service.set_language(lang_enum)
    if VISION_SERVICE_URL and ANSWER_SERVICE_URL:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                vr = await client.post(f"{VISION_SERVICE_URL}/analyze", json={"image_base64": encoded})
                vr.raise_for_status()
                vision = vr.json()
                ar = await client.post(f"{ANSWER_SERVICE_URL}/image", json={"question": req.question, "language": req.language, "input_type": req.input_type, "vision": vision})
                ar.raise_for_status()
                return _format_result(ar.json())
        except Exception as exc:
            logger.exception("AI service routing failed")
            raise HTTPException(503, "AI image services are temporarily unavailable. Please try again.") from exc
    return await _run_image(eng, img, req.question, req.input_type)

@app.post("/api/query/image-upload")
async def process_image_upload(file: UploadFile = File(...), question: str = Form(""), language: str = Form("en"), input_type: str = Form("upload")):
    try:
        lang_enum = SupportedLanguage(language)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {language}")
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("empty")
        img = Image.open(io.BytesIO(contents))
        img.load()
    except Exception as exc:
        raise HTTPException(400, "Invalid image file. Please upload a valid JPG or PNG image.") from exc
    language_service.set_language(lang_enum)
    eng = get_engine()
    eng.language_service.set_language(lang_enum)
    if VISION_SERVICE_URL and ANSWER_SERVICE_URL:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                vr = await client.post(f"{VISION_SERVICE_URL}/analyze", json={"image_base64": encoded})
                vr.raise_for_status()
                vision = vr.json()
                ar = await client.post(f"{ANSWER_SERVICE_URL}/image", json={"question": question, "language": language, "input_type": input_type, "vision": vision})
                ar.raise_for_status()
                return _format_result(ar.json())
        except Exception as exc:
            logger.exception("AI service routing failed")
            raise HTTPException(503, "AI image services are temporarily unavailable. Please try again.") from exc
    return await _run_image(eng, img, question, input_type)

@app.post("/api/weather")
async def get_weather(req: WeatherRequest):
    return get_engine().weather_service.get_weather(req.region)

@app.get("/api/weather/regions")
async def get_weather_regions():
    return get_engine().weather_service.get_available_regions()

@app.post("/api/market")
async def get_market(req: MarketRequest):
    return get_engine().market_service.get_crop_prices(req.crop)

@app.get("/api/history")
async def get_history(limit: int = 20):
    eng = get_engine()
    return {"session_id": eng.session_id, "records": eng.database_service.get_session_history(eng.session_id, limit)}

@app.get("/api/stats")
async def get_stats():
    from config import settings
    eng = get_engine()
    return {"rag": eng.rag_service.get_statistics(), "database": eng.database_service.get_stats(), "services": {"llm": settings.llm.active_provider, "stt": settings.speech.active_stt_provider, "tts": settings.speech.active_tts_provider, "vision": settings.vision.active_provider, "weather": settings.weather.active_provider, "market": settings.market.active_provider, "translation": settings.translation.active_provider}}

@app.post("/api/narration")
async def get_narration(text: str = "", language: str = "en"):
    try:
        lang_enum = SupportedLanguage(language)
    except ValueError:
        raise HTTPException(400, f"Unsupported language: {language}")
    language_service.set_language(lang_enum)
    eng = get_engine()
    eng.language_service.set_language(lang_enum)
    narration = eng.narration_service.prepare_narration(text)
    return {"audio_base64": narration.get("audio_b64"), "format": narration.get("format", "mp3"), "mime_type": narration.get("mime_type", "audio/mpeg"), "language": narration.get("language"), "is_fallback": narration.get("is_fallback", False), "fallback_notice": narration.get("fallback_notice")}

def _format_result(result: Dict) -> Dict:
    if not result.get("success"):
        return {"success": False, "error": result.get("error", "Analysis failed"), "warnings": result.get("warnings", [])}
    vision = result.get("vision_result", {})
    narration = result.get("narration", {})
    response = {"success": True, "response_text": result.get("response_text", ""), "sources": result.get("sources", []), "evidence_count": result.get("evidence_count", 0), "is_demo": result.get("is_demo", False), "input_type": result.get("input_type", "unknown"), "llm_provider": result.get("llm_provider", "unknown"), "translation_info": result.get("translation_info", {})}
    if vision:
        response["vision"] = {"prediction": vision.get("prediction"), "confidence": vision.get("confidence", 0), "risk_level": vision.get("risk_level", "unknown"), "crop_detected": vision.get("crop_detected"), "disease_detected": vision.get("disease_detected"), "all_predictions": vision.get("all_predictions", []), "warnings": vision.get("warnings", []), "image_quality": vision.get("image_quality", {}), "reason": vision.get("reason", ""), "uncertain": vision.get("uncertain", False)}
    if narration and narration.get("audio_b64"):
        response["narration"] = {"audio_base64": narration["audio_b64"], "format": narration.get("format", "mp3"), "mime_type": narration.get("mime_type", "audio/mpeg"), "is_fallback": narration.get("is_fallback", False), "fallback_notice": narration.get("fallback_notice")}
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
