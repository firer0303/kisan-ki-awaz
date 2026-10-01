"""
Kisan Ki Awaz - FastAPI Backend
=================================
REST API server exposing all AI services for the Android/web app.
"""
import base64
import io
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from services.language_service import LanguageService, SupportedLanguage
from services.recommendation_engine import RecommendationEngine

app = FastAPI(title="Kisan Ki Awaz API", description="AI Farming Assistant API for Pakistani Farmers", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
STATIC_DIR = PROJECT_ROOT / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

language_service = LanguageService()
engine: Optional[RecommendationEngine] = None

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
class ExplainRequest(BaseModel):
    result_id: str = ""

@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Kisan Ki Awaz API is running. Visit /docs for API documentation."}

@app.get("/api/health")
async def health():
    return {"status": "healthy", "service": "Kisan Ki Awaz", "version": "1.0.0"}

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
        result = eng.process_voice_query(req.text.strip())
        return _format_result(result)
    except Exception as exc:
        from loguru import logger
        logger.exception("Voice query failed")
        raise HTTPException(500, "AI response could not be generated. Please try again.") from exc

async def _run_image(engine_obj, image: Image.Image, question: str, input_type: str):
    try:
        result = engine_obj.process_image(image, question, input_type)
        return _format_result(result)
    except Exception as exc:
        from loguru import logger
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
    except (ValueError, UnidentifiedImageError, Exception) as exc:
        raise HTTPException(400, "Invalid image data. Please upload a valid JPG or PNG image.") from exc
    language_service.set_language(lang_enum)
    eng = get_engine()
    eng.language_service.set_language(lang_enum)
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
    except (UnidentifiedImageError, ValueError, Exception) as exc:
        raise HTTPException(400, "Invalid image file. Please upload a valid JPG or PNG image.") from exc
    language_service.set_language(lang_enum)
    eng = get_engine()
    eng.language_service.set_language(lang_enum)
    return await _run_image(eng, img, question, input_type)

# Preserve existing non-image endpoints from the original API below.
try:
    from api_legacy_endpoints import register_legacy_endpoints
    register_legacy_endpoints(app, get_engine)
except ImportError:
    pass


def _format_result(result):
    return result
