import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.language_service import LanguageService, SupportedLanguage
from services.recommendation_engine import RecommendationEngine

app = FastAPI(title="Kisan Ki Awaz - Answer AI Service", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
engine: Optional[RecommendationEngine] = None

def get_engine():
    global engine
    if engine is None:
        engine = RecommendationEngine()
    return engine

class VoiceRequest(BaseModel):
    text: str
    language: str = "en"

class ImageAnswerRequest(BaseModel):
    question: str = ""
    language: str = "en"
    input_type: str = "upload"
    vision: dict

@app.get("/health")
async def health():
    e=get_engine()
    return {"status":"healthy","service":"selected-language-answer","llm_provider":e.llm_service.__class__.__name__}

@app.post("/voice")
async def voice(req: VoiceRequest):
    if not req.text.strip():
        raise HTTPException(400,"Question text cannot be empty.")
    try:
        lang=SupportedLanguage(req.language)
    except ValueError:
        raise HTTPException(400,f"Unsupported language: {req.language}")
    e=get_engine(); e.language_service.set_language(lang)
    return e.process_voice_query(req.text.strip())

@app.post("/image")
async def image(req: ImageAnswerRequest):
    try:
        lang=SupportedLanguage(req.language)
    except ValueError:
        raise HTTPException(400,f"Unsupported language: {req.language}")
    e=get_engine(); e.language_service.set_language(lang)
    vision=e._localize_vision_result(req.vision, req.language)
    if not vision.get("success", True):
        raise HTTPException(503, vision.get("error","Vision analysis failed"))
    crop=vision.get("crop_detected","") or "Unknown"
    disease=vision.get("disease_detected","")
    query=f"{crop} {disease}".strip() or crop
    if req.question: query=f"{query} {req.question}"
    evidence=e.rag_service.retrieve_evidence(query,crop=crop.lower() if crop else None,top_k=3)
    prompt=e.rag_service.build_rag_prompt(req.question or f"Analyze this {crop} image for disease/pest issues.", evidence=evidence,
        image_analysis={"prediction":vision.get("prediction","Unknown"),"confidence":vision.get("confidence",0),"risk_level":vision.get("risk_level","unknown")},
        language=req.language)
    try:
        llm=e.llm_service.generate(prompt)
        text, translation=e._localize_response(llm.get("text",""),req.language,fallback_prompt=prompt)
        provider=llm.get("provider","unknown"); demo=bool(llm.get("is_demo",False)) or bool(vision.get("is_demo",False))
    except Exception as exc:
        text,translation=e._image_fallback_response(vision,req.language,exc)
        provider="answer-fallback"; demo=True

    market_crop = e._detect_market_crop(query, evidence)
    market = e.market_service.get_crop_prices(market_crop) if market_crop else {}
    market_text = e._market_section(market, req.language)
    if market_text:
        text = text.rstrip() + "\n\n" + market_text

    narration=e.narration_service.prepare_narration(text)
    return {"success":True,"response_text":text,"vision_result":vision,
            "sources":e._localize_result_metadata(evidence.get("citations",[]),req.language),
            "evidence_count":evidence.get("evidence_count",0),"market":market,
            "narration":narration,"translation_info":translation,
            "llm_provider":provider,"is_demo":demo or bool(market.get("is_demo",False)),
            "input_type":req.input_type}
