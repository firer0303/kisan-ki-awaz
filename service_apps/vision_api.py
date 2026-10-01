from fastapi import FastAPI
from services.vision_service import VisionService
app=FastAPI(title="Kisan Ki Awaz AI Vision")
vision=VisionService()
@app.get("/health")
def health():
    return {"status":"healthy","service":"ai-vision","is_demo":vision._is_demo}
