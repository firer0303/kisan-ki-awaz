import base64,io
from fastapi import FastAPI,HTTPException
from PIL import Image
from services.vision_service import VisionService
app=FastAPI(title="Kisan Ki Awaz AI Vision",version="1.0.0")
vision=VisionService()
@app.get("/health")
def health():
    return {"status":"healthy","service":"ai-vision","is_demo":vision._is_demo}
@app.post("/analyze")
def analyze(payload:dict):
    try:
        raw=payload.get("image_base64","").split(",",1)[-1]
        image=Image.open(io.BytesIO(base64.b64decode(raw,validate=True)))
        image.load()
    except Exception as exc:
        raise HTTPException(400,"Invalid image data") from exc
    result=vision.analyze_image(image)
    if not result.get("success"):
        raise HTTPException(503,result.get("error","Image analysis failed"))
    return result
