from fastapi import FastAPI
app=FastAPI(title="Kisan Ki Awaz Language API")
@app.get("/health")
def health():
    return {"status":"healthy","service":"language"}
