from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.inference import InferenceEngine
from src.utils import env_or_default

app = FastAPI(title="CV Assessment Inference API", version="1.0.0")

CHECKPOINT_PATH = env_or_default("CHECKPOINT_PATH", "artifacts/models/strong_best.pt")
IMAGE_SIZE = int(env_or_default("IMAGE_SIZE", "224"))
TOP_K = int(env_or_default("TOP_K", "3"))
ALLOWED_ORIGINS = env_or_default("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if Path(CHECKPOINT_PATH).exists():
    ENGINE = InferenceEngine(checkpoint_path=CHECKPOINT_PATH, image_size=IMAGE_SIZE, top_k=TOP_K)
else:
    ENGINE = None

@app.get("/")
def root():
    return {"message": "Welcome to the CV Assessment Inference API"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ENGINE is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if ENGINE is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded. Expected at {CHECKPOINT_PATH}")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        result = ENGINE.predict(temp_path)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return result
