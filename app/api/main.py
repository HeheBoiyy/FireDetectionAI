from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.services.model import YoloService
from app.services.video import process_video

app = FastAPI(title="Fire/Smoke Early Detection API", version="1.0.0")

yolo: Optional[YoloService] = None


@app.on_event("startup")
def _startup():
    global yolo
    yolo = YoloService(
        weights_path=settings.model_path,
        conf_thres=settings.conf_thres,
        iou_thres=settings.iou_thres,
        device=0,  # если нет GPU, поставь None или "cpu"
    )
    settings.artifacts_path()


@app.get("/health")
def health():
    return {"status": "ok", "model_path": settings.model_path}


@app.post("/detect/frame")
async def detect_frame(file: UploadFile = File(...)):
    assert yolo is not None

    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Не удалось декодировать изображение"})

    dets = yolo.predict(img)
    return {
        "filename": file.filename,
        "detections": [d.__dict__ for d in dets],
    }


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    assert yolo is not None

    artifacts = settings.artifacts_path()
    inp = artifacts / f"upload_{Path(file.filename).name}"
    with inp.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    result = process_video(
        video_path=str(inp),
        yolo=yolo,
        artifacts_dir=artifacts,
        sample_fps=settings.sample_fps,
        window_sec=settings.window_sec,
        min_hits_ratio=settings.min_hits_ratio,
        fire_consecutive=settings.fire_consecutive,
    )
    return result