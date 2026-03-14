from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.services.model import Detection, YoloService


@dataclass
class FrameResult:
    frame_index: int
    timestamp_ms: int
    detections: list[Detection]


@dataclass
class Event:
    level: str  # "ALARM"
    timestamp_ms: int
    reason: str


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.xyxy)
        label = f"{d.class_name} {d.conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def iter_sampled_frames(cap: cv2.VideoCapture, sample_fps: float) -> Iterable[tuple[int, int, np.ndarray]]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    step = max(1, int(round(fps / sample_fps)))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ts_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            yield idx, ts_ms, frame
        idx += 1


def _make_writer(out_path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    # пробуем H.264, если не получилось — mp4v
    for fourcc_str in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), size)
        if writer.isOpened():
            return writer
    raise RuntimeError("Не удалось открыть VideoWriter (кодеки H264/mp4v недоступны). Установи ffmpeg или OpenCV с кодеками.")


def process_video(
    video_path: str,
    yolo: YoloService,
    artifacts_dir: Path,
    sample_fps: float,
    fire_consecutive: int,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    out_path = artifacts_dir / (Path(video_path).stem + "_annotated.mp4")
    writer = _make_writer(out_path, fps=float(sample_fps), size=(w, h))

    events: list[Event] = []
    fire_run = 0
    fired_alarm = False

    frames_processed = 0
    max_conf = 0.0

    for frame_idx, ts_ms, frame in iter_sampled_frames(cap, sample_fps=sample_fps):
        dets = yolo.predict(frame)
        frames_processed += 1

        if dets:
            max_conf = max(max_conf, max(d.conf for d in dets))

        # у тебя один класс fire; но оставим проверку имени на всякий
        has_fire = any(d.class_name.lower() in ("fire", "flame") for d in dets) or (len(dets) > 0)

        fire_run = fire_run + 1 if has_fire else 0

        if (not fired_alarm) and fire_run >= fire_consecutive:
            fired_alarm = True
            events.append(Event(level="ALARM", timestamp_ms=ts_ms, reason=f"Огонь {fire_run} кадров подряд"))

        annotated = _draw_detections(frame, dets)
        writer.write(annotated)

    cap.release()
    writer.release()

    return {
        "video_in": str(video_path),
        "video_out": str(out_path),
        "source_fps": float(src_fps),
        "processed_fps": float(sample_fps),
        "frames_processed": int(frames_processed),
        "max_conf": float(max_conf),
        "events": [e.__dict__ for e in events],
    }