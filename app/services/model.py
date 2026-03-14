from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    conf: float
    xyxy: list[float]


class YoloService:
    def __init__(self, weights_path: str, conf_thres: float = 0.35, iou_thres: float = 0.5, device: Any = None):
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device  # можно None; ultralytics сам выберет

        # имена классов из модели (важно)
        self.names = self.model.names

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        # verbose=False чтобы не спамить в консоль
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        r0 = results[0]
        dets: list[Detection] = []

        if r0.boxes is None or len(r0.boxes) == 0:
            return dets

        boxes = r0.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(cls)):
            cid = int(cls[i])
            dets.append(
                Detection(
                    class_id=cid,
                    class_name=str(self.names.get(cid, cid)),
                    conf=float(conf[i]),
                    xyxy=[float(x) for x in xyxy[i].tolist()],
                )
            )
        return dets