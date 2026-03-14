from pydantic import BaseModel
from pathlib import Path


class Settings(BaseModel):
    model_path: str = "runs/detect/train5/weights/best.pt"

    conf_thres: float = 0.35
    iou_thres: float = 0.5

    sample_fps: float = 5.0
    fire_consecutive: int = 3

    artifacts_dir: str = "artifacts"

    def artifacts_path(self) -> Path:
        p = Path(self.artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()