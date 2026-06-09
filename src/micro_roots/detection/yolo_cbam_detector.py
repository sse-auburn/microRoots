from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class YOLODetectionConfig:
    weights: str | Path
    confidence: float = 0.80
    iou: float = 0.70
    image_size: int = 1024
    device: str | None = "cuda"


class YOLOCBAMDetector:
    """YOLOv12x + CBAM detector loaded from a trained Ultralytics `.pt` file."""

    def __init__(self, config: YOLODetectionConfig):
        self.config = config
        self.model = None

    def load(self):
        from ultralytics import YOLO

        weights = Path(self.config.weights)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")

        self.model = YOLO(str(weights))
        return self

    @property
    def names(self) -> Dict[int, str]:
        if self.model is None:
            return {}
        names = self.model.names
        if isinstance(names, dict):
            return names
        return {i: name for i, name in enumerate(names)}

    def detect(self, image_rgb: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO detection on one RGB image and return box dictionaries."""
        if self.model is None:
            self.load()

        kwargs = dict(
            conf=self.config.confidence,
            iou=self.config.iou,
            imgsz=self.config.image_size,
            verbose=False,
        )
        if self.config.device:
            kwargs["device"] = self.config.device

        results = self.model(image_rgb, **kwargs)
        boxes_out: List[Dict[str, Any]] = []

        if not results or results[0].boxes is None:
            return boxes_out

        for box in results[0].boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().numpy()]
            cls_id = int(box.cls[0].detach().cpu().numpy())
            conf = float(box.conf[0].detach().cpu().numpy())
            class_name = self.names.get(cls_id, str(cls_id))
            boxes_out.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "conf": conf,
                    "class": cls_id,
                    "class_name": class_name,
                }
            )

        return boxes_out
