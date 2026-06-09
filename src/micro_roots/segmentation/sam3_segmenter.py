from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from PIL import Image

from micro_roots.utils.colors import get_distinct_colors


@dataclass
class SAM3Config:
    checkpoint: str | Path
    sam3_root: str | Path
    confidence_threshold: float = 0.90
    device: str = "cuda"


class SAM3Segmenter:
    """Fine-tuned SAM3 segmenter using YOLO boxes as instance prompts."""

    def __init__(self, config: SAM3Config):
        self.config = config
        self.model = None
        self.processor = None
        self.device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return device

    def load(self):
        checkpoint = Path(self.config.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

        sam3_root = Path(self.config.sam3_root)
        if sam3_root.exists() and str(sam3_root) not in sys.path:
            sys.path.insert(0, str(sam3_root))

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = build_sam3_image_model(
            checkpoint_path=str(checkpoint),
            device=self.device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=True,
            compile=False,
        )

        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=self.config.confidence_threshold,
        )
        return self

    @staticmethod
    def _to_numpy_mask(mask: Any) -> np.ndarray:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.dtype != bool:
            mask = mask > 0.5
        return mask.astype(np.uint8)

    @staticmethod
    def _to_float_score(score: Any) -> float:
        if isinstance(score, torch.Tensor):
            score = score.detach().cpu().numpy()
        try:
            return float(np.asarray(score).reshape(-1)[0])
        except Exception:
            return 0.0

    def segment_boxes(self, image_rgb: np.ndarray, boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Segment all YOLO boxes in one image."""
        if self.model is None or self.processor is None:
            self.load()
        if len(boxes) == 0:
            return []

        image_pil = Image.fromarray(image_rgb)
        try:
            inference_state = self.processor.set_image(image_pil)
        except Exception as exc:
            print(f"Warning: SAM3 could not set image: {exc}")
            return []

        instances: List[Dict[str, Any]] = []
        colors = get_distinct_colors(len(boxes) + 10)

        for idx, box in enumerate(boxes):
            try:
                input_box = np.array([box["x1"], box["y1"], box["x2"], box["y2"]], dtype=np.float32)
                with torch.inference_mode():
                    masks, scores, _ = self.model.predict_inst(
                        inference_state,
                        point_coords=None,
                        point_labels=None,
                        box=input_box[None, :],
                        multimask_output=False,
                    )

                mask = self._to_numpy_mask(masks[0])
                score = self._to_float_score(scores[0])
                color = colors[idx]

                instances.append(
                    {
                        "mask": mask,
                        "score": score,
                        "color": color,
                        "class": box.get("class_name", "unknown"),
                        "box": box,
                    }
                )
            except Exception as exc:
                print(f"Warning: SAM3 failed on box {idx}: {exc}")
                continue

        return instances

    @staticmethod
    def create_instance_mask(instances: List[Dict[str, Any]], img_h: int, img_w: int) -> np.ndarray:
        """Create RGB color-coded instance mask."""
        mask_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        for inst in instances:
            mask = inst["mask"]
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            color = inst["color"]
            for c in range(3):
                mask_image[:, :, c] = np.where(mask > 0, color[c], mask_image[:, :, c])
        return mask_image

    @staticmethod
    def create_overlay(image_rgb: np.ndarray, instances: List[Dict[str, Any]], alpha: float = 0.5) -> np.ndarray:
        """Create RGB overlay visualization."""
        overlay = image_rgb.copy()
        img_h, img_w = image_rgb.shape[:2]
        for inst in instances:
            mask = inst["mask"]
            if mask.shape != (img_h, img_w):
                mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            color = np.array(inst["color"], dtype=np.uint8)
            colored = np.zeros_like(overlay)
            colored[mask.astype(bool)] = color
            mask_bool = mask.astype(bool)
            blended = cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0)
            overlay[mask_bool] = blended[mask_bool]
        return overlay

    @staticmethod
    def metadata_from_instances(instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create JSON-serializable metadata for each segmented instance."""
        metadata = []
        for idx, inst in enumerate(instances, start=1):
            mask = inst["mask"]
            metadata.append(
                {
                    "instance_id": idx,
                    "class_name": inst.get("class", "unknown"),
                    "sam3_score": float(inst.get("score", 0.0)),
                    "color_rgb": [int(v) for v in inst.get("color", (0, 0, 0))],
                    "area_px": int(np.sum(mask > 0)),
                    "box": inst.get("box", {}),
                }
            )
        return metadata
