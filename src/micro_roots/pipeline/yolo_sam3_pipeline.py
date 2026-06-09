from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from micro_roots.detection.yolo_cbam_detector import YOLOCBAMDetector, YOLODetectionConfig
from micro_roots.segmentation.sam3_segmenter import SAM3Config, SAM3Segmenter
from micro_roots.utils.io_utils import ensure_dirs, list_images, read_image_rgb, safe_filename, save_image_rgb


@dataclass
class PipelineConfig:
    input_path: str | Path
    output_dir: str | Path
    yolo_weights: str | Path
    sam3_weights: str | Path
    sam3_root: str | Path
    device: str = "cuda"
    yolo_conf: float = 0.80
    yolo_iou: float = 0.70
    yolo_imgsz: int = 1024
    sam3_conf: float = 0.90


def run_yolo_sam3_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Run YOLO detection followed by SAM3 box-prompted instance segmentation."""
    output_dir = Path(config.output_dir)
    detections_dir = output_dir / "detections"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    metadata_dir = output_dir / "metadata"
    ensure_dirs(detections_dir, masks_dir, overlays_dir, metadata_dir)

    image_paths = list_images(config.input_path)
    if not image_paths:
        raise FileNotFoundError(f"No supported input images found in: {config.input_path}")

    detector = YOLOCBAMDetector(
        YOLODetectionConfig(
            weights=config.yolo_weights,
            confidence=config.yolo_conf,
            iou=config.yolo_iou,
            image_size=config.yolo_imgsz,
            device=config.device,
        )
    ).load()

    segmenter = SAM3Segmenter(
        SAM3Config(
            checkpoint=config.sam3_weights,
            sam3_root=config.sam3_root,
            confidence_threshold=config.sam3_conf,
            device=config.device,
        )
    ).load()

    all_detections: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    for image_path in tqdm(image_paths, desc="Segmenting images"):
        image_rgb = read_image_rgb(image_path)
        img_h, img_w = image_rgb.shape[:2]
        safe_name = safe_filename(image_path.stem)

        boxes = detector.detect(image_rgb)
        instances = segmenter.segment_boxes(image_rgb, boxes)

        instance_mask = segmenter.create_instance_mask(instances, img_h, img_w)
        overlay = segmenter.create_overlay(image_rgb, instances)
        metadata = segmenter.metadata_from_instances(instances)

        mask_path = masks_dir / f"{safe_name}_mask.png"
        overlay_path = overlays_dir / f"{safe_name}_overlay.jpg"
        metadata_path = metadata_dir / f"{safe_name}_instances.json"

        save_image_rgb(mask_path, instance_mask)
        save_image_rgb(overlay_path, overlay)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        all_detections[safe_name] = {
            "image_path": str(image_path),
            "image_height": img_h,
            "image_width": img_w,
            "boxes": boxes,
            "mask_path": str(mask_path),
            "overlay_path": str(overlay_path),
            "metadata_path": str(metadata_path),
        }

        summary_rows.append(
            {
                "image": image_path.name,
                "safe_name": safe_name,
                "num_yolo_boxes": len(boxes),
                "num_sam3_instances": len(instances),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
                "metadata_path": str(metadata_path),
            }
        )

    (detections_dir / "all_detections.json").write_text(
        json.dumps(all_detections, indent=2), encoding="utf-8"
    )

    summary = {
        "num_images": len(image_paths),
        "output_dir": str(output_dir),
        "yolo_weights": str(config.yolo_weights),
        "sam3_weights": str(config.sam3_weights),
        "sam3_root": str(config.sam3_root),
        "device": config.device,
        "rows": summary_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
