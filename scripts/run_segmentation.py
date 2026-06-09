from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from micro_roots.pipeline.yolo_sam3_pipeline import PipelineConfig, run_yolo_sam3_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv12x-CBAM + fine-tuned SAM3 segmentation.")
    parser.add_argument("--input", required=True, help="Input image file or folder.")
    parser.add_argument("--output", default="outputs/segmentation", help="Output folder.")
    parser.add_argument("--yolo-weights", default="weights/last.pt", help="Path to YOLO last.pt.")
    parser.add_argument("--sam3-weights", default="weights/FinalFT.pt", help="Path to SAM3 FinalFT.pt.")
    parser.add_argument("--sam3-root", default="external/sam3", help="Path to SAM3 source root.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--yolo-conf", type=float, default=0.80, help="YOLO confidence threshold.")
    parser.add_argument("--yolo-iou", type=float, default=0.70, help="YOLO NMS IoU threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=1024, help="YOLO inference image size.")
    parser.add_argument("--sam3-conf", type=float, default=0.90, help="SAM3 confidence threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_yolo_sam3_pipeline(
        PipelineConfig(
            input_path=args.input,
            output_dir=args.output,
            yolo_weights=args.yolo_weights,
            sam3_weights=args.sam3_weights,
            sam3_root=args.sam3_root,
            device=args.device,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_imgsz=args.yolo_imgsz,
            sam3_conf=args.sam3_conf,
        )
    )
    print("\nSegmentation complete.")
    print(f"Images processed: {summary['num_images']}")
    print(f"Output folder: {summary['output_dir']}")


if __name__ == "__main__":
    main()
