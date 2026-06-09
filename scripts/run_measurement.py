from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from micro_roots.measurement.trait_measurement import MeasurementConfig, measure_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run root-hair trait measurement on color-coded instance masks.")
    parser.add_argument("--mask-dir", required=True, help="Folder containing *_mask.png files.")
    parser.add_argument("--metadata-dir", default=None, help="Optional folder containing *_instances.json files.")
    parser.add_argument("--output", default="results/traits.csv", help="Output summary CSV path.")
    parser.add_argument("--per-instance-output", default="results/per_instance_traits.csv", help="Output per-instance CSV path.")
    parser.add_argument("--annotated-dir", default="results/annotated_measurements", help="Folder for annotated measurement images.")
    parser.add_argument("--conversion-factor", type=float, default=0.495, help="Micrometers per pixel.")
    parser.add_argument("--prune-threshold", type=int, default=5, help="Skeleton spur pruning threshold in pixels.")
    parser.add_argument("--width-max-distance", type=int, default=20, help="Maximum width sampling distance in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = measure_folder(
        mask_dir=args.mask_dir,
        output_csv=args.output,
        per_instance_csv=args.per_instance_output,
        metadata_dir=args.metadata_dir,
        annotated_dir=args.annotated_dir,
        config=MeasurementConfig(
            conversion_factor=args.conversion_factor,
            skeleton_prune_threshold_px=args.prune_threshold,
            width_max_distance_px=args.width_max_distance,
        ),
    )
    print("\nMeasurement complete.")
    print(f"Masks found: {summary['num_masks']}")
    print(f"Successful: {summary['num_success']}")
    print(f"Failed: {summary['num_failed']}")
    print(f"Summary CSV: {summary['output_csv']}")
    if summary.get("per_instance_csv"):
        print(f"Per-instance CSV: {summary['per_instance_csv']}")
    if summary["failed"]:
        print("\nFailures:")
        for row in summary["failed"]:
            print(f"- {row['image']}: {row['error']}")


if __name__ == "__main__":
    main()
