# Usage

## 1. Prepare input images

Place soybean root-hair microscopy images in:

```text
examples/input_images/
```

Supported formats:

```text
.jpg
.jpeg
.png
.tif
.tiff
```

## 2. Run segmentation

```bash
python scripts/run_segmentation.py \
  --input examples/input_images \
  --output outputs/segmentation \
  --yolo-weights weights/last.pt \
  --sam3-weights weights/FinalFT.pt \
  --sam3-root external/sam3 \
  --device cuda
```

Outputs:

```text
outputs/segmentation/
├── detections/
│   └── all_detections.json
├── masks/
│   └── *_mask.png
├── overlays/
│   └── *_overlay.jpg
├── metadata/
│   └── *_instances.json
└── summary.json
```

## 3. Run measurement

```bash
python scripts/run_measurement.py \
  --mask-dir outputs/segmentation/masks \
  --metadata-dir outputs/segmentation/metadata \
  --output results/traits.csv \
  --per-instance-output results/per_instance_traits.csv \
  --annotated-dir results/annotated_measurements
```

Outputs:

```text
results/
├── traits.csv
├── per_instance_traits.csv
└── annotated_measurements/
    └── *_measurement.png
```

## Measurement mask format

The measurement module expects color-coded instance masks:

- black background = no instance
- each root-hair instance = unique non-black color
- main stem = one non-black instance

If metadata JSON files from segmentation are present, the measurement script uses class labels from metadata. If metadata is not present, it identifies the main stem as the largest colored instance.
