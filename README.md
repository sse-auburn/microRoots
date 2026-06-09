# microRoots

**microRoots** is a soybean root-hair phenotyping pipeline that combines:

1. **YOLOv12x + CBAM** for detecting root-hair (`RH`) and main-stem (`MS`) candidate regions.
2. **Fine-tuned SAM3** for detector-prompted instance segmentation using YOLO bounding boxes.
3. **Skeleton-based trait measurement** for extracting main-stem length, root-hair count, root-hair length, root-hair width, and root-hair density.

The trained weights are hosted externally on Google Drive and are **not** stored inside this repository.

---

## Pipeline overview

```text
Input soybean root-hair microscopy image
        |
        v
YOLOv12x + CBAM detector
        |
        v
Bounding boxes for RH and MS
        |
        v
Fine-tuned SAM3 box-prompted segmentation
        |
        v
Color-coded instance masks + overlay images + metadata JSON
        |
        v
Skeleton-based trait measurement
        |
        v
CSV files with root-hair phenotyping traits
```

---

## Repository structure

```text
microRoots/
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── inference_config.yaml
│   └── model_paths.yaml
├── docs/
│   ├── installation.md
│   ├── usage.md
│   ├── model_weights.md
│   ├── sam3_setup.md
│   ├── github_browser_upload.md
│   └── troubleshooting.md
├── weights/
│   └── README.md
├── examples/
│   ├── input_images/
│   ├── output_masks/
│   └── measurement_results/
├── scripts/
│   ├── download_weights.py
│   ├── run_segmentation.py
│   └── run_measurement.py
├── src/
│   └── micro_roots/
│       ├── detection/
│       ├── segmentation/
│       ├── pipeline/
│       ├── measurement/
│       └── utils/
└── legacy/
    └── original_user_scripts/
```

---

## Required model weights

Download both files from Google Drive and place them in the `weights/` folder.

| Model | Required file name | Purpose |
|---|---|---|
| YOLOv12x + CBAM | `last.pt` | Detects root-hair and main-stem candidate boxes |
| Fine-tuned SAM3 | `FinalFT.pt` | Generates box-prompted instance masks |

Links are provided in [`weights/README.md`](weights/README.md) and [`docs/model_weights.md`](docs/model_weights.md).

Expected final structure:

```text
weights/
├── README.md
├── last.pt
└── FinalFT.pt
```

Do **not** commit `.pt` model files to GitHub.

---

## Installation

A CUDA-capable GPU is recommended for YOLO + SAM3 inference.

```bash
conda create -n microRoots python=3.10 -y
conda activate microRoots
pip install -r requirements.txt
pip install -e .
```

SAM3 must also be installed or provided locally. See [`docs/sam3_setup.md`](docs/sam3_setup.md).

---

## Quick start

### 1. Add input images

Put microscopy images in:

```text
examples/input_images/
```

Supported formats:

```text
.jpg, .jpeg, .png, .tif, .tiff
```

### 2. Run YOLOv12x-CBAM + SAM3 segmentation

```bash
python scripts/run_segmentation.py \
  --input examples/input_images \
  --output outputs/segmentation \
  --yolo-weights weights/last.pt \
  --sam3-weights weights/FinalFT.pt \
  --sam3-root external/sam3 \
  --device cuda
```

Expected outputs:

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

### 3. Run root-hair trait measurement

```bash
python scripts/run_measurement.py \
  --mask-dir outputs/segmentation/masks \
  --metadata-dir outputs/segmentation/metadata \
  --output results/traits.csv \
  --per-instance-output results/per_instance_traits.csv \
  --annotated-dir results/annotated_measurements
```

Expected outputs:

```text
results/
├── traits.csv
├── per_instance_traits.csv
└── annotated_measurements/
    └── *_measurement.png
```

---

## Measurement traits

The measurement module reports:

| Trait | Description |
|---|---|
| Main-stem length | Best-fit line on the root-hair side of the main stem, extended to image edges |
| Root-hair count | Number of detected root-hair instances |
| Average root-hair length | Skeleton geodesic length averaged across root hairs |
| Average root-hair width | Perpendicular skeleton-based width sampling averaged across root hairs |
| Root-hair density | Root-hair count divided by main-stem length |

Default pixel-to-micron conversion factor:

```text
0.495 μm/pixel
```

This value can be changed in `configs/inference_config.yaml` or passed directly to the measurement script.

---

## Important notes

- The measurement code expects **color-coded instance masks**.
- Black pixels are treated as background.
- The segmentation pipeline also writes metadata JSON files, which help the measurement module identify `RH` and `MS` instances.
- If metadata is unavailable, the measurement module assumes that the largest colored instance is the main stem.

---

## Citation

If you use this repository, please cite the associated microRoots manuscript or project record when available.

---

## License

License is not finalized in this package. Add a suitable license before public release, for example MIT, Apache-2.0, BSD-3-Clause, or a university-approved research software license.
