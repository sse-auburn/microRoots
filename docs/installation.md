# Installation

microRoots combines YOLOv12x-CBAM detection, fine-tuned SAM3 instance segmentation, and skeleton-based trait measurement.

## Recommended system

A CUDA-enabled GPU is recommended for full inference.

Browser-only editing is fine for setting up the repository, but actual model execution requires a Python environment with PyTorch, Ultralytics, OpenCV, and SAM3.

## Create environment

```bash
conda create -n microRoots python=3.10 -y
conda activate microRoots
pip install -r requirements.txt
pip install -e .
```

Alternative:

```bash
conda env create -f environment.yml
conda activate microRoots
pip install -e .
```

## Required weights

Download both trained weights and place them in `weights/`:

```text
weights/last.pt
weights/FinalFT.pt
```

See:

```text
weights/README.md
docs/model_weights.md
```

## SAM3 source code

The pipeline imports SAM3 using:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
```

Therefore, the SAM3 source code must be available locally. The recommended location is:

```text
external/sam3
```

See `docs/sam3_setup.md` for details.
