# SAM3 Setup

microRoots uses your fine-tuned SAM3 checkpoint through the SAM3 Python source code.

The segmentation module imports:

```python
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
```

## Recommended folder structure

Place or clone the SAM3 source code into:

```text
external/sam3
```

Expected structure:

```text
microRoots/
├── external/
│   └── sam3/
│       └── sam3/
│           ├── model_builder.py
│           └── model/
│               └── sam3_image_processor.py
```

## Passing SAM3 path manually

If your SAM3 source code is somewhere else, pass its path:

```bash
python scripts/run_segmentation.py \
  --input examples/input_images \
  --output outputs/segmentation \
  --yolo-weights weights/last.pt \
  --sam3-weights weights/FinalFT.pt \
  --sam3-root /path/to/sam3 \
  --device cuda
```

## Official reference

Use the official Meta SAM3 repository and documentation for installation details:

```text
https://github.com/facebookresearch/sam3
https://ai.meta.com/research/sam3/
```
