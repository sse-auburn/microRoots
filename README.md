<p align="center">
  <img src="assets/microRoots.png" width="420">
</p>

# microRoots

microRoots is a soybean root-hair phenotyping pipeline for microscopy images. It uses YOLOv12 + CBAM for object detection, fine-tuned SAM3 for detector-prompted instance segmentation, and a separate trait-measurement module for extracting root-hair phenotyping traits.

## Pipeline

1. YOLOv12 + CBAM detects main-root and root-hair candidate regions.
2. Fine-tuned SAM3 generates instance-level masks from detector prompts.
3. microRoots measurement code extracts root-hair traits from the predicted masks.

## Model Components

### 1. YOLOv12 + CBAM Detector

The detector weights will be provided in the `weights/` section or release page.

Users should first follow the official YOLOv12/Ultralytics installation and inference guide.

Official guide: YOLO12 documentation.

Expected detector weight:



## Repository status

This repository is currently being prepared for public use. Installation instructions, model weights, example images, and inference scripts will be added progressively.
