from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def safe_filename(filename: str) -> str:
    """Convert a file stem to a filesystem-safe ASCII-like name."""
    safe_name = filename.replace(" ", "_")
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in safe_name)


def list_images(input_path: str | Path) -> List[Path]:
    """Return sorted image files from a file or folder."""
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image_rgb(path: str | Path) -> np.ndarray:
    """Read an image with OpenCV and return RGB array."""
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    """Save an RGB image with OpenCV, falling back to PIL if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image_rgb.ndim == 3:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_rgb
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        if image_rgb.ndim == 3:
            Image.fromarray(image_rgb).save(path)
        else:
            Image.fromarray(image_rgb).save(path)


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
