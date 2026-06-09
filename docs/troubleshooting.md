# Troubleshooting

## YOLO weight does not load

Check that this file exists:

```text
weights/last.pt
```

Install Ultralytics:

```bash
pip install ultralytics
```

## SAM3 import error

If you see:

```text
ModuleNotFoundError: No module named 'sam3'
```

then the SAM3 source path is incorrect.

Check your SAM3 folder and pass the path using:

```bash
--sam3-root /path/to/sam3
```

## CUDA error

If CUDA is not available, try:

```bash
--device cpu
```

CPU inference may be slow and may not be practical for large images.

## No detections

Try lowering the YOLO confidence threshold:

```bash
--yolo-conf 0.50
```

## Measurement identifies the wrong main stem

If metadata is not provided, the measurement module identifies the largest colored instance as the main stem.

To improve reliability, run measurement with the segmentation metadata folder:

```bash
--metadata-dir outputs/segmentation/metadata
```

## Google Drive download fails

Large Google Drive files can sometimes fail through automated download scripts. Download manually from the browser and place the files in `weights/`.
