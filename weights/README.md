# Model Weights

The trained model weights are not stored directly in this GitHub repository because they are large files.

Download the two required model files from Google Drive and place them in this folder.

## Required files

| Model | File name | Description |
|---|---|---|
| YOLOv12x + CBAM detector | `last.pt` | Detects root-hair and main-stem candidate regions |
| Fine-tuned SAM3 segmenter | `FinalFT.pt` | Generates instance masks from YOLO bounding-box prompts |

## YOLOv12x + CBAM detector

Download from Google Drive:

```text
https://drive.google.com/file/d/1EyUSBc4SF1V-CwltEebyGszGYc2FV2L1/view?usp=sharing
```

Place the file as:

```text
weights/last.pt
```

## Fine-tuned SAM3 segmenter

Download from Google Drive:

```text
https://drive.google.com/file/d/1DccnrhCXTw7qAqmeAmEJd2EOgAtDx8gN/view?usp=sharing
```

Place the file as:

```text
weights/FinalFT.pt
```

## Final expected structure

```text
weights/
├── README.md
├── last.pt
└── FinalFT.pt
```

