# Model Weights

The trained weights are hosted externally on Google Drive.

They are not included in the GitHub repository because they are large binary files.

## Required files

| File | Model | Purpose |
|---|---|---|
| `last.pt` | YOLOv12x + CBAM | Detection of root-hair and main-stem candidate boxes |
| `FinalFT.pt` | Fine-tuned SAM3 | Box-prompted instance segmentation |

## YOLOv12x + CBAM

Google Drive:

```text
https://drive.google.com/file/d/1EyUSBc4SF1V-CwltEebyGszGYc2FV2L1/view?usp=sharing
```

Expected path:

```text
weights/last.pt
```

## Fine-tuned SAM3

Google Drive:

```text
https://drive.google.com/file/d/1DccnrhCXTw7qAqmeAmEJd2EOgAtDx8gN/view?usp=sharing
```

Expected path:

```text
weights/FinalFT.pt
```

## Automatic download

An optional helper script is provided:

```bash
python scripts/download_weights.py
```

For very large files, Google Drive may sometimes require manual download in the browser.
