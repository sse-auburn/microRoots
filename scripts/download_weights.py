from __future__ import annotations

from pathlib import Path

import gdown

WEIGHTS = {
    "last.pt": "1EyUSBc4SF1V-CwltEebyGszGYc2FV2L1",
    "FinalFT.pt": "1DccnrhCXTw7qAqmeAmEJd2EOgAtDx8gN",
}


def main() -> None:
    output_dir = Path("weights")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, file_id in WEIGHTS.items():
        output_path = output_dir / filename
        if output_path.exists():
            print(f"{filename} already exists. Skipping.")
            continue

        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)

    print("Done. If download failed, manually download the files from weights/README.md.")


if __name__ == "__main__":
    main()
