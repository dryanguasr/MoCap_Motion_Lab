"""Download the official MediaPipe Pose Landmarker Full model bundle."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DESTINATION = Path("models/pose_landmarker_full.task")


def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists():
        print(f"Model already exists: {DESTINATION}")
        return

    print(f"Downloading model to {DESTINATION} ...")
    urlretrieve(MODEL_URL, DESTINATION)
    print("Done.")


if __name__ == "__main__":
    main()
