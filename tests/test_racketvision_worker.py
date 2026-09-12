from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np


def _worker_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_racketvision_worker.py"
    spec = importlib.util.spec_from_file_location("seima_racketvision_worker_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_frames_uses_bounded_temporal_samples(tmp_path):
    worker = _worker_module()
    video = tmp_path / "tiny.mp4"
    frames = tmp_path / "frames"
    frames.mkdir()
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()

    paths, median_path, fps, image_shape = worker.extract_frames(video, frames)

    assert len(paths) == 20
    assert median_path.exists()
    assert fps == 10.0
    assert image_shape == (48, 64)
