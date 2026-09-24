"""Run pinned RacketVision models on an arbitrary video and emit SEIMA cache v1."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from seima_mocap.racketvision_adapter import UPSTREAM_COMMIT, sha256_file, write_cache


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def timestamps_for_video(path: Path, frame_count: int, fps: float) -> list[float]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "frame=best_effort_timestamp_time", "-of", "json", str(path)]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
        values = [float(item["best_effort_timestamp_time"]) for item in payload["frames"]]
        if len(values) == frame_count and all(b > a for a, b in zip(values, values[1:])):
            return [value - values[0] for value in values]
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError):
        pass
    return [index / fps for index in range(frame_count)]


def extract_frames(video: Path, directory: Path):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = set(np.linspace(0, max(expected - 1, 0), min(17, expected), dtype=int))
    paths, samples = [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        path = directory / f"{index:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 96]):
            raise RuntimeError(f"cannot write {path}")
        paths.append(path)
        if index in sample_indices:
            samples.append(frame.copy())
        index += 1
    cap.release()
    if len(paths) < 3:
        raise ValueError("video must contain at least three frames")
    if len(samples) < 3:
        raise ValueError("video must contain at least three sampled frames")
    median = np.median(np.stack(samples), axis=0).astype(np.uint8)
    median_path = directory / "median.npz"
    np.savez_compressed(median_path, median=median)
    return paths, median_path, fps, samples[0].shape[:2]


def reuse_extracted_frames(video: Path, directory: Path):
    """Load a previously extracted frame sequence without touching the source video."""
    paths = sorted(directory.glob("*.jpg"))
    if len(paths) < 3:
        raise ValueError(f"need at least three JPEG frames in {directory}")
    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"cannot decode {paths[0]}")
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise ValueError(f"cannot read FPS from {video}")
    median_path = directory / "median.npz"
    if not median_path.exists():
        targets = np.linspace(0, len(paths) - 1, min(17, len(paths)), dtype=int)
        samples = [cv2.imread(str(paths[index])) for index in targets]
        if any(sample is None for sample in samples):
            raise ValueError(f"cannot decode sampled frame in {directory}")
        np.savez_compressed(median_path, median=np.median(np.stack(samples), axis=0).astype(np.uint8))
    return paths, median_path, fps, first.shape[:2]


class PeakCollectorMixin:
    thresholds = (.5, .35, .2)
    max_peaks = 5

    def _predict_location(self, heatmap):
        if getattr(self, "capture_heatmaps", True):
            self.captured_heatmaps.append(np.asarray(heatmap, dtype=np.float16))
        peaks = []
        for threshold in self.thresholds:
            mask = (heatmap > threshold).astype("uint8") * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                confidence = float(np.mean(heatmap[y:y + h, x:x + w]))
                center = np.array([x + w / 2, y + h / 2])
                if any(np.linalg.norm(center - old[0]) < 4 for old in peaks):
                    continue
                peaks.append((center, (x, y, w, h), confidence, threshold))
        peaks.sort(key=lambda item: -item[2])
        self.captured_peaks.append(peaks[:self.max_peaks])
        if not peaks:
            return 0, 0, 0, 0, 0.0
        _, bbox, confidence, _ = peaks[0]
        return *bbox, confidence


def run_balltrack(source_root: Path, frame_paths, median_path, device, batch_size, image_shape,
                  save_heatmaps=True, chunk_frames=64):
    module_root = source_root / "source" / "BallTrack"
    sys.path.insert(0, str(module_root))
    module = import_file("racketvision_ball_inference", module_root / "inference.py")

    class MultiPeakInferencer(PeakCollectorMixin, module.BallInferencer):
        def __init__(self, *args, **kwargs):
            self.captured_peaks = []
            self.captured_heatmaps = []
            self.capture_heatmaps = save_heatmaps
            super().__init__(*args, **kwargs)

    inferencer = MultiPeakInferencer(
        str(module_root / "configs" / "tracknetv3_base.py"),
        str(module_root / "checkpoints" / "balltrack_best.pth"),
        device=device, thre=.5, batchsize=batch_size,
    )
    if chunk_frames < inferencer.seq_len + 1:
        raise ValueError(f"chunk_frames must be at least {inferencer.seq_len + 1}")
    started = time.perf_counter()
    captured_peaks, captured_heatmaps = [], []
    for start in range(0, len(frame_paths), chunk_frames):
        end = min(len(frame_paths), start + chunk_frames)
        context_start = max(0, start - inferencer.seq_len)
        discard = start - context_start
        inferencer.captured_peaks = []
        inferencer.captured_heatmaps = []
        inferencer([str(path) for path in frame_paths[context_start:end]], str(median_path))
        captured_peaks.extend(inferencer.captured_peaks[discard:])
        if save_heatmaps:
            captured_heatmaps.extend(inferencer.captured_heatmaps[discard:])
        print(f"BallTrack {end}/{len(frame_paths)} frames", flush=True)
    elapsed_ms = 1000 * (time.perf_counter() - started)
    height, width = image_shape
    sx, sy = width / inferencer.width, height / inferencer.height
    output = []
    for peaks in captured_peaks:
        items = []
        for center, (_, _, w, h), confidence, threshold in peaks:
            items.append({"x_px": float(center[0] * sx), "y_px": float(center[1] * sy),
                          "confidence": confidence, "size_px": float(max(w * sx, h * sy)),
                          "source": "racketvision_balltrack",
                          "heatmap_threshold": threshold})
        output.append(items)
    heatmaps = np.stack(captured_heatmaps) if save_heatmaps else None
    return output, elapsed_ms / len(frame_paths), heatmaps


def run_racketpose(source_root: Path, frame_paths, device, bbox_threshold):
    module_root = source_root / "source" / "RacketPose"
    sys.path.insert(0, str(module_root))
    module = import_file("racketvision_racket_inference", module_root / "tools" / "inference.py")
    from mmdet.apis import init_detector
    from mmpose.apis import init_model as init_pose_model

    detector = init_detector(str(module_root / "configs/detection/rtmdet_m_racket_infer.py"),
                             str(module_root / "checkpoints/epoch_300.pth"), device=device)
    pose = init_pose_model(str(module_root / "configs/pose/rtmpose_m_racket_infer.py"),
                           str(module_root / "checkpoints/best_PCK_epoch_90.pth"), device=device)
    output, elapsed = [], []
    for path in frame_paths:
        image = cv2.imread(str(path))
        started = time.perf_counter()
        predictions = module.detect_and_estimate_pose(detector, pose, image, 1,
                                                       bbox_thr=bbox_threshold, max_instances=4)
        elapsed.append(1000 * (time.perf_counter() - started))
        items = []
        for prediction in predictions:
            keypoint_scores = np.asarray(prediction["keypoint_scores"], dtype=float)
            bbox_score = float(prediction["bbox_score"])
            items.append({"bbox_xyxy": prediction["bbox"][0],
                          "keypoints_xy": prediction["keypoints"],
                          "keypoint_scores": keypoint_scores.tolist(),
                          "confidence": float(bbox_score * np.mean(keypoint_scores)),
                          "source": "racketvision_racketpose"})
        output.append(items)
    return output, float(np.mean(elapsed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT / ".cache/racketvision/source")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bbox-threshold", type=float, default=.3)
    parser.add_argument("--skip-ball", action="store_true")
    parser.add_argument("--skip-racket", action="store_true")
    parser.add_argument("--skip-heatmap-cache", action="store_true",
                        help="Keep ball peaks but omit dense heatmaps (recommended for long videos).")
    parser.add_argument("--frames-dir", type=Path,
                        help="Reuse a complete directory of extracted JPEG frames after an interrupted run.")
    parser.add_argument("--ball-chunk-frames", type=int, default=64,
                        help="Maximum frames materialized at once by BallTrack (default: 64).")
    args = parser.parse_args()
    git_commit = subprocess.check_output(["git", "-C", str(args.source_root), "rev-parse", "HEAD"], text=True).strip()
    if git_commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"RacketVision source is {git_commit}; expected {UPSTREAM_COMMIT}")
    temporary = None
    if args.frames_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="seima-racketvision-")
        frame_paths, median_path, fps, image_shape = extract_frames(args.video, Path(temporary.name))
    else:
        frame_paths, median_path, fps, image_shape = reuse_extracted_frames(args.video, args.frames_dir)
    try:
        timestamps = timestamps_for_video(args.video, len(frame_paths), fps)
        if args.skip_ball:
            balls, ball_ms, heatmaps = [[] for _ in frame_paths], None, None
        else:
            balls, ball_ms, heatmaps = run_balltrack(
                args.source_root, frame_paths, median_path, args.device, args.batch_size, image_shape,
                save_heatmaps=not args.skip_heatmap_cache, chunk_frames=args.ball_chunk_frames)
        rackets, racket_ms = ([[] for _ in frame_paths], None) if args.skip_racket else run_racketpose(
            args.source_root, frame_paths, args.device, args.bbox_threshold)
        rows = ({"frame": index, "timestamp_s": timestamps[index],
                 "ball_candidates": balls[index], "rackets": rackets[index],
                 "inference_ms": None if ball_ms is None and racket_ms is None else (ball_ms or 0) + (racket_ms or 0)}
                for index in range(len(frame_paths)))
        checkpoint_root = args.source_root / "source"
        heatmap_path = args.output.with_suffix(".heatmaps.npz")
        if heatmaps is not None:
            heatmap_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(heatmap_path, heatmaps=heatmaps,
                                model_width=heatmaps.shape[2], model_height=heatmaps.shape[1])
        metadata = {
            "video": str(args.video), "video_sha256": sha256_file(args.video),
            "device": args.device, "frames": len(frame_paths),
            "ball_mean_inference_ms": ball_ms, "racket_mean_inference_ms": racket_ms,
            "heatmaps": None if heatmaps is None else {
                "path": str(heatmap_path), "shape": list(heatmaps.shape),
                "dtype": str(heatmaps.dtype), "sha256": sha256_file(heatmap_path),
            },
            "checkpoint_sha256": {
                "balltrack": sha256_file(checkpoint_root / "BallTrack/checkpoints/balltrack_best.pth") if not args.skip_ball else None,
                "racket_detector": sha256_file(checkpoint_root / "RacketPose/checkpoints/epoch_300.pth") if not args.skip_racket else None,
                "racket_pose": sha256_file(checkpoint_root / "RacketPose/checkpoints/best_PCK_epoch_90.pth") if not args.skip_racket else None,
            },
        }
        manifest = write_cache(args.output, rows, metadata)
    finally:
        if temporary is not None:
            temporary.cleanup()
    print(json.dumps({"cache": str(args.output), "manifest": str(manifest), **metadata}, indent=2))


if __name__ == "__main__":
    main()
