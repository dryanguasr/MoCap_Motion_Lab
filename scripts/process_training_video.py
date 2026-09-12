"""Process a training video with foreground-aware MediaPipe pose tracking.

The generated confidence value is an interpretable tracking-quality proxy built
from MediaPipe landmark visibility/presence, landmark completeness, target
scale, and temporal continuity. It is not a calibrated probability of
biomechanical accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from seima_mocap.landmarks import LANDMARK_NAMES, POSE_CONNECTIONS, RIGHT_WRIST
from seima_mocap.output_layout import artifact_path, ensure_output_layout


@dataclass
class Candidate:
    index: int
    center: np.ndarray
    area: float
    mean_visibility: float
    mean_presence: float
    completeness: float
    continuity: float
    foreground: float
    score: float


def _quality(landmark, name: str) -> float:
    value = getattr(landmark, name, None)
    return 1.0 if value is None else float(np.clip(value, 0.0, 1.0))


def _candidate_features(landmarks, previous_center, previous_area) -> tuple:
    xy = np.array([[p.x, p.y] for p in landmarks], dtype=float)
    visibility = np.array([_quality(p, "visibility") for p in landmarks])
    presence = np.array([_quality(p, "presence") for p in landmarks])
    reliable = (visibility >= 0.35) & (presence >= 0.35)
    points = xy[reliable] if reliable.sum() >= 8 else xy
    low = np.percentile(points, 5, axis=0)
    high = np.percentile(points, 95, axis=0)
    width, height = np.maximum(high - low, 1e-5)
    area = float(np.clip(width * height, 0.0, 1.0))
    center = (low + high) / 2.0
    completeness = float(reliable.mean())

    # Nearby people are larger and normally extend farther down the image.
    scale_term = float(np.clip(math.sqrt(area / 0.22), 0.0, 1.0))
    lower_term = float(np.clip((center[1] - 0.25) / 0.55, 0.0, 1.0))
    foreground = 0.80 * scale_term + 0.20 * lower_term

    if previous_center is None or previous_area is None:
        continuity = foreground
    else:
        distance = float(np.linalg.norm(center - previous_center))
        center_match = math.exp(-((distance / 0.18) ** 2))
        ratio = max(area, 1e-6) / max(previous_area, 1e-6)
        scale_match = math.exp(-abs(math.log(ratio)))
        continuity = 0.75 * center_match + 0.25 * scale_match

    return (
        center,
        area,
        float(visibility.mean()),
        float(presence.mean()),
        completeness,
        float(continuity),
        float(foreground),
    )


def _select_candidate(all_landmarks, previous_center, previous_area) -> Candidate | None:
    candidates: list[Candidate] = []
    for index, landmarks in enumerate(all_landmarks):
        values = _candidate_features(landmarks, previous_center, previous_area)
        center, area, visibility, presence, completeness, continuity, foreground = values
        landmark_quality = math.sqrt(max(visibility * presence, 0.0))
        if previous_center is None:
            score = 0.68 * foreground + 0.22 * landmark_quality + 0.10 * completeness
        else:
            score = (
                0.45 * foreground
                + 0.35 * continuity
                + 0.15 * landmark_quality
                + 0.05 * completeness
            )
        candidates.append(
            Candidate(
                index,
                center,
                area,
                visibility,
                presence,
                completeness,
                continuity,
                foreground,
                float(np.clip(score, 0.0, 1.0)),
            )
        )
    return max(candidates, key=lambda item: item.score) if candidates else None


def _pixel(landmark, width: int, height: int) -> tuple[int, int]:
    return (
        int(np.clip(landmark.x, 0.0, 1.0) * (width - 1)),
        int(np.clip(landmark.y, 0.0, 1.0) * (height - 1)),
    )


def _draw_panel(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (625, 206), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    y = 49
    for text, color in lines:
        cv2.putText(frame, text, (35, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
        y += 34


def _draw_pose(frame, landmarks, quality_threshold: float, wrist_trail) -> tuple | None:
    height, width = frame.shape[:2]
    pixels: dict[int, tuple[int, int]] = {}
    for idx, landmark in enumerate(landmarks):
        quality = min(_quality(landmark, "visibility"), _quality(landmark, "presence"))
        # Keep the racket-side arm visible as an explicit low-confidence
        # estimate during occlusions; all other landmarks use the threshold.
        if quality >= quality_threshold or (
            idx in (14, RIGHT_WRIST) and _quality(landmark, "presence") >= 0.25
        ):
            pixels[idx] = _pixel(landmark, width, height)

    for a, b in POSE_CONNECTIONS:
        if a in pixels and b in pixels:
            is_racket_arm = (a, b) == (12, 14) or (a, b) == (14, 16)
            color = (255, 80, 210) if is_racket_arm else (40, 235, 95)
            cv2.line(frame, pixels[a], pixels[b], color, 6 if is_racket_arm else 3, cv2.LINE_AA)

    for idx, point in pixels.items():
        if idx == RIGHT_WRIST:
            continue
        cv2.circle(frame, point, 5, (20, 30, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 4, (70, 255, 120), -1, cv2.LINE_AA)

    wrist = pixels.get(RIGHT_WRIST)
    if wrist is not None:
        wrist_visibility = _quality(landmarks[RIGHT_WRIST], "visibility")
        reliable = wrist_visibility >= 0.50
        wrist_trail.append(wrist)
        trail = list(wrist_trail)
        for idx in range(1, len(trail)):
            strength = idx / max(len(trail) - 1, 1)
            cv2.line(frame, trail[idx - 1], trail[idx], (255, int(80 + 130 * strength), 220), 2, cv2.LINE_AA)
        wrist_color = (255, 45, 210) if reliable else (0, 165, 255)
        cv2.circle(frame, wrist, 20, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(frame, wrist, 13, wrist_color, -1, cv2.LINE_AA)
        cv2.circle(frame, wrist, 5, (255, 255, 255), -1, cv2.LINE_AA)
        label_x = min(wrist[0] + 25, width - 340)
        label_y = max(wrist[1] - 20, 35)
        cv2.putText(
            frame,
            "MUNECA DERECHA / PROXY RAQUETA" if reliable else "MUNECA DER. ESTIMADA / BAJA VIS.",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "MUNECA DERECHA / PROXY RAQUETA" if reliable else "MUNECA DER. ESTIMADA / BAJA VIS.",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            wrist_color,
            2,
            cv2.LINE_AA,
        )
    return wrist


def _write_summary(path: Path, rows: list[dict], video_info: dict) -> dict:
    detected = [row for row in rows if row["target_detected"]]
    wrist_rows = [row for row in detected if row["right_wrist_available"]]
    wrist_reliable_rows = [row for row in detected if row["right_wrist_reliable"]]

    def stats(key: str, source: list[dict]) -> dict:
        values = np.array([float(row[key]) for row in source], dtype=float)
        if not len(values):
            return {"mean": None, "median": None, "min": None, "p05": None, "max": None}
        return {
            "mean": round(float(values.mean()), 5),
            "median": round(float(np.median(values)), 5),
            "min": round(float(values.min()), 5),
            "p05": round(float(np.percentile(values, 5)), 5),
            "max": round(float(values.max()), 5),
        }

    summary = {
        "video": video_info,
        "target_definition": "foreground player with black shirt and red shorts",
        "selection_logic": {
            "foreground": "80% apparent pose scale + 20% vertical image position",
            "temporal": "pose-center distance and apparent-scale continuity",
            "quality": "mean MediaPipe visibility/presence and landmark completeness",
        },
        "confidence_interpretation": (
            "Tracking-quality proxy in [0,1]; not a calibrated probability and not "
            "a biomechanical accuracy guarantee."
        ),
        "frames_total": len(rows),
        "frames_target_detected": len(detected),
        "target_detection_rate": round(len(detected) / max(len(rows), 1), 5),
        "frames_right_wrist_available": len(wrist_rows),
        "right_wrist_availability_rate": round(len(wrist_rows) / max(len(rows), 1), 5),
        "frames_right_wrist_reliable": len(wrist_reliable_rows),
        "right_wrist_reliable_rate": round(len(wrist_reliable_rows) / max(len(rows), 1), 5),
        "target_confidence": stats("target_confidence", detected),
        "mean_landmark_visibility": stats("mean_visibility", detected),
        "mean_landmark_presence": stats("mean_presence", detected),
        "right_wrist_visibility": stats("right_wrist_visibility", wrist_rows),
        "right_wrist_presence": stats("right_wrist_presence", wrist_rows),
        "continuity": stats("continuity", detected),
        "pose_area_fraction": stats("pose_area_fraction", detected),
        "caveats": [
            "RIGHT_WRIST is an anatomical proxy correlated with the racket grip; the racket itself is not detected.",
            "Monocular 3D coordinates are model estimates, not triangulated measurements.",
            "Occlusion, motion blur, and a rear-facing body can reduce anatomical certainty.",
        ],
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def process(input_path: Path, output_dir: Path, model_path: Path) -> None:
    ensure_output_layout(output_dir)
    stem = input_path.stem
    silent_path = artifact_path(output_dir, "videos", stem, "pose", "silent", ".mp4")
    output_video = artifact_path(output_dir, "videos", stem, "pose", "annotated", ".mp4")
    csv_path = artifact_path(output_dir, "metrics", stem, "pose", "frame_metrics", ".csv")
    summary_path = artifact_path(output_dir, "summaries", stem, "pose", "summary", ".json")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {silent_path}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=4,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
        output_segmentation_masks=False,
    )

    rows: list[dict] = []
    previous_center = None
    previous_area = None
    previous_wrist = None
    previous_timestamp = None
    wrist_trail = deque(maxlen=max(8, int(round(fps * 0.65))))
    last_timestamp_ms = -1

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_s = frame_index / fps
                timestamp_ms = max(last_timestamp_ms + 1, int(round(timestamp_s * 1000)))
                last_timestamp_ms = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                selected = _select_candidate(result.pose_landmarks, previous_center, previous_area)
                row = {
                    "frame": frame_index,
                    "timestamp_s": round(timestamp_s, 6),
                    "poses_detected": len(result.pose_landmarks),
                    "target_detected": selected is not None,
                    "selected_pose_index": "",
                    "target_confidence": 0.0,
                    "mean_visibility": 0.0,
                    "mean_presence": 0.0,
                    "landmark_completeness": 0.0,
                    "foreground_score": 0.0,
                    "continuity": 0.0,
                    "pose_area_fraction": 0.0,
                    "right_wrist_available": False,
                    "right_wrist_reliable": False,
                    "right_wrist_visibility": 0.0,
                    "right_wrist_presence": 0.0,
                    "right_wrist_x_norm": "",
                    "right_wrist_y_norm": "",
                    "right_wrist_z_norm": "",
                    "right_wrist_x_px": "",
                    "right_wrist_y_px": "",
                    "right_wrist_speed_px_s": "",
                    "right_wrist_x_world_m": "",
                    "right_wrist_y_world_m": "",
                    "right_wrist_z_world_m": "",
                }

                if selected is not None:
                    landmarks = result.pose_landmarks[selected.index]
                    world = result.pose_world_landmarks[selected.index]
                    wrist_landmark = landmarks[RIGHT_WRIST]
                    wrist_visibility = _quality(wrist_landmark, "visibility")
                    wrist_presence = _quality(wrist_landmark, "presence")
                    wrist_available = wrist_presence >= 0.25
                    wrist_reliable = min(wrist_visibility, wrist_presence) >= 0.50
                    quality = math.sqrt(max(selected.mean_visibility * selected.mean_presence, 0.0))
                    target_confidence = float(
                        np.clip(
                            0.40 * quality
                            + 0.20 * selected.completeness
                            + 0.20 * selected.foreground
                            + 0.20 * selected.continuity,
                            0.0,
                            1.0,
                        )
                    )
                    wrist_point = _draw_pose(frame, landmarks, 0.25, wrist_trail)
                    previous_center = selected.center
                    previous_area = selected.area

                    speed_px_s = ""
                    if wrist_point is not None and previous_wrist is not None and previous_timestamp is not None:
                        dt = max(timestamp_s - previous_timestamp, 1e-6)
                        speed_px_s = float(np.linalg.norm(np.array(wrist_point) - previous_wrist) / dt)
                    if wrist_point is not None:
                        previous_wrist = np.array(wrist_point, dtype=float)
                        previous_timestamp = timestamp_s

                    row.update(
                        {
                            "selected_pose_index": selected.index,
                            "target_confidence": target_confidence,
                            "mean_visibility": selected.mean_visibility,
                            "mean_presence": selected.mean_presence,
                            "landmark_completeness": selected.completeness,
                            "foreground_score": selected.foreground,
                            "continuity": selected.continuity,
                            "pose_area_fraction": selected.area,
                            "right_wrist_available": wrist_available,
                            "right_wrist_reliable": wrist_reliable,
                            "right_wrist_visibility": wrist_visibility,
                            "right_wrist_presence": wrist_presence,
                            "right_wrist_x_norm": wrist_landmark.x,
                            "right_wrist_y_norm": wrist_landmark.y,
                            "right_wrist_z_norm": wrist_landmark.z,
                            "right_wrist_x_px": wrist_point[0] if wrist_point else "",
                            "right_wrist_y_px": wrist_point[1] if wrist_point else "",
                            "right_wrist_speed_px_s": speed_px_s,
                            "right_wrist_x_world_m": world[RIGHT_WRIST].x,
                            "right_wrist_y_world_m": world[RIGHT_WRIST].y,
                            "right_wrist_z_world_m": world[RIGHT_WRIST].z,
                        }
                    )
                    color = (80, 255, 120) if target_confidence >= 0.75 else (0, 210, 255)
                    _draw_panel(
                        frame,
                        [
                            ("OBJETIVO: JUGADOR EN PRIMER PLANO", (255, 255, 255)),
                            (f"Calidad seguimiento: {target_confidence:0.3f}", color),
                            (f"Visibilidad media: {selected.mean_visibility:0.3f}", (220, 220, 220)),
                            (f"Muneca der. vis/pres: {wrist_visibility:0.3f} / {wrist_presence:0.3f}", (255, 105, 225)),
                            (f"Poses candidatas: {len(result.pose_landmarks)} | continuidad: {selected.continuity:0.3f}", (220, 220, 220)),
                        ],
                    )
                else:
                    wrist_trail.clear()
                    _draw_panel(frame, [("OBJETIVO NO DETECTADO", (0, 80, 255))])

                cv2.putText(
                    frame,
                    "Indicadores de calidad del modelo; no medicion biomecanica calibrada",
                    (24, height - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                rows.append(row)
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Processed {frame_index}/{expected_frames} frames", flush=True)
    finally:
        cap.release()
        writer.release()

    if not rows:
        raise RuntimeError("No frames were decoded.")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    video_info = {
        "input": str(input_path),
        "output": str(output_video),
        "width": width,
        "height": height,
        "fps": round(fps, 5),
        "duration_s": round(len(rows) / fps, 5),
    }
    summary = _write_summary(summary_path, rows, video_info)

    command = [
        "ffmpeg", "-y", "-i", str(silent_path), "-i", str(input_path),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(output_video),
    ]
    subprocess.run(command, check=True)
    silent_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_full.task"))
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    process(args.input, args.output_dir, args.model)
