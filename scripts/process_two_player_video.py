"""Track two table-tennis players by side, with occlusion-aware pose metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from seima_mocap.landmarks import POSE_CONNECTIONS, RIGHT_WRIST


UPPER_BODY = (0, 11, 12, 13, 14, 15, 16, 23, 24)
PLAYER_CONFIG = {
    "left": {"label": "JUGADOR IZQUIERDO (NEGRO)", "color": (255, 210, 40)},
    "right": {"label": "JUGADOR DERECHO (ROJO)", "color": (65, 235, 80)},
}


@dataclass
class Track:
    side: str
    center: np.ndarray | None = None
    area: float | None = None
    last_landmarks: list | None = None
    missing_frames: int = 0
    wrist_trail: deque | None = None

    def __post_init__(self):
        self.wrist_trail = deque(maxlen=20)


@dataclass
class Detection:
    index: int
    center: np.ndarray
    area: float
    visibility: float
    presence: float
    upper_visibility: float
    completeness: float


@dataclass
class MappedLandmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


def quality(landmark, name: str) -> float:
    value = getattr(landmark, name, None)
    return 1.0 if value is None else float(np.clip(value, 0.0, 1.0))


def pixel(landmark, width: int, height: int) -> tuple[int, int]:
    return (
        int(np.clip(landmark.x, 0.0, 1.0) * (width - 1)),
        int(np.clip(landmark.y, 0.0, 1.0) * (height - 1)),
    )


def describe(index: int, landmarks) -> Detection:
    xy = np.array([[p.x, p.y] for p in landmarks], dtype=float)
    vis = np.array([quality(p, "visibility") for p in landmarks])
    pres = np.array([quality(p, "presence") for p in landmarks])
    reliable = (vis >= 0.25) & (pres >= 0.25)
    points = xy[reliable] if reliable.sum() >= 6 else xy
    low = np.percentile(points, 5, axis=0)
    high = np.percentile(points, 95, axis=0)
    size = np.maximum(high - low, 1e-5)
    return Detection(
        index=index,
        center=(low + high) / 2.0,
        area=float(np.clip(size[0] * size[1], 0.0, 1.0)),
        visibility=float(vis.mean()),
        presence=float(pres.mean()),
        upper_visibility=float(vis[list(UPPER_BODY)].mean()),
        completeness=float(reliable.mean()),
    )


def map_crop_landmarks(landmarks, x0: int, crop_width: int, full_width: int) -> list[MappedLandmark]:
    scale = crop_width / full_width
    return [
        MappedLandmark(
            x=(x0 + point.x * crop_width) / full_width,
            y=point.y,
            z=point.z * scale,
            visibility=quality(point, "visibility"),
            presence=quality(point, "presence"),
        )
        for point in landmarks
    ]


def assignment_score(detection: Detection, track: Track) -> float:
    expected_x = 0.25 if track.side == "left" else 0.75
    side_match = math.exp(-(((detection.center[0] - expected_x) / 0.28) ** 2))
    quality_score = math.sqrt(max(detection.upper_visibility * detection.presence, 0.0))
    if track.center is None:
        continuity = side_match
    else:
        distance = float(np.linalg.norm(detection.center - track.center))
        center_match = math.exp(-((distance / 0.20) ** 2))
        if track.area:
            scale_match = math.exp(-abs(math.log(max(detection.area, 1e-6) / track.area)))
        else:
            scale_match = 1.0
        continuity = 0.80 * center_match + 0.20 * scale_match
    return 0.50 * side_match + 0.30 * continuity + 0.20 * quality_score


def assign_detections(detections: list[Detection], tracks: dict[str, Track]) -> dict[str, Detection | None]:
    result: dict[str, Detection | None] = {"left": None, "right": None}
    if not detections:
        return result
    # Evaluate all unique assignments so identity continuity wins over output order.
    possibilities: list[tuple[float, str, Detection]] = []
    for side, track in tracks.items():
        for detection in detections:
            possibilities.append((assignment_score(detection, track), side, detection))
    used_sides: set[str] = set()
    used_detections: set[int] = set()
    for _, side, detection in sorted(possibilities, reverse=True, key=lambda item: item[0]):
        if side in used_sides or detection.index in used_detections:
            continue
        # Hard side constraint prevents identity swaps across the table center.
        if side == "left" and detection.center[0] >= 0.55:
            continue
        if side == "right" and detection.center[0] <= 0.45:
            continue
        result[side] = detection
        used_sides.add(side)
        used_detections.add(detection.index)
    return result


def draw_pose(frame, landmarks, track: Track, detected: bool) -> tuple[int, int] | None:
    height, width = frame.shape[:2]
    base_color = PLAYER_CONFIG[track.side]["color"]
    if not detected:
        base_color = tuple(int(channel * 0.45) for channel in base_color)
    points: dict[int, tuple[int, int]] = {}
    for idx, landmark in enumerate(landmarks):
        q = min(quality(landmark, "visibility"), quality(landmark, "presence"))
        if q >= 0.20 or (idx in (14, RIGHT_WRIST) and quality(landmark, "presence") >= 0.20):
            points[idx] = pixel(landmark, width, height)

    for a, b in POSE_CONNECTIONS:
        if a in points and b in points:
            racket_arm = (a, b) in ((12, 14), (14, 16))
            color = (255, 60, 220) if racket_arm and detected else base_color
            cv2.line(frame, points[a], points[b], color, 5 if racket_arm else 3, cv2.LINE_AA)
    for idx, point in points.items():
        if idx != RIGHT_WRIST:
            cv2.circle(frame, point, 4, base_color, -1, cv2.LINE_AA)

    wrist = points.get(RIGHT_WRIST)
    if wrist is None:
        return None
    wrist_visibility = quality(landmarks[RIGHT_WRIST], "visibility")
    reliable = wrist_visibility >= 0.50 and detected
    wrist_color = (255, 45, 220) if reliable else (0, 165, 255)
    if detected:
        track.wrist_trail.append(wrist)
    trail = list(track.wrist_trail)
    for i in range(1, len(trail)):
        cv2.line(frame, trail[i - 1], trail[i], wrist_color, 2, cv2.LINE_AA)
    cv2.circle(frame, wrist, 18, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, wrist, 11, wrist_color, -1, cv2.LINE_AA)
    cv2.circle(frame, wrist, 4, (255, 255, 255), -1, cv2.LINE_AA)
    label = f"{track.side.upper()} RW" if reliable else f"{track.side.upper()} RW EST."
    cv2.putText(frame, label, (wrist[0] + 18, max(28, wrist[1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, wrist_color, 2, cv2.LINE_AA)
    return wrist


def draw_bottom_panels(frame, states: dict[str, dict]) -> None:
    height, width = frame.shape[:2]
    panel_width, panel_height, gap = 535, 148, 24
    total_width = panel_width * 2 + gap
    start_x = (width - total_width) // 2
    top = height - panel_height - 18
    overlay = frame.copy()
    for offset, side in enumerate(("left", "right")):
        x = start_x + offset * (panel_width + gap)
        cv2.rectangle(overlay, (x, top), (x + panel_width, top + panel_height), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    for offset, side in enumerate(("left", "right")):
        state = states[side]
        x = start_x + offset * (panel_width + gap)
        color = PLAYER_CONFIG[side]["color"]
        status_color = (65, 235, 80) if state["detected"] else (0, 165, 255)
        lines = [
            (PLAYER_CONFIG[side]["label"], color),
            (f"Estado: {'DETECTADO' if state['detected'] else 'OCLUSION / PREDICCION'}", status_color),
            (f"Calidad pose: {state['confidence']:.3f} | continuidad: {state['continuity']:.3f}", (235, 235, 235)),
            (f"Muneca der. vis/pres: {state['wrist_visibility']:.3f} / {state['wrist_presence']:.3f}", (255, 100, 225)),
        ]
        y = top + 28
        for text, line_color in lines:
            cv2.putText(frame, text, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, line_color, 2, cv2.LINE_AA)
            y += 34


def empty_state(track: Track) -> dict:
    return {
        "detected": False,
        "confidence": 0.0,
        "continuity": 0.0,
        "wrist_visibility": 0.0,
        "wrist_presence": 0.0,
        "wrist_reliable": False,
        "wrist": None,
    }


def summarize(rows: list[dict], video_info: dict, output_path: Path) -> dict:
    summary: dict = {
        "video": video_info,
        "identity_strategy": "independent overlapping crop per table side plus temporal center/scale continuity",
        "occlusion_strategy": "retain the last pose for at most 12 frames as a visible prediction",
        "confidence_interpretation": "quality proxy, not a calibrated probability or biomechanical accuracy",
        "players": {},
        "caveats": [
            "RIGHT_WRIST is a racket-grip proxy; the racket itself is not detected.",
            "Orange wrist markers are low-visibility estimates, not reliable observations.",
            "The table strongly occludes lower-body landmarks, so upper-body quality is weighted more heavily.",
        ],
    }
    for side in ("left", "right"):
        detected = [r for r in rows if r[f"{side}_detected"]]
        reliable = [r for r in detected if r[f"{side}_wrist_reliable"]]

        def stat(field: str) -> dict:
            values = np.array([float(r[field]) for r in detected], dtype=float)
            if not len(values):
                return {"mean": None, "min": None, "p05": None, "median": None, "max": None}
            return {name: round(float(value), 5) for name, value in {
                "mean": values.mean(), "min": values.min(), "p05": np.percentile(values, 5),
                "median": np.median(values), "max": values.max(),
            }.items()}

        summary["players"][side] = {
            "label": PLAYER_CONFIG[side]["label"],
            "frames_detected": len(detected),
            "detection_rate": round(len(detected) / max(len(rows), 1), 5),
            "frames_occluded": len(rows) - len(detected),
            "right_wrist_reliable_frames": len(reliable),
            "right_wrist_reliable_rate": round(len(reliable) / max(len(rows), 1), 5),
            "pose_confidence": stat(f"{side}_confidence"),
            "upper_body_visibility": stat(f"{side}_upper_visibility"),
            "right_wrist_visibility": stat(f"{side}_wrist_visibility"),
            "continuity": stat(f"{side}_continuity"),
        }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def process(input_path: Path, output_dir: Path, model_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    silent_path = output_dir / f"{stem}_two_players_silent.mp4"
    video_path = output_dir / f"{stem}_two_players_annotated.mp4"
    csv_path = output_dir / f"{stem}_two_players_metrics.csv"
    summary_path = output_dir / f"{stem}_two_players_summary.json"

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {silent_path}")

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        min_tracking_confidence=0.25,
        output_segmentation_masks=False,
    )
    tracks = {side: Track(side) for side in ("left", "right")}
    rows: list[dict] = []
    last_timestamp_ms = -1

    try:
        with ExitStack() as stack:
            landmarkers = {
                side: stack.enter_context(mp.tasks.vision.PoseLandmarker.create_from_options(options))
                for side in ("left", "right")
            }
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_s = frame_index / fps
                timestamp_ms = max(last_timestamp_ms + 1, int(round(timestamp_s * 1000)))
                last_timestamp_ms = timestamp_ms
                crop_bounds = {
                    "left": (0, int(round(width * 0.56))),
                    "right": (int(round(width * 0.44)), width),
                }
                side_landmarks: dict[str, list | None] = {}
                for side, (x0, x1) in crop_bounds.items():
                    crop_rgb = cv2.cvtColor(frame[:, x0:x1], cv2.COLOR_BGR2RGB)
                    side_result = landmarkers[side].detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb), timestamp_ms
                    )
                    side_landmarks[side] = (
                        map_crop_landmarks(side_result.pose_landmarks[0], x0, x1 - x0, width)
                        if side_result.pose_landmarks else None
                    )
                states: dict[str, dict] = {}
                row: dict = {
                    "frame": frame_index,
                    "timestamp_s": round(timestamp_s, 6),
                    "poses_detected": sum(value is not None for value in side_landmarks.values()),
                }

                for side, track in tracks.items():
                    landmarks = side_landmarks[side]
                    detection = describe(0, landmarks) if landmarks is not None else None
                    state = empty_state(track)
                    if detection is not None:
                        if track.center is None:
                            continuity = 1.0
                        else:
                            distance = float(np.linalg.norm(detection.center - track.center))
                            continuity = math.exp(-((distance / 0.20) ** 2))
                        confidence = float(np.clip(
                            0.40 * detection.upper_visibility + 0.20 * detection.visibility
                            + 0.15 * detection.presence + 0.10 * detection.completeness
                            + 0.15 * continuity, 0.0, 1.0
                        ))
                        track.center, track.area = detection.center, detection.area
                        track.last_landmarks = list(landmarks)
                        track.missing_frames = 0
                        wrist = draw_pose(frame, landmarks, track, True)
                        wrist_vis = quality(landmarks[RIGHT_WRIST], "visibility")
                        wrist_pres = quality(landmarks[RIGHT_WRIST], "presence")
                        state.update({
                            "detected": True, "confidence": confidence, "continuity": continuity,
                            "wrist_visibility": wrist_vis, "wrist_presence": wrist_pres,
                            "wrist_reliable": min(wrist_vis, wrist_pres) >= 0.50, "wrist": wrist,
                            "upper_visibility": detection.upper_visibility,
                        })
                    else:
                        track.missing_frames += 1
                        state["upper_visibility"] = 0.0
                        if track.last_landmarks is not None and track.missing_frames <= 12:
                            state["wrist"] = draw_pose(frame, track.last_landmarks, track, False)
                            state["wrist_visibility"] = quality(track.last_landmarks[RIGHT_WRIST], "visibility")
                            state["wrist_presence"] = quality(track.last_landmarks[RIGHT_WRIST], "presence")
                    states[side] = state
                    row.update({
                        f"{side}_detected": state["detected"],
                        f"{side}_occlusion_frames": track.missing_frames,
                        f"{side}_confidence": state["confidence"],
                        f"{side}_continuity": state["continuity"],
                        f"{side}_upper_visibility": state["upper_visibility"],
                        f"{side}_wrist_visibility": state["wrist_visibility"],
                        f"{side}_wrist_presence": state["wrist_presence"],
                        f"{side}_wrist_reliable": state["wrist_reliable"],
                        f"{side}_wrist_x_px": state["wrist"][0] if state["wrist"] else "",
                        f"{side}_wrist_y_px": state["wrist"][1] if state["wrist"] else "",
                    })

                draw_bottom_panels(frame, states)
                cv2.putText(frame, "Magenta: muneca fiable | Naranja: estimacion baja visibilidad", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame)
                rows.append(row)
                frame_index += 1
                if frame_index % 30 == 0:
                    print(f"Processed {frame_index}/{expected} frames", flush=True)
    finally:
        cap.release()
        writer.release()

    if not rows:
        raise RuntimeError("No frames decoded")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        output = csv.DictWriter(handle, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)
    info = {
        "input": str(input_path), "output": str(video_path), "width": width, "height": height,
        "fps": round(fps, 5), "frames": len(rows), "duration_s": round(len(rows) / fps, 5),
    }
    summary = summarize(rows, info, summary_path)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(silent_path), "-i", str(input_path),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k", str(video_path),
    ], check=True)
    silent_path.unlink(missing_ok=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_full.task"))
    args = parser.parse_args()
    process(args.input, args.output_dir, args.model)


if __name__ == "__main__":
    main()
