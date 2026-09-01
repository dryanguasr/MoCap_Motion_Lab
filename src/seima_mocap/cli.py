"""Webcam prototype for markerless human-motion analysis."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .filters import FilteredDerivative
from .kinematics import JointKinematicsEstimator
from .landmarks import LANDMARK_NAMES, POSE_CONNECTIONS, RIGHT_WRIST
from .recorder import SessionRecorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SEIMA-MoCap webcam prototype using MediaPipe Pose Landmarker."
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_full.task"),
        help="Path to the MediaPipe Pose Landmarker model bundle.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="EMA smoothing factor in (0,1]. Smaller values smooth more.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="Minimum landmark visibility used for kinematic calculations.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record landmark and joint data to data/processed/.",
    )
    return parser


def _visibility(landmark) -> float:
    value = getattr(landmark, "visibility", None)
    return 1.0 if value is None else float(value)


def _draw_pose(frame, normalized_landmarks, min_visibility: float):
    height, width = frame.shape[:2]
    pixels: dict[int, tuple[int, int]] = {}

    for idx, landmark in enumerate(normalized_landmarks):
        if _visibility(landmark) < min_visibility:
            continue
        x = int(np.clip(landmark.x, 0.0, 1.0) * (width - 1))
        y = int(np.clip(landmark.y, 0.0, 1.0) * (height - 1))
        pixels[idx] = (x, y)

    for a, b in POSE_CONNECTIONS:
        if a in pixels and b in pixels:
            cv2.line(frame, pixels[a], pixels[b], (220, 220, 220), 2)

    for idx, point in pixels.items():
        radius = 7 if idx == RIGHT_WRIST else 4
        color = (0, 220, 255) if idx == RIGHT_WRIST else (0, 200, 0)
        cv2.circle(frame, point, radius, color, -1)


def _overlay_text(frame, wrist_state, joint_estimates, fps: float):
    lines = [f"FPS: {fps:5.1f}"]

    if wrist_state is None:
        lines.append("Right wrist: unavailable")
    else:
        position, velocity = wrist_state
        speed = float(np.linalg.norm(velocity))
        lines.extend(
            [
                "Dominant wrist assumption: RIGHT",
                f"p [m] = ({position[0]:+.3f}, {position[1]:+.3f}, {position[2]:+.3f})",
                f"v [m/s] = ({velocity[0]:+.3f}, {velocity[1]:+.3f}, {velocity[2]:+.3f})",
                f"|v| = {speed:.3f} m/s",
            ]
        )

    right_elbow = joint_estimates.get("right_elbow")
    if right_elbow is not None:
        lines.append(
            f"Right elbow: {right_elbow.angle_deg:.1f} deg | "
            f"{right_elbow.angular_velocity_deg_s:+.1f} deg/s"
        )

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26


def run(args: argparse.Namespace) -> None:
    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run 'python scripts/download_model.py' first."
        )
    if not 0.0 < args.alpha <= 1.0:
        raise ValueError("--alpha must be in (0, 1].")
    if not 0.0 <= args.min_visibility <= 1.0:
        raise ValueError("--min-visibility must be in [0, 1].")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}.")

    base_options = mp.tasks.BaseOptions(model_asset_path=str(args.model.resolve()))
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )

    landmark_filters = {
        idx: FilteredDerivative(args.alpha, args.alpha)
        for idx in range(len(LANDMARK_NAMES))
    }
    joint_estimator = JointKinematicsEstimator(alpha=args.alpha)

    recorder = None
    if args.record:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        recorder = SessionRecorder(Path("data/processed") / session_name)
        print(f"Recording session to: {recorder.session_dir}")

    start = time.perf_counter()
    last_timestamp_ms = -1
    fps_filtered = 0.0
    previous_loop_time = start

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame could not be read; stopping.")
                    break

                now = time.perf_counter()
                timestamp_s = now - start
                timestamp_ms = max(last_timestamp_ms + 1, int(timestamp_s * 1000.0))
                last_timestamp_ms = timestamp_ms

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                wrist_state = None
                joint_estimates = {}

                if result.pose_landmarks and result.pose_world_landmarks:
                    normalized = result.pose_landmarks[0]
                    world = result.pose_world_landmarks[0]
                    _draw_pose(frame, normalized, args.min_visibility)

                    filtered_positions: dict[int, np.ndarray] = {}

                    for idx, landmark in enumerate(world):
                        visibility = _visibility(landmark)
                        if visibility < args.min_visibility:
                            continue

                        measurement = np.array([landmark.x, landmark.y, landmark.z], dtype=float)
                        position, velocity = landmark_filters[idx].update(
                            measurement, timestamp_s
                        )
                        filtered_positions[idx] = position

                        if idx == RIGHT_WRIST:
                            wrist_state = (position, velocity)

                        if recorder is not None:
                            recorder.write_landmark(
                                timestamp_s,
                                idx,
                                visibility,
                                position,
                                velocity,
                            )

                    joint_estimates = joint_estimator.update(
                        filtered_positions, timestamp_s
                    )

                    if recorder is not None:
                        for name, estimate in joint_estimates.items():
                            recorder.write_joint(timestamp_s, name, estimate)

                loop_dt = max(now - previous_loop_time, 1e-6)
                instantaneous_fps = 1.0 / loop_dt
                fps_filtered = (
                    instantaneous_fps
                    if fps_filtered == 0.0
                    else 0.1 * instantaneous_fps + 0.9 * fps_filtered
                )
                previous_loop_time = now

                _overlay_text(frame, wrist_state, joint_estimates, fps_filtered)
                cv2.imshow("SEIMA-MoCap | q or ESC to exit", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if recorder is not None:
            recorder.close()


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
