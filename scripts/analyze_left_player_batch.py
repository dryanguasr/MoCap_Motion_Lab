"""Batch motion and heuristic semantic analysis for the left-side player."""

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

from seima_mocap.landmarks import POSE_CONNECTIONS, RIGHT_WRIST
from seima_mocap.output_layout import artifact_path, ensure_output_layout


UPPER = np.array([0, 11, 12, 13, 14, 15, 16, 23, 24])
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
RIGHT_ELBOW = 14
FOOT_LANDMARKS = np.array([27, 28, 29, 30, 31, 32])
TORSO_HEIGHT_FRACTION = 0.288


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float
    presence: float


def q(point, name: str) -> float:
    value = getattr(point, name, None)
    return 1.0 if value is None else float(np.clip(value, 0.0, 1.0))


def map_landmarks(points, crop_width: int, full_width: int) -> list[Landmark]:
    scale = crop_width / full_width
    return [
        Landmark(
            x=point.x * scale,
            y=point.y,
            z=point.z * scale,
            visibility=q(point, "visibility"),
            presence=q(point, "presence"),
        )
        for point in points
    ]


def point_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denominator <= 1e-9:
        return math.nan
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / denominator, -1.0, 1.0))))


def interpolate_and_smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full_like(values, np.nan)
    if values.ndim == 1:
        columns = values[:, None]
        output_columns = output[:, None]
    else:
        columns = values
        output_columns = output
    x = np.arange(len(values))
    kernel = np.ones(window, dtype=float) / window
    for column_index in range(columns.shape[1]):
        column = columns[:, column_index]
        valid = np.isfinite(column)
        if valid.sum() < 2:
            continue
        filled = np.interp(x, x[valid], column[valid])
        padded = np.pad(filled, (window // 2, window // 2), mode="edge")
        output_columns[:, column_index] = np.convolve(padded, kernel, mode="valid")[: len(values)]
    return output


def vector_speed(points: np.ndarray, fps: float, scales: np.ndarray | None = None) -> np.ndarray:
    smoothed = interpolate_and_smooth(points, 5)
    velocity = np.gradient(smoothed, 1.0 / fps, axis=0)
    if scales is not None:
        velocity = velocity * scales
    return np.linalg.norm(velocity, axis=1)


def detect_peaks(
    speed: np.ndarray, fps: float, reliable: np.ndarray, forward_motion: np.ndarray
) -> list[int]:
    usable = speed[np.isfinite(speed) & reliable]
    if len(usable) < max(10, int(fps)):
        return []
    median = float(np.median(usable))
    mad = float(np.median(np.abs(usable - median)))
    threshold = max(float(np.percentile(usable, 72)), median + 1.8 * mad, 1.25)
    candidates = [
        i for i in range(2, len(speed) - 2)
        if reliable[i] and forward_motion[i] and speed[i] >= threshold
        and speed[i] == np.max(speed[i - 2 : i + 3])
    ]
    minimum_gap = max(1, int(round(0.38 * fps)))
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: speed[item], reverse=True):
        if all(abs(index - other) >= minimum_gap for other in selected):
            selected.append(index)
    return sorted(selected)


def semantic_phases(frame_count: int, peaks: list[int], fps: float, speed: np.ndarray) -> tuple[list[str], list[int]]:
    phases = ["transicion_espera"] * frame_count
    event_ids = [-1] * frame_count
    windows = [
        (-0.42, -0.16, "preparacion"),
        (-0.16, -0.03, "aceleracion"),
        (-0.03, 0.04, "impacto_proxy"),
        (0.04, 0.28, "seguimiento"),
        (0.28, 0.55, "recuperacion"),
    ]
    for event_id, peak in enumerate(peaks, start=1):
        for start_s, end_s, label in windows:
            start = max(0, peak + int(round(start_s * fps)))
            end = min(frame_count, peak + int(round(end_s * fps)))
            for index in range(start, end):
                if event_ids[index] == -1 or abs(index - peak) < abs(index - peaks[event_ids[index] - 1]):
                    phases[index] = label
                    event_ids[index] = event_id
    finite = speed[np.isfinite(speed)]
    low_threshold = float(np.percentile(finite, 35)) if len(finite) else 0.0
    for index in range(frame_count):
        if event_ids[index] == -1 and np.isfinite(speed[index]) and speed[index] > low_threshold:
            phases[index] = "movimiento_no_clasificado"
    return phases, event_ids


def safe_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "median": None, "min": None, "p05": None, "p95": None, "p99": None, "max": None}
    return {
        "mean": round(float(values.mean()), 5),
        "median": round(float(np.median(values)), 5),
        "min": round(float(values.min()), 5),
        "p05": round(float(np.percentile(values, 5)), 5),
        "p95": round(float(np.percentile(values, 95)), 5),
        "p99": round(float(np.percentile(values, 99)), 5),
        "max": round(float(values.max()), 5),
    }


def nanmean_rows(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    totals = np.nansum(values, axis=1)
    return np.divide(totals, counts, out=np.full(len(values), np.nan), where=counts > 0)


def nanmax_rows(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    safe = np.where(valid, values, -np.inf)
    result = safe.max(axis=1)
    result[~valid.any(axis=1)] = np.nan
    return result


def extract_pose(video_path: Path, model_path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop_width = int(round(width * 0.64))
    normalized_rows: list[np.ndarray] = []
    world_rows: list[np.ndarray] = []
    last_timestamp_ms = -1
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        min_tracking_confidence=0.25,
        output_segmentation_masks=False,
    )
    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
            frame_index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp_ms = max(last_timestamp_ms + 1, int(round(frame_index / fps * 1000)))
                last_timestamp_ms = timestamp_ms
                crop = cv2.cvtColor(frame[:, :crop_width], cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop), timestamp_ms)
                norm = np.full((33, 5), np.nan, dtype=float)
                world = np.full((33, 5), np.nan, dtype=float)
                if result.pose_landmarks:
                    mapped = map_landmarks(result.pose_landmarks[0], crop_width, width)
                    norm = np.array([[p.x, p.y, p.z, p.visibility, p.presence] for p in mapped], dtype=float)
                    world_points = result.pose_world_landmarks[0]
                    world = np.array([
                        [p.x, p.y, p.z, q(p, "visibility"), q(p, "presence")] for p in world_points
                    ], dtype=float)
                normalized_rows.append(norm)
                world_rows.append(world)
                frame_index += 1
                if frame_index % 300 == 0:
                    print(f"  inference {frame_index}/{expected}", flush=True)
    finally:
        cap.release()
    info = {"fps": fps, "width": width, "height": height, "frames": len(normalized_rows)}
    return info, np.stack(normalized_rows), np.stack(world_rows)


def calculate_metrics(
    info: dict, norm: np.ndarray, world: np.ndarray, player_height_m: float
) -> tuple[dict, list[dict], list[dict]]:
    fps, width, height = info["fps"], info["width"], info["height"]
    frame_count = len(norm)
    detection = np.isfinite(norm[:, 0, 0])
    mean_visibility = nanmean_rows(norm[:, :, 3])
    upper_visibility = nanmean_rows(norm[:, UPPER, 3])
    upper_presence = nanmean_rows(norm[:, UPPER, 4])
    pose_confidence = np.sqrt(np.clip(upper_visibility * upper_presence, 0.0, 1.0))

    hip = np.column_stack([
        nanmean_rows(norm[:, [LEFT_HIP, RIGHT_HIP], axis]) for axis in (0, 1)
    ])
    shoulder = np.column_stack([
        nanmean_rows(norm[:, [LEFT_SHOULDER, RIGHT_SHOULDER], axis]) for axis in (0, 1)
    ])
    hip_smooth = interpolate_and_smooth(hip, 5)
    shoulder_smooth = interpolate_and_smooth(shoulder, 5)
    hip_height_norm = 1.0 - hip_smooth[:, 1]
    hip_vertical_speed_norm_s = np.abs(np.gradient(hip_height_norm, 1.0 / fps))
    torso_length_px = np.linalg.norm((shoulder_smooth - hip_smooth) * np.array([width, height]), axis=1)
    body_scale_px = float(np.nanmedian(torso_length_px[upper_visibility >= 0.50]))
    torso_physical_m = player_height_m * TORSO_HEIGHT_FRACTION
    torso_scale_px = interpolate_and_smooth(torso_length_px, 9)
    torso_scale_px = np.clip(torso_scale_px, 0.40 * body_scale_px, 3.00 * body_scale_px)
    meters_per_pixel = torso_physical_m / np.maximum(torso_scale_px, 1e-6)

    foot_quality = np.minimum(norm[:, FOOT_LANDMARKS, 3], norm[:, FOOT_LANDMARKS, 4])
    foot_y = np.where(foot_quality >= 0.35, norm[:, FOOT_LANDMARKS, 1], np.nan)
    ground_y_norm = nanmax_rows(foot_y)
    hip_height_calibrated_m = (ground_y_norm - hip_smooth[:, 1]) * height * meters_per_pixel
    hip_height_calibrated_m[
        (~np.isfinite(hip_height_calibrated_m))
        | (hip_height_calibrated_m < 0.25)
        | (hip_height_calibrated_m > 1.35)
    ] = np.nan

    wrist_xy = norm[:, RIGHT_WRIST, :2]
    wrist_px = wrist_xy * np.array([width, height])
    wrist_speed_px_s = vector_speed(wrist_xy, fps, np.array([width, height]))
    wrist_speed_body_s = wrist_speed_px_s / max(body_scale_px, 1e-6)
    wrist_speed_height_calibrated_m_s = wrist_speed_px_s * meters_per_pixel
    wrist_velocity_px_s = np.gradient(interpolate_and_smooth(wrist_px, 5), 1.0 / fps, axis=0)
    forward_motion = wrist_velocity_px_s[:, 0] > 0.08 * np.maximum(wrist_speed_px_s, 1e-6)
    wrist_world = world[:, RIGHT_WRIST, :3]
    wrist_speed_world_m_s = vector_speed(wrist_world, fps)
    world_shoulder = np.column_stack([
        nanmean_rows(world[:, [LEFT_SHOULDER, RIGHT_SHOULDER], axis]) for axis in (0, 1, 2)
    ])
    world_hip = np.column_stack([
        nanmean_rows(world[:, [LEFT_HIP, RIGHT_HIP], axis]) for axis in (0, 1, 2)
    ])
    world_torso_length = np.linalg.norm(world_shoulder - world_hip, axis=1)
    world_torso_median = float(np.nanmedian(world_torso_length[upper_visibility >= 0.50]))
    world_height_scale = torso_physical_m / max(world_torso_median, 1e-6)
    wrist_speed_world_height_scaled_m_s = wrist_speed_world_m_s * world_height_scale
    wrist_visibility = norm[:, RIGHT_WRIST, 3]
    wrist_presence = norm[:, RIGHT_WRIST, 4]
    wrist_reliable = detection & (wrist_visibility >= 0.50) & (wrist_presence >= 0.50)

    torso_vector = shoulder_smooth - hip_smooth
    torso_lean_deg = np.degrees(np.arctan2(torso_vector[:, 0], -torso_vector[:, 1]))
    stance_width_body = np.linalg.norm(
        (norm[:, LEFT_ANKLE, :2] - norm[:, RIGHT_ANKLE, :2]) * np.array([width, height]), axis=1
    ) / max(body_scale_px, 1e-6)

    right_elbow_angle = np.full(frame_count, np.nan)
    left_knee_angle = np.full(frame_count, np.nan)
    right_knee_angle = np.full(frame_count, np.nan)
    for i in range(frame_count):
        right_elbow_angle[i] = point_angle(norm[i, RIGHT_SHOULDER, :2], norm[i, RIGHT_ELBOW, :2], norm[i, RIGHT_WRIST, :2])
        left_knee_angle[i] = point_angle(norm[i, LEFT_HIP, :2], norm[i, LEFT_KNEE, :2], norm[i, LEFT_ANKLE, :2])
        right_knee_angle[i] = point_angle(norm[i, RIGHT_HIP, :2], norm[i, RIGHT_KNEE, :2], norm[i, RIGHT_ANKLE, :2])

    peaks = detect_peaks(wrist_speed_body_s, fps, wrist_reliable, forward_motion)
    phases, event_ids = semantic_phases(frame_count, peaks, fps, wrist_speed_body_s)
    hip_visibility = nanmean_rows(norm[:, [LEFT_HIP, RIGHT_HIP], 3])
    reliable_hip = detection & np.isfinite(hip_height_norm) & (hip_visibility >= 0.40)
    hip_values = hip_height_norm[reliable_hip]
    low_hip, high_hip = (
        (float(np.percentile(hip_values, 33)), float(np.percentile(hip_values, 67)))
        if len(hip_values) else (0.0, 1.0)
    )
    height_states = [
        "bajo_flexionado" if value <= low_hip else "alto_erguido" if value >= high_hip else "medio"
        for value in hip_height_norm
    ]

    rows: list[dict] = []
    for i in range(frame_count):
        rows.append({
            "frame": i,
            "timestamp_s": round(i / fps, 6),
            "detected": bool(detection[i]),
            "pose_confidence": pose_confidence[i],
            "mean_visibility": mean_visibility[i],
            "upper_body_visibility": upper_visibility[i],
            "hip_center_x_norm": hip_smooth[i, 0],
            "hip_center_height_norm": hip_height_norm[i],
            "hip_center_height_px_from_bottom": hip_height_norm[i] * height,
            "hip_center_height_calibrated_m": hip_height_calibrated_m[i],
            "meters_per_pixel_from_height": meters_per_pixel[i],
            "hip_vertical_speed_norm_s": hip_vertical_speed_norm_s[i],
            "torso_lean_deg": torso_lean_deg[i],
            "stance_width_torso_units": stance_width_body[i],
            "right_elbow_angle_deg": right_elbow_angle[i],
            "left_knee_angle_deg": left_knee_angle[i],
            "right_knee_angle_deg": right_knee_angle[i],
            "right_wrist_x_px": wrist_px[i, 0],
            "right_wrist_y_px": wrist_px[i, 1],
            "right_wrist_visibility": wrist_visibility[i],
            "right_wrist_presence": wrist_presence[i],
            "right_wrist_reliable": bool(wrist_reliable[i]),
            "racket_proxy_speed_px_s": wrist_speed_px_s[i],
            "racket_proxy_speed_torso_lengths_s": wrist_speed_body_s[i],
            "racket_proxy_speed_height_calibrated_m_s": wrist_speed_height_calibrated_m_s[i],
            "racket_proxy_speed_world_est_m_s": wrist_speed_world_m_s[i],
            "racket_proxy_speed_world_height_scaled_m_s": wrist_speed_world_height_scaled_m_s[i],
            "semantic_phase": phases[i],
            "semantic_event_id": event_ids[i] if event_ids[i] >= 0 else "",
            "hip_height_state": height_states[i],
        })

    events: list[dict] = []
    for event_id, peak in enumerate(peaks, start=1):
        vx = np.gradient(interpolate_and_smooth(wrist_xy, 5), 1.0 / fps, axis=0)[peak, 0]
        direction = "hacia_derecha_imagen" if vx > 0 else "hacia_izquierda_imagen"
        events.append({
            "event_id": event_id,
            "frame_peak": peak,
            "timestamp_peak_s": round(peak / fps, 5),
            "semantic_label": "pico_cinematico_hacia_oponente",
            "direction_image": direction,
            "peak_speed_px_s": round(float(wrist_speed_px_s[peak]), 5),
            "peak_speed_torso_lengths_s": round(float(wrist_speed_body_s[peak]), 5),
            "peak_speed_height_calibrated_m_s": round(float(wrist_speed_height_calibrated_m_s[peak]), 5),
            "peak_speed_world_est_m_s": round(float(wrist_speed_world_m_s[peak]), 5),
            "peak_speed_world_height_scaled_m_s": round(float(wrist_speed_world_height_scaled_m_s[peak]), 5),
            "wrist_visibility": round(float(wrist_visibility[peak]), 5),
            "hip_height_norm": round(float(hip_height_norm[peak]), 5),
            "hip_height_calibrated_m": (
                round(float(hip_height_calibrated_m[peak]), 5)
                if np.isfinite(hip_height_calibrated_m[peak]) else ""
            ),
            "hip_height_state": height_states[peak],
            "right_elbow_angle_deg": round(float(right_elbow_angle[peak]), 5),
            "torso_lean_deg": round(float(torso_lean_deg[peak]), 5),
        })

    reliable_speed = wrist_speed_body_s[wrist_reliable]
    reliable_px_speed = wrist_speed_px_s[wrist_reliable]
    reliable_world_speed = wrist_speed_world_m_s[wrist_reliable]
    reliable_height_speed = wrist_speed_height_calibrated_m_s[wrist_reliable]
    reliable_scaled_world_speed = wrist_speed_world_height_scaled_m_s[wrist_reliable]
    summary = {
        "player_height_m": player_height_m,
        "anthropometric_torso_fraction": TORSO_HEIGHT_FRACTION,
        "assumed_torso_length_m": round(torso_physical_m, 5),
        "frames": frame_count,
        "detection_rate": round(float(detection.mean()), 5),
        "right_wrist_reliable_rate": round(float(wrist_reliable.mean()), 5),
        "body_scale_median_torso_length_px": round(body_scale_px, 5),
        "height_calibration_median_m_per_px": round(float(np.nanmedian(meters_per_pixel)), 7),
        "world_height_scale_factor": round(world_height_scale, 5),
        "pose_confidence": safe_stats(pose_confidence[detection]),
        "hip_center_height_norm": safe_stats(hip_height_norm[reliable_hip]),
        "hip_center_height_calibrated_m": safe_stats(hip_height_calibrated_m[reliable_hip]),
        "hip_vertical_displacement_norm": round(float(np.ptp(hip_values)), 5) if len(hip_values) else None,
        "hip_vertical_speed_norm_s": safe_stats(hip_vertical_speed_norm_s[reliable_hip]),
        "racket_proxy_speed_px_s": safe_stats(reliable_px_speed),
        "racket_proxy_speed_torso_lengths_s": safe_stats(reliable_speed),
        "racket_proxy_speed_height_calibrated_m_s": safe_stats(reliable_height_speed),
        "racket_proxy_speed_world_est_m_s": safe_stats(reliable_world_speed),
        "racket_proxy_speed_world_height_scaled_m_s": safe_stats(reliable_scaled_world_speed),
        "torso_lean_deg": safe_stats(torso_lean_deg[detection]),
        "stance_width_torso_units": safe_stats(stance_width_body[detection]),
        "right_elbow_angle_deg": safe_stats(right_elbow_angle[detection]),
        "left_knee_angle_deg": safe_stats(left_knee_angle[detection]),
        "right_knee_angle_deg": safe_stats(right_knee_angle[detection]),
        "semantic_stroke_events": len(events),
    }
    return summary, rows, events


def draw_annotated_video(video_path: Path, output_path: Path, silent_path: Path, norm: np.ndarray, rows: list[dict]) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(silent_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    trail = deque(maxlen=max(8, int(round(fps * 0.55))))
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            values = norm[frame_index]
            row = rows[frame_index]
            points: dict[int, tuple[int, int]] = {}
            if row["detected"]:
                for idx in range(33):
                    if min(values[idx, 3], values[idx, 4]) >= 0.20 or idx in (RIGHT_ELBOW, RIGHT_WRIST):
                        points[idx] = (
                            int(np.clip(values[idx, 0], 0.0, 1.0) * (width - 1)),
                            int(np.clip(values[idx, 1], 0.0, 1.0) * (height - 1)),
                        )
                for a, b in POSE_CONNECTIONS:
                    if a in points and b in points:
                        racket_arm = (a, b) in ((RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST))
                        cv2.line(frame, points[a], points[b], (255, 45, 220) if racket_arm else (40, 230, 255), 5 if racket_arm else 3, cv2.LINE_AA)
                for idx, point in points.items():
                    if idx != RIGHT_WRIST:
                        cv2.circle(frame, point, 4, (40, 230, 255), -1, cv2.LINE_AA)
                hip = (
                    int(row["hip_center_x_norm"] * width),
                    int((1.0 - row["hip_center_height_norm"]) * height),
                )
                cv2.circle(frame, hip, 11, (255, 255, 40), -1, cv2.LINE_AA)
                cv2.putText(frame, "CENTRO CADERA", (hip[0] + 15, hip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 40), 2, cv2.LINE_AA)
                wrist = points.get(RIGHT_WRIST)
                if wrist:
                    trail.append(wrist)
                    wrist_color = (255, 45, 220) if row["right_wrist_reliable"] else (0, 165, 255)
                    for i in range(1, len(trail)):
                        cv2.line(frame, trail[i - 1], trail[i], wrist_color, 2, cv2.LINE_AA)
                    cv2.circle(frame, wrist, 18, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.circle(frame, wrist, 11, wrist_color, -1, cv2.LINE_AA)

            panel_width, panel_height = 1040, 150
            x0, y0 = (width - panel_width) // 2, height - panel_height - 18
            overlay = frame.copy()
            cv2.rectangle(overlay, (x0, y0), (x0 + panel_width, y0 + panel_height), (8, 8, 8), -1)
            cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
            lines = [
                f"JUGADOR PRINCIPAL IZQUIERDO | pose {row['pose_confidence']:.3f} | RW vis {row['right_wrist_visibility']:.3f}",
                f"Altura cadera: {row['hip_center_height_calibrated_m']:.3f} m ({row['hip_height_state']}) | torso {row['torso_lean_deg']:+.1f} deg",
                f"Vel. proxy raqueta: {row['racket_proxy_speed_height_calibrated_m_s']:.2f} m/s | {row['racket_proxy_speed_torso_lengths_s']:.2f} torsos/s",
                f"FASE SEMANTICA: {row['semantic_phase']} | evento {row['semantic_event_id'] or '-'}",
            ]
            colors = [(40, 230, 255), (255, 255, 40), (255, 80, 220), (255, 255, 255)]
            for line_index, (text, color) in enumerate(zip(lines, colors)):
                cv2.putText(frame, text, (x0 + 18, y0 + 29 + line_index * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.57, color, 2, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
            if frame_index % 600 == 0:
                print(f"  render {frame_index}/{len(rows)}", flush=True)
    finally:
        cap.release()
        writer.release()
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(silent_path), "-i", str(video_path),
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", str(output_path),
    ], check=True)
    silent_path.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(video_name: str, duration_s: float, summary: dict) -> dict:
    return {
        "video": video_name,
        "duration_s": round(duration_s, 5),
        "frames": summary["frames"],
        "body_scale_median_torso_length_px": summary["body_scale_median_torso_length_px"],
        "detection_rate": summary["detection_rate"],
        "right_wrist_reliable_rate": summary["right_wrist_reliable_rate"],
        "pose_confidence_mean": summary["pose_confidence"]["mean"],
        "hip_height_mean_norm": summary["hip_center_height_norm"]["mean"],
        "hip_height_min_norm": summary["hip_center_height_norm"]["min"],
        "hip_height_max_norm": summary["hip_center_height_norm"]["max"],
        "hip_height_mean_calibrated_m": summary["hip_center_height_calibrated_m"]["mean"],
        "hip_height_p05_calibrated_m": summary["hip_center_height_calibrated_m"]["p05"],
        "hip_height_p95_calibrated_m": summary["hip_center_height_calibrated_m"]["p95"],
        "hip_vertical_displacement_norm": summary["hip_vertical_displacement_norm"],
        "racket_speed_p95_px_s": summary["racket_proxy_speed_px_s"]["p95"],
        "racket_speed_max_px_s": summary["racket_proxy_speed_px_s"]["max"],
        "racket_speed_p95_torso_lengths_s": summary["racket_proxy_speed_torso_lengths_s"]["p95"],
        "racket_speed_max_torso_lengths_s": summary["racket_proxy_speed_torso_lengths_s"]["max"],
        "racket_speed_p95_height_calibrated_m_s": summary["racket_proxy_speed_height_calibrated_m_s"]["p95"],
        "racket_speed_max_height_calibrated_m_s": summary["racket_proxy_speed_height_calibrated_m_s"]["max"],
        "racket_speed_p95_world_est_m_s": summary["racket_proxy_speed_world_est_m_s"]["p95"],
        "racket_speed_max_world_est_m_s": summary["racket_proxy_speed_world_est_m_s"]["max"],
        "racket_speed_p95_world_height_scaled_m_s": summary["racket_proxy_speed_world_height_scaled_m_s"]["p95"],
        "torso_lean_median_deg": summary["torso_lean_deg"]["median"],
        "stance_width_median_torso_units": summary["stance_width_torso_units"]["median"],
        "right_elbow_angle_median_deg": summary["right_elbow_angle_deg"]["median"],
        "semantic_stroke_events": summary["semantic_stroke_events"],
    }


def process_batch(
    inputs: list[Path], output_root: Path, model_path: Path, player_height_m: float,
    pipeline: str = "left_player",
) -> None:
    ensure_output_layout(output_root)
    batch_rows: list[dict] = []
    all_events: list[dict] = []
    per_video: dict[str, dict] = {}
    for number, video_path in enumerate(inputs, start=1):
        print(f"[{number}/{len(inputs)}] {video_path.name}", flush=True)
        stem = video_path.stem
        info, norm, world = extract_pose(video_path, model_path)
        np.savez_compressed(
            artifact_path(output_root, "arrays", stem, pipeline, "pose_landmarks", ".npz"),
            normalized=norm,
            world=world,
            fps=info["fps"],
            width=info["width"],
            height=info["height"],
        )
        summary, rows, events = calculate_metrics(info, norm, world, player_height_m)
        duration_s = info["frames"] / info["fps"]
        summary.update({
            "video": video_path.name,
            "fps": round(info["fps"], 5),
            "duration_s": round(duration_s, 5),
        })
        write_csv(artifact_path(output_root, "metrics", stem, pipeline, "frame_metrics", ".csv"), rows)
        write_csv(artifact_path(output_root, "events", stem, pipeline, "semantic_events", ".csv"), events)
        artifact_path(output_root, "summaries", stem, pipeline, "motion_summary", ".json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for event in events:
            all_events.append({"video": video_path.name, **event})
        batch_rows.append(flatten_summary(video_path.name, duration_s, summary))
        per_video[video_path.name] = summary
        draw_annotated_video(
            video_path,
            artifact_path(output_root, "videos", stem, pipeline, "annotated", ".mp4"),
            artifact_path(output_root, "videos", stem, pipeline, "silent", ".mp4"),
            norm,
            rows,
        )
        print(
            f"  done: detection={summary['detection_rate']:.3f}, "
            f"wrist={summary['right_wrist_reliable_rate']:.3f}, events={len(events)}",
            flush=True,
        )
    write_csv(artifact_path(output_root, "summaries", "batch", pipeline, "motion_summary", ".csv"), batch_rows)
    write_csv(artifact_path(output_root, "events", "batch", pipeline, "semantic_events", ".csv"), all_events)
    report = {
        "scope": "left/main player only; right/secondary player intentionally excluded",
        "player_height_m": player_height_m,
        "videos": len(inputs),
        "methodology": {
            "target": "independent detector on left 64% of image",
            "height_calibration": "known 1.84 m stature and assumed shoulder-to-hip torso fraction of 0.288",
            "hip_height": "vertical hip-to-visible-foot distance converted using per-frame torso pixel scale",
            "racket_speed": "smoothed RIGHT_WRIST speed; pixels/s and median shoulder-to-hip torso-lengths/s",
            "world_speed": "MediaPipe monocular world estimate plus a separate height-scaled variant",
            "semantic_events": "adaptive wrist-speed peaks with heuristic temporal phases",
        },
        "interpretation_limits": [
            "The racket is not detected directly; RIGHT_WRIST is a grip/racket proxy.",
            "Maximum raw speed is noise-sensitive; prefer p95/p99 and shoulder-width-normalized values.",
            "Height-calibrated meters depend on the 0.288 anthropometric torso proportion and visible foot support.",
            "World m/s remains a monocular model estimate even after height scaling.",
            "Impact and movement phases are heuristic semantic labels, not ball-contact ground truth.",
            "Occluded knee/ankle angles can be inferred by the model and require visibility filtering.",
        ],
        "per_video": per_video,
    }
    artifact_path(output_root, "metadata", "batch", pipeline, "analysis", ".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--pipeline", default="left_player",
                        help="Filename namespace, e.g. left_player or height_calibration")
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_full.task"))
    parser.add_argument("--player-height-m", type=float, default=1.84)
    args = parser.parse_args()
    process_batch(args.inputs, args.output_root, args.model, args.player_height_m, args.pipeline)


if __name__ == "__main__":
    main()
