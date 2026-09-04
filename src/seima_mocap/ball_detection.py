"""Visual candidate detector for small, blurred table-tennis balls.

This module deliberately detects *candidates*, not a ball identity.  Temporal
association lives in :mod:`seima_mocap.ball_video_tracking`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .ball_tracking import BallObservation


@dataclass(frozen=True)
class BallDetectorConfig:
    motion_threshold: float = 14.0
    foreground_threshold: float = 18.0
    min_area_px: float = 3.0
    max_area_px: float = 1800.0
    min_dimension_px: int = 2
    max_dimension_px: int = 90
    target_area_px: float = 60.0
    max_candidates: int = 100
    min_confidence: float = 0.12
    prediction_sigma_px: float = 80.0

    @classmethod
    def from_json(cls, path: str | Path) -> "BallDetectorConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class BallCandidate:
    """A visual observation plus the evidence used to score it."""

    observation: BallObservation
    bbox_xywh: tuple[int, int, int, int]
    area_px: float
    circularity: float
    elongation: float
    mean_intensity: float
    mean_saturation: float
    contrast: float
    motion_strength: float
    foreground_strength: float
    reason_scores: Mapping[str, float] = field(default_factory=dict)


def polygon_mask(shape: tuple[int, int], polygon_xy: Sequence[Sequence[float]] | None) -> np.ndarray:
    mask = np.full(shape, 255, dtype=np.uint8)
    if polygon_xy:
        mask.fill(0)
        polygon = np.round(np.asarray(polygon_xy, dtype=float)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


def estimate_static_background(video_path: str | Path, sample_count: int = 17) -> np.ndarray:
    """Return a median grayscale background from sparse decoded frames.

    This is useful only for a fixed camera.  It is evidence in the candidate
    score and never a hard claim that a foreground pixel is the ball.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 3:
        cap.release()
        raise ValueError("video must contain at least three frames")
    samples = []
    targets = set(np.linspace(0, total - 1, min(sample_count, total), dtype=int).tolist())
    # Sequential decode avoids backend/keyframe-dependent seeking differences.
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index in targets:
            samples.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        index += 1
    cap.release()
    if len(samples) < 3:
        raise ValueError("could not decode enough frames for a background")
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def _contour_features(contour: np.ndarray, frame: np.ndarray, motion: np.ndarray,
                      foreground: np.ndarray | None) -> tuple[dict, np.ndarray | None]:
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    perimeter = float(cv2.arcLength(contour, True))
    circularity = 0.0 if perimeter <= 0 else float(np.clip(4 * math.pi * area / perimeter**2, 0, 1))
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    minor, major = sorted((max(float(rw), 1.0), max(float(rh), 1.0)))
    elongation = major / minor
    angle = math.radians(rect[2] + (90 if rw < rh else 0))
    blur = None if elongation < 1.5 else np.array([math.cos(angle), math.sin(angle)]) * major
    local_mask = np.zeros((h, w), dtype=np.uint8)
    shifted = contour - np.array([[[x, y]]], dtype=contour.dtype)
    cv2.drawContours(local_mask, [shifted], -1, 255, -1)
    hsv = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    intensity = float(cv2.mean(hsv[..., 2], mask=local_mask)[0])
    saturation = float(cv2.mean(hsv[..., 1], mask=local_mask)[0])
    motion_strength = float(cv2.mean(motion[y:y + h, x:x + w], mask=local_mask)[0])
    foreground_strength = 0.0 if foreground is None else float(
        cv2.mean(foreground[y:y + h, x:x + w], mask=local_mask)[0]
    )
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    ring = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 2]
    contrast = float(max(0.0, intensity - np.median(ring)))
    return ({"area": area, "bbox": (x, y, w, h), "circularity": circularity,
             "elongation": elongation, "intensity": intensity, "saturation": saturation,
             "contrast": contrast, "motion": motion_strength, "foreground": foreground_strength}, blur)


def detect_ball_candidates(
    previous_bgr: np.ndarray,
    current_bgr: np.ndarray,
    next_bgr: np.ndarray,
    timestamp_s: float,
    config: BallDetectorConfig,
    *,
    roi_polygon_xy: Sequence[Sequence[float]] | None = None,
    background_gray: np.ndarray | None = None,
    predicted_pixel_xy: np.ndarray | None = None,
) -> list[BallCandidate]:
    """Generate zero or more weak visual candidates for the current frame."""
    if previous_bgr.shape != current_bgr.shape or next_bgr.shape != current_bgr.shape:
        raise ValueError("triplet frames must have the same shape")
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in (previous_bgr, current_bgr, next_bgr)]
    # Present-frame support must differ from both temporal neighbours.  This
    # rejects a large fraction of static high-contrast background structure.
    motion = cv2.min(cv2.absdiff(gray[1], gray[0]), cv2.absdiff(gray[1], gray[2]))
    foreground = None if background_gray is None else cv2.absdiff(gray[1], background_gray)
    mask = (motion >= config.motion_threshold).astype(np.uint8) * 255
    if foreground is not None:
        weak_motion = (motion >= 0.65 * config.motion_threshold)
        foreground_support = (foreground >= config.foreground_threshold)
        mask = ((mask > 0) | (weak_motion & foreground_support)).astype(np.uint8) * 255
    mask &= polygon_mask(mask.shape, roi_polygon_xy)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[BallCandidate] = []
    for contour in contours:
        f, blur = _contour_features(contour, current_bgr, motion, foreground)
        x, y, w, h = f["bbox"]
        if not config.min_area_px <= f["area"] <= config.max_area_px:
            continue
        if min(w, h) < config.min_dimension_px or max(w, h) > config.max_dimension_px:
            continue
        motion_score = float(np.clip((f["motion"] - config.motion_threshold) / 70.0, 0, 1))
        foreground_score = float(np.clip((f["foreground"] - config.foreground_threshold) / 100.0, 0, 1))
        contrast_score = float(np.clip(f["contrast"] / 100.0, 0, 1))
        bright_unsaturated = float(np.clip(f["intensity"] / 220.0, 0, 1) *
                                   np.clip(1 - f["saturation"] / 255.0, 0, 1))
        area_score = float(math.exp(-0.5 * (math.log(max(f["area"], 1) / config.target_area_px) / 1.25) ** 2))
        shape_score = float(max(f["circularity"], math.exp(-0.5 * ((f["elongation"] - 3.0) / 2.5) ** 2)))
        prediction_score = 0.5
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        if predicted_pixel_xy is not None:
            distance = float(np.linalg.norm(center - np.asarray(predicted_pixel_xy)))
            prediction_score = math.exp(-0.5 * (distance / config.prediction_sigma_px) ** 2)
        reasons = {"motion": motion_score, "foreground": foreground_score, "contrast": contrast_score,
                   "bright_unsaturated": bright_unsaturated, "size": area_score,
                   "shape_or_blur": shape_score, "prediction_proximity": prediction_score}
        confidence = float(np.clip(0.25 * motion_score + 0.10 * foreground_score +
                                   0.18 * contrast_score + 0.20 * bright_unsaturated +
                                   0.14 * area_score + 0.08 * shape_score +
                                   0.05 * prediction_score, 0, 1))
        if confidence < config.min_confidence:
            continue
        observation = BallObservation(float(timestamp_s), center, confidence,
                                      size_px=float(max(w, h)), blur_vector_px=blur,
                                      source="temporal_contrast_blob")
        candidates.append(BallCandidate(observation, (x, y, w, h), f["area"], f["circularity"],
                                        f["elongation"], f["intensity"], f["saturation"], f["contrast"],
                                        f["motion"], f["foreground"], reasons))
    candidates.sort(key=lambda item: (-item.observation.confidence,
                                      float(item.observation.pixel_xy[1]),
                                      float(item.observation.pixel_xy[0])))
    return candidates[:config.max_candidates]
