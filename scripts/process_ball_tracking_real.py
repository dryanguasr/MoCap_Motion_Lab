"""Detect and track a table-tennis ball in one or two short real clips.

The diagnostic video and frame CSV remain local. Compact summary.json and
REPORT.md are eligible for versioning, but this script never commits them.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seima_mocap.ball_detection import BallDetectorConfig, detect_ball_candidates, estimate_static_background
from seima_mocap.ball_tracking import (BallPhysicalParams, BallState, WorldBallObservation,
                                       fit_initial_state_world, predict_trajectory, propagate_state)
from seima_mocap.ball_video_tracking import PlanarSceneCalibration, ProjectedBallTracker, TrackStatus

DEFAULT_CLIPS = ("20251212_132025_1", "20251212_140101_1")
STATUS_COLORS = {TrackStatus.OBSERVED: (80, 230, 80), TrackStatus.PREDICTED: (0, 165, 255),
                 TrackStatus.UNINITIALIZED: (170, 170, 170), TrackStatus.LOST: (30, 30, 220)}


def video_timestamps(path: Path) -> tuple[np.ndarray, str]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "frame=best_effort_timestamp_time", "-of", "json", str(path)]
    try:
        payload = json.loads(subprocess.check_output(command, text=True))
        timestamps = np.array([float(frame["best_effort_timestamp_time"]) for frame in payload["frames"]])
        if len(timestamps) >= 3 and np.all(np.diff(timestamps) > 0):
            return timestamps - timestamps[0], "ffprobe_best_effort_timestamp_time"
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError):
        pass
    cap = cv2.VideoCapture(str(path))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or frames < 3:
        raise ValueError(f"No physical video timing available for {path}")
    return np.arange(frames) / fps, "FALLBACK_frame_index_over_container_fps"


def load_wrist(stem: str, timestamps: np.ndarray, width: int, height: int):
    directory = ROOT / "data/processed/left_player_batch" / stem
    cache = directory / "pose_landmarks.npz"
    metrics = directory / "frame_metrics.csv"
    result = {"xy": np.full((len(timestamps), 2), np.nan), "quality": np.zeros(len(timestamps)),
              "wrist_speed_px_s": np.full(len(timestamps), np.nan),
              "elbow_angle_deg": np.full(len(timestamps), np.nan),
              "elbow_angular_speed_deg_s": np.full(len(timestamps), np.nan), "source": "unavailable"}
    if cache.exists():
        with np.load(cache) as data:
            normalized = data["normalized"]
        n = min(len(timestamps), len(normalized))
        result["xy"][:n] = normalized[:n, 16, :2] * [width, height]
        result["quality"][:n] = np.minimum(normalized[:n, 16, 3], normalized[:n, 16, 4])
        result["source"] = str(cache.relative_to(ROOT))
    if metrics.exists():
        with metrics.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        n = min(len(rows), len(timestamps))
        for i in range(n):
            for source, target in (("racket_proxy_speed_px_s", "wrist_speed_px_s"),
                                   ("right_elbow_angle_deg", "elbow_angle_deg")):
                try:
                    result[target][i] = float(rows[i][source])
                except (KeyError, TypeError, ValueError):
                    pass
        valid = np.isfinite(result["elbow_angle_deg"])
        if valid.sum() >= 3:
            filled = np.interp(timestamps, timestamps[valid], result["elbow_angle_deg"][valid])
            angular = np.gradient(filled, timestamps)
            result["elbow_angular_speed_deg_s"][valid] = angular[valid]
    return result


def frame_record(index, timestamp, candidates, isolated, temporal, physical):
    record = {"frame": index, "timestamp_s": timestamp, "candidate_count": len(candidates),
              "isolated_detected": isolated is not None}
    if isolated is not None:
        record.update({"isolated_x_px": isolated.observation.pixel_xy[0],
                       "isolated_y_px": isolated.observation.pixel_xy[1],
                       "isolated_confidence": isolated.observation.confidence})
    for name, tracked in (("temporal", temporal), ("physical", physical)):
        record[f"{name}_status"] = tracked.status.value
        record[f"{name}_confidence"] = tracked.confidence
        record[f"{name}_missed_frames"] = tracked.missed_frames
        for kind, point in (("observed", tracked.observed_xy), ("predicted", tracked.predicted_xy)):
            record[f"{name}_{kind}_x_px"] = "" if point is None else point[0]
            record[f"{name}_{kind}_y_px"] = "" if point is None else point[1]
    return record


def draw_overlay(frame, candidates, tracked, trail, scene, events_now):
    cv2.polylines(frame, [np.round(scene.table_polygon_xy).astype(np.int32)], True, (210, 160, 40), 2)
    for candidate in candidates[:20]:
        x, y, w, h = candidate.bbox_xywh
        cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 120, 120), 1)
    if tracked.predicted_xy is not None:
        p = tuple(np.round(tracked.predicted_xy).astype(int))
        cv2.circle(frame, p, 14, STATUS_COLORS[TrackStatus.PREDICTED], 2, cv2.LINE_AA)
        cv2.line(frame, (p[0] - 10, p[1]), (p[0] + 10, p[1]), STATUS_COLORS[TrackStatus.PREDICTED], 1)
    if tracked.observed_xy is not None:
        p = tuple(np.round(tracked.observed_xy).astype(int))
        cv2.circle(frame, p, 8, STATUS_COLORS[TrackStatus.OBSERVED], -1, cv2.LINE_AA)
        cv2.circle(frame, p, 15, (255, 255, 255), 2, cv2.LINE_AA)
    for (a, sa), (b, sb) in zip(trail[:-1], trail[1:]):
        cv2.line(frame, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), STATUS_COLORS[sb], 2)
    if tracked.state is not None:
        future_times = [tracked.timestamp_s + i / 60 for i in range(1, 9)]
        for state in predict_trajectory(tracked.state, future_times, BallPhysicalParams.from_json(
                ROOT / "config/ball_tracking_defaults.json")):
            p = tuple(np.round(scene.project(state.position_m)).astype(int))
            cv2.circle(frame, p, 2, (0, 125, 255), -1)
    color = STATUS_COLORS[tracked.status]
    cv2.rectangle(frame, (18, 18), (650, 125), (20, 20, 20), -1)
    cv2.putText(frame, f"BALL TRACK: {tracked.status.value}", (35, 52), cv2.FONT_HERSHEY_SIMPLEX, .85, color, 2)
    cv2.putText(frame, f"conf {tracked.confidence:.2f} | candidatos {len(candidates)} | perdidos {tracked.missed_frames}",
                (35, 84), cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 235, 235), 1)
    cv2.putText(frame, "verde=observado  naranja=predicho  gris=candidato", (35, 111),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (210, 210, 210), 1)
    for j, event in enumerate(events_now):
        cv2.putText(frame, event["type"], (frame.shape[1] // 2 - 180, 70 + 34 * j),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (40, 240, 255), 2)


def detect_event(observed_history, wrist, frame_index, scene, existing):
    if len(observed_history) < 3:
        return []
    a, b, c = observed_history[-3:]
    dt1, dt2 = b[0] - a[0], c[0] - b[0]
    if dt1 <= 0 or dt2 <= 0 or max(dt1, dt2) > .16:
        return []
    v1, v2 = (b[1] - a[1]) / dt1, (c[1] - b[1]) / dt2
    delta = float(np.linalg.norm(v2 - v1))
    events = []
    if v1[1] > 120 and v2[1] < -80 and scene.close_to_table_surface(b[1], 35):
        confidence = float(np.clip((v1[1] - v2[1]) / 1200, .15, .9))
        events.append({"type": "possible_table_bounce", "timestamp_s": b[0], "frame": b[2],
                       "x_px": b[1][0], "y_px": b[1][1], "confidence": confidence})
    wrist_xy = wrist["xy"][b[2]] if b[2] < len(wrist["xy"]) else np.array([np.nan, np.nan])
    distance = float(np.linalg.norm(b[1] - wrist_xy)) if np.all(np.isfinite(wrist_xy)) else math.inf
    cosine = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-6))
    if distance <= 130 and (delta >= 350 or cosine < .45):
        confidence = float(np.clip(.45 * (1 - distance / 160) + .35 * min(delta / 1200, 1) +
                                   .2 * wrist["quality"][b[2]], .1, .95))
        events.append({"type": "possible_racket_contact", "timestamp_s": b[0], "frame": b[2],
                       "x_px": b[1][0], "y_px": b[1][1], "confidence": confidence,
                       "right_wrist_distance_px": distance, "trajectory_velocity_change_px_s": delta})
    return [event for event in events if all(event["type"] != old["type"] or
                                              abs(event["timestamp_s"] - old["timestamp_s"]) > .25
                                              for old in existing)]


def outgoing_records(events, records, wrist):
    output = []
    for event in events:
        if event["type"] != "possible_racket_contact":
            continue
        observations = []
        predicted = 0
        for row in records:
            dt = row["timestamp_s"] - event["timestamp_s"]
            if 0 < dt <= .25:
                if row["physical_status"] == "OBSERVED" and row["physical_observed_x_px"] != "":
                    observations.append((row["timestamp_s"], row["physical_observed_x_px"], row["physical_observed_y_px"]))
                elif row["physical_status"] == "PREDICTED":
                    predicted += 1
        frame = int(event["frame"])
        row = {**event, "observed_outgoing_frames": len(observations), "predicted_outgoing_frames": predicted,
               "ball_speed_px_s": None, "direction_image_deg": None,
               "speed_units": "px/s; no calibrated metric 3D", "effective_spin": "spin_not_identifiable",
               "right_wrist_speed_px_s": None, "right_elbow_angle_deg": None,
               "right_elbow_angular_speed_deg_s": None}
        if len(observations) >= 2:
            values = np.asarray(observations)
            relative_t = values[:, 0] - values[0, 0]
            vx, vy = (np.polyfit(relative_t, values[:, axis], 1)[0] for axis in (1, 2))
            row["ball_speed_px_s"] = float(np.hypot(vx, vy))
            row["direction_image_deg"] = float(np.degrees(np.arctan2(vy, vx)))
        for key in ("wrist_speed_px_s", "elbow_angle_deg", "elbow_angular_speed_deg_s"):
            value = wrist[key][frame] if frame < len(wrist[key]) else np.nan
            row["right_" + key if not key.startswith("wrist") else "right_" + key] = None if not np.isfinite(value) else float(value)
        output.append(row)
    return output


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_manual_reference(stem: str, records, events):
    path = ROOT / "data/annotations" / f"ball_tracking_{stem}.json"
    if not path.exists():
        return {"available": False}
    reference = json.loads(path.read_text(encoding="utf-8"))
    methods = {}
    for method in ("isolated", "temporal", "physical"):
        errors, observed_errors, predicted_errors, misses, false_positive_frames = [], [], [], [], []
        for point in reference["points"]:
            row = records[int(point["frame"])]
            if method == "isolated":
                x, y = row.get("isolated_x_px", ""), row.get("isolated_y_px", "")
                source = "observed"
            else:
                status = row[f"{method}_status"]
                kind = "observed" if status == "OBSERVED" else "predicted" if status == "PREDICTED" else None
                x = "" if kind is None else row[f"{method}_{kind}_x_px"]
                y = "" if kind is None else row[f"{method}_{kind}_y_px"]
                source = kind
            if x == "" or y == "" or source is None:
                misses.append(int(point["frame"]))
                continue
            error = float(np.hypot(float(x) - point["x_px"], float(y) - point["y_px"]))
            errors.append(error)
            (predicted_errors if source == "predicted" else observed_errors).append(error)
            if error > max(50, 2 * float(point["uncertainty_px"])):
                false_positive_frames.append(int(point["frame"]))
        methods[method] = {
            "annotated_points": len(reference["points"]), "available_points": len(errors),
            "coverage_pct": 100 * len(errors) / len(reference["points"]),
            "median_error_px": None if not errors else float(np.median(errors)),
            "p95_error_px": None if not errors else float(np.percentile(errors, 95)),
            "max_error_px": None if not errors else float(np.max(errors)),
            "rmse_px": None if not errors else float(np.sqrt(np.mean(np.square(errors)))),
            "within_25px": sum(error <= 25 for error in errors),
            "observed_points": len(observed_errors), "predicted_points": len(predicted_errors),
            "predicted_median_error_px": None if not predicted_errors else float(np.median(predicted_errors)),
            "missed_frames": misses, "observable_false_positive_frames": false_positive_frames,
        }
    event_results = []
    for expected in reference.get("events", []):
        matches = [event for event in events if event["type"] == expected["type"] and
                   abs(int(event["frame"]) - int(expected["frame"])) <= int(expected["tolerance_frames"])]
        event_results.append({**expected, "detected": bool(matches),
                              "detected_frame": None if not matches else int(min(matches, key=lambda e: abs(e["frame"] - expected["frame"]))["frame"])})
    return {"available": True, "reference": str(path.relative_to(ROOT)), "methods": methods,
            "events": event_results,
            "warning": "sparse approximate subset; detector centroids were visually accepted/corrected, so localization error is optimistic and not independent"}


def evaluate_flight_models(stem: str, scene: PlanarSceneCalibration, timestamps: np.ndarray):
    """Held-out projected fit comparison on explicitly reviewed flight segments.

    This is an identifiability diagnostic, not metric calibration. Magnus spin is
    a small fixed grid about the out-of-plane axis; it is never called measured.
    """
    path = ROOT / "data/annotations" / f"ball_tracking_{stem}.json"
    if not path.exists():
        return {"available": False, "conclusion": "spin_not_identifiable"}
    reference = json.loads(path.read_text(encoding="utf-8"))
    points = {int(item["frame"]): item for item in reference.get("points", [])}
    default = BallPhysicalParams.from_json(ROOT / "config/ball_tracking_defaults.json")
    models = {
        "gravity_only": (BallPhysicalParams(**{**default.__dict__, "drag_coefficient": 0.0,
                                                "magnus_enabled": False}), 0.0),
        "gravity_plus_drag": (BallPhysicalParams(**{**default.__dict__, "magnus_enabled": False}), 0.0),
        "magnus_minus_200_rad_s": (BallPhysicalParams(**{**default.__dict__, "magnus_enabled": True}), -200.0),
        "magnus_plus_200_rad_s": (BallPhysicalParams(**{**default.__dict__, "magnus_enabled": True}), 200.0),
    }
    segment_results = []
    for segment in reference.get("flight_segments", []):
        frames = [int(frame) for frame in segment["frames"] if int(frame) in points]
        if len(frames) < 4:
            continue
        train_frames, validation_frames = frames[:3], frames[3:]
        train_world = [scene.image_to_plane([points[frame]["x_px"], points[frame]["y_px"]]) for frame in train_frames]
        dt = timestamps[train_frames[-1]] - timestamps[train_frames[0]]
        velocity = (train_world[-1] - train_world[0]) / max(float(dt), 1e-6)
        errors = {}
        fitted_spins = {}
        for name, (params, spin_y) in models.items():
            initial = BallState(train_world[0], velocity, float(timestamps[train_frames[0]]),
                                np.array([0.0, spin_y, 0.0]))
            observations = [WorldBallObservation(float(timestamps[frame]), world)
                            for frame, world in zip(train_frames, train_world)]
            try:
                fit = fit_initial_state_world(observations, initial, params, max_iterations=15)
                pixel_errors = []
                for frame in validation_frames:
                    prediction = propagate_state(fit.state, float(timestamps[frame]), fit.params)
                    target = np.array([points[frame]["x_px"], points[frame]["y_px"]])
                    pixel_errors.append(float(np.linalg.norm(scene.project(prediction.position_m) - target)))
                errors[name] = float(np.sqrt(np.mean(np.square(pixel_errors))))
                fitted_spins[name] = spin_y
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                errors[name] = None
        finite = {name: error for name, error in errors.items() if error is not None and np.isfinite(error)}
        best = None if not finite else min(finite, key=finite.get)
        segment_results.append({"name": segment["name"], "training_frames": train_frames,
                                "held_out_frames": validation_frames, "held_out_rmse_px": errors,
                                "best_grid_model": best})
    winners = [item["best_grid_model"] for item in segment_results if item["best_grid_model"]]
    magnus_wins = [winner for winner in winners if winner.startswith("magnus")]
    return {"available": bool(segment_results), "segments": segment_results,
            "magnus_grid_wins": len(magnus_wins), "segments_evaluated": len(segment_results),
            "conclusion": "spin_not_identifiable",
            "reason": "approximate monocular plane, few held-out points, fixed coarse spin grid, and no stable independently calibrated improvement"}


def process(stem: str, output_root: Path, make_overlay: bool):
    video = ROOT / "data/raw" / f"{stem}.mp4"
    scenes = json.loads((ROOT / "config/ball_tracking_scenes.json").read_text(encoding="utf-8"))
    if stem not in scenes:
        raise KeyError(f"No explicit scene calibration for {stem}")
    scene_config = scenes[stem]
    detector_config = BallDetectorConfig.from_json(ROOT / "config/ball_detection_defaults.json")
    physical_params = BallPhysicalParams.from_json(ROOT / "config/ball_tracking_defaults.json")
    timestamps, timestamp_source = video_timestamps(video)
    background = estimate_static_background(video)
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 1 / np.median(np.diff(timestamps))
    if len(timestamps) != int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
        cap.release()
        raise ValueError("decoded frame count and timestamp count differ")
    scene = PlanarSceneCalibration(np.array(scene_config["table_polygon_xy"], float),
                                   pixels_per_m_vertical=scene_config["pixels_per_m_vertical"])
    physical = ProjectedBallTracker(scene, physical_params, use_physics=True)
    temporal = ProjectedBallTracker(scene, physical_params, use_physics=False)
    wrist = load_wrist(stem, timestamps, width, height)
    output = output_root / stem
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    if make_overlay:
        writer = cv2.VideoWriter(str(output / "ball_tracking_diagnostic.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("could not open diagnostic video writer")
    ok0, previous = cap.read()
    ok1, current = cap.read()
    if not (ok0 and ok1):
        raise ValueError("could not decode initial frames")
    records, observed_history, events, trail = [], [], [], []

    def consume(index, frame, candidates):
        nonlocal trail
        predicted = physical.predict_pixel(timestamps[index])
        isolated = candidates[0] if candidates else None
        temporal_result = temporal.step(timestamps[index], candidates)
        physical_result = physical.step(timestamps[index], candidates)
        # A free-flight model must not be propagated blindly through racket
        # impact. If the first missing observation occurs next to the main
        # player's right wrist, close the segment and seed a fresh trajectory.
        if physical_result.status == TrackStatus.PREDICTED and physical_result.missed_frames == 1 and observed_history:
            last_time, last_xy, last_frame = observed_history[-1]
            wrist_xy = wrist["xy"][last_frame]
            distance = float(np.linalg.norm(last_xy - wrist_xy)) if np.all(np.isfinite(wrist_xy)) else math.inf
            if distance <= 140 and wrist["quality"][last_frame] >= .3:
                contact = {"type": "possible_racket_contact", "timestamp_s": last_time, "frame": last_frame,
                           "x_px": last_xy[0], "y_px": last_xy[1],
                           "confidence": float(np.clip(.7 * (1 - distance / 180) + .3 * wrist["quality"][last_frame], .1, .9)),
                           "right_wrist_distance_px": distance,
                           "trajectory_velocity_change_px_s": None,
                           "reason": "track_break_near_right_wrist"}
                if all(contact["type"] != old["type"] or abs(contact["timestamp_s"] - old["timestamp_s"]) > .25
                       for old in events):
                    events.append(contact)
                physical.reset_for_discontinuity()
                physical_result = physical.step(timestamps[index], candidates)
                trail = []
        record = frame_record(index, float(timestamps[index]), candidates, isolated, temporal_result, physical_result)
        records.append(record)
        events_now = []
        point = physical_result.observed_xy if physical_result.observed_xy is not None else physical_result.predicted_xy
        if point is not None and physical_result.status in (TrackStatus.OBSERVED, TrackStatus.PREDICTED):
            trail.append((point.copy(), physical_result.status))
            trail[:] = trail[-35:]
        elif physical_result.status == TrackStatus.LOST:
            trail = []
        if physical_result.observed_xy is not None:
            observed_history.append((float(timestamps[index]), physical_result.observed_xy.copy(), index))
            observed_history[:] = observed_history[-8:]
            events_now = detect_event(observed_history, wrist, index, scene, events)
            events.extend(events_now)
        if writer is not None:
            annotated = frame.copy()
            draw_overlay(annotated, candidates, physical_result, trail, scene, events_now)
            writer.write(annotated)

    consume(0, previous, [])
    index = 1
    while True:
        ok, following = cap.read()
        if not ok:
            break
        predicted = physical.predict_pixel(float(timestamps[index]))
        candidates = detect_ball_candidates(previous, current, following, float(timestamps[index]), detector_config,
                                             roi_polygon_xy=scene_config["roi_polygon_xy"],
                                             background_gray=background, predicted_pixel_xy=predicted)
        consume(index, current, candidates)
        previous, current = current, following
        index += 1
    consume(index, current, [])
    cap.release()
    if writer is not None:
        writer.release()
    if len(records) != len(timestamps):
        raise AssertionError(f"processed {len(records)} records for {len(timestamps)} timestamps")

    strokes = outgoing_records(events, records, wrist)
    manual_validation = evaluate_manual_reference(stem, records, events)
    model_comparison = evaluate_flight_models(stem, scene, timestamps)
    write_csv(output / "frames.csv", records)
    write_csv(output / "events.csv", events)
    write_csv(output / "stroke_dataset.csv", strokes)
    counts = lambda prefix: {status.value: sum(row[prefix + "_status"] == status.value for row in records)
                             for status in TrackStatus}
    physical_counts, temporal_counts = counts("physical"), counts("temporal")
    isolated_count = sum(row["isolated_detected"] for row in records)
    summary = {
        "scope": "real-video exploratory projected tracking; not ground truth validation",
        "video": str(video.relative_to(ROOT)), "difficulty": scene_config["difficulty"],
        "frames": len(records), "duration_s": float(timestamps[-1]), "timestamp_source": timestamp_source,
        "timing_fallback_used": timestamp_source.startswith("FALLBACK"),
        "detector_isolated": {"frames_with_candidates": isolated_count,
                              "coverage_pct": 100 * isolated_count / len(records),
                              "warning": "coverage includes visual distractors and is not track accuracy"},
        "temporal_constant_velocity": {"state_counts": temporal_counts,
                                       "track_coverage_pct": 100 * (temporal_counts["OBSERVED"] + temporal_counts["PREDICTED"]) / len(records)},
        "physics_informed_projected": {"state_counts": physical_counts,
            "observed_frames": physical_counts["OBSERVED"], "predicted_frames": physical_counts["PREDICTED"],
            "track_coverage_pct": 100 * (physical_counts["OBSERVED"] + physical_counts["PREDICTED"]) / len(records),
            "observed_fraction_within_track_pct": 100 * physical_counts["OBSERVED"] /
                max(physical_counts["OBSERVED"] + physical_counts["PREDICTED"], 1),
            "loss_transitions": sum(records[i - 1]["physical_status"] != "LOST" and records[i]["physical_status"] == "LOST"
                                    for i in range(1, len(records)))},
        "events": {"possible_racket_contacts": sum(e["type"] == "possible_racket_contact" for e in events),
                   "possible_table_bounces": sum(e["type"] == "possible_table_bounce" for e in events)},
        "outgoing_strokes": strokes,
        "manual_validation": manual_validation,
        "flight_model_comparison": model_comparison,
        "player_kinematics_source": wrist["source"],
        "geometry": {"table_corners_px_manual": scene_config["table_polygon_xy"],
                     "table_length_m_prior": 2.74, "pixels_per_m_vertical_prior": scene_config["pixels_per_m_vertical"],
                     "status": scene_config["calibration_status"]},
        "identifiability": {"ball_speed_px_s": "estimated when >=2 post-contact observations",
                            "metric_ball_speed_m_s": "not identifiable from current monocular approximate calibration",
                            "spin": "spin_not_identifiable", "magnus_used": False},
        "limitations": ["candidate coverage is not precision", "manual approximate table polygon",
                        "single view does not recover depth", "30 fps undersamples fast ball flight",
                        "events are hypotheses, not verified contacts", "foreground includes moving players"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"clip": stem, "isolated": isolated_count, "temporal": temporal_counts,
                      "physical": physical_counts, "events": summary["events"]}, ensure_ascii=False), flush=True)
    return summary


def write_report(output_root: Path, summaries):
    lines = ["# Primera prueba real de seguimiento de bola", "",
             "Resultados exploratorios. Observación visual, predicción y pérdida se conservan como estados distintos. "
             "La calibración es planar y aproximada: no se reporta velocidad métrica ni spin medido.", "",
             "| Video | Dificultad | Candidatos aislados | Temporal | Físico proyectado | Observados / predichos | Contactos / botes |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        p = s["physics_informed_projected"]
        lines.append(f"| {Path(s['video']).name} | {s['difficulty']} | {s['detector_isolated']['coverage_pct']:.1f}% | "
                     f"{s['temporal_constant_velocity']['track_coverage_pct']:.1f}% | {p['track_coverage_pct']:.1f}% | "
                     f"{p['observed_frames']} / {p['predicted_frames']} | "
                     f"{s['events']['possible_racket_contacts']} / {s['events']['possible_table_bounces']} |")
    for s in summaries:
        validation = s.get("manual_validation", {})
        if not validation.get("available"):
            continue
        lines += ["", f"## Auditoría manual: {Path(s['video']).name}", "",
                  "| Método | Cobertura | Mediana / P95 / máximo / RMSE (px) | Falsos positivos observables | Frames perdidos |",
                  "|---|---:|---:|---:|---:|"]
        for name, label in (("isolated", "Detector aislado"), ("temporal", "Continuidad temporal"),
                            ("physical", "Tracker físico proyectado")):
            metric = validation["methods"][name]
            lines.append(f"| {label} | {metric['coverage_pct']:.1f}% | {metric['median_error_px']:.1f} / "
                         f"{metric['p95_error_px']:.1f} / {metric['max_error_px']:.1f} / {metric['rmse_px']:.1f} | "
                         f"{metric['observable_false_positive_frames']} | {metric['missed_frames']} |")
        annotated_points = validation["methods"]["physical"]["annotated_points"]
        physical_validation = validation["methods"]["physical"]
        audit_note = (f"Referencia escasa ({annotated_points} puntos) y no independiente para precisión subpíxel: "
                      "los centroides se aceptaron/corrigieron visualmente.")
        if physical_validation["predicted_points"]:
            audit_note += (f" El tracker físico tuvo {physical_validation['predicted_points']} predicción(es) "
                           f"anotada(s), con mediana de error {physical_validation['predicted_median_error_px']:.1f} px.")
        audit_note += " No extrapolar estas cifras al video completo."
        lines += ["", audit_note]
        comparison = s.get("flight_model_comparison", {})
        if comparison.get("available"):
            lines += ["", "### Comparación exploratoria de vuelo", "",
                      "| Segmento | Frames de ajuste | Frames retenidos | Gravedad | Gravedad + drag | "
                      "Magnus -200 | Magnus +200 | Mejor en la rejilla |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"]
            for segment in comparison["segments"]:
                error = segment["held_out_rmse_px"]
                lines.append(f"| {segment['name']} | {segment['training_frames']} | {segment['held_out_frames']} | "
                             f"{error['gravity_only']:.1f} | {error['gravity_plus_drag']:.1f} | "
                             f"{error['magnus_minus_200_rad_s']:.1f} | {error['magnus_plus_200_rad_s']:.1f} | "
                             f"{segment['best_grid_model']} |")
            lines += ["", f"Resultado: `{comparison['conclusion']}`. Que una variante Magnus gane esta rejilla "
                      "pequeña no identifica spin: la proyección es monocular aproximada, hay pocos puntos retenidos "
                      "y el parámetro no se estimó ni se validó de forma independiente."]
    lines += ["", "`frames.csv` y el MP4 diagnóstico permanecen locales. Los conteos no prueban exactitud: la auditoría "
              "manual es pequeña y debe ampliarse antes de ajustar umbrales o estudiar Magnus.", "",
              "Variables identificables por ahora: coordenadas 2D, estado observado/predicho, cobertura y velocidad aparente px/s "
              "si hay suficientes observaciones. No identificables: posición/velocidad 3D calibradas y spin; resultado de spin: "
              "`spin_not_identifiable`."]
    (output_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="*", default=DEFAULT_CLIPS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/ball_tracking_real")
    parser.add_argument("--no-overlay", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = [process(stem, args.output_root, not args.no_overlay) for stem in args.clips]
    write_report(args.output_root, summaries)


if __name__ == "__main__":
    main()
