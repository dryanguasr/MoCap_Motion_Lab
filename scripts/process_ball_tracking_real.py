"""Detect and track a table-tennis ball in one or two short real clips.

The diagnostic videos and frame CSVs remain local. Compact summaries and the
report are eligible for versioning, but this script never commits them.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
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
from seima_mocap.interaction_tracking import (
    InteractionAwareBallTracker, InteractionContext, InteractionMode, TwoRacketTracker,
    interaction_metrics, smooth_event_gaps,
)
from seima_mocap.landmarks import POSE_CONNECTIONS
from seima_mocap.output_layout import artifact_path, ensure_output_layout
from seima_mocap.pose_context import load_body_frames
from seima_mocap.racketvision_adapter import RacketVisionFrame, load_cache
from seima_mocap.table_geometry import annotation_path as table_annotation_path, load_table_geometry

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
    output_root = ROOT / "data/processed"
    cache = artifact_path(output_root, "arrays", stem, "left_player", "pose_landmarks", ".npz")
    metrics = artifact_path(output_root, "metrics", stem, "left_player", "frame_metrics", ".csv")
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


def frame_record(index, timestamp, candidates, isolated, temporal, physical, multimodal=None, context=None):
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
    if multimodal is not None:
        tracked = multimodal.track
        record.update({
            "multimodal_status": tracked.status.value,
            "multimodal_confidence": tracked.confidence,
            "multimodal_missed_frames": tracked.missed_frames,
            "interaction_mode": multimodal.mode.value,
            "candidate_sources": json.dumps(multimodal.candidate_count_by_source, sort_keys=True),
            "alternative_hypotheses": json.dumps(multimodal.alternatives, ensure_ascii=False),
            "selected_candidate_source": "" if tracked.selected_candidate is None else tracked.selected_candidate.observation.source,
            "selected_visual_evidence": "" if tracked.selected_candidate is None else tracked.selected_candidate.reason_scores.get("visual_evidence", ""),
            "selected_dynamic_evidence": "" if tracked.selected_candidate is None else tracked.selected_candidate.reason_scores.get("dynamic_evidence", ""),
            "selected_body_suppression_factor": "" if tracked.selected_candidate is None else tracked.selected_candidate.reason_scores.get("body_suppression_factor", ""),
            "selected_interaction_evidence": "" if tracked.selected_candidate is None else tracked.selected_candidate.reason_scores.get("interaction_evidence", ""),
        })
        for kind, point in (("observed", tracked.observed_xy), ("predicted", tracked.predicted_xy)):
            record[f"multimodal_{kind}_x_px"] = "" if point is None else point[0]
            record[f"multimodal_{kind}_y_px"] = "" if point is None else point[1]
        metrics = {metric.player_id: metric for metric in multimodal.interaction_metrics}
        for player in ("left", "right"):
            metric = metrics.get(player)
            record[f"{player}_ball_racket_distance_px"] = "" if metric is None else metric.distance_px
            record[f"{player}_closing_speed_px_s"] = "" if metric is None else metric.closing_speed_px_s
            record[f"{player}_time_to_closest_s"] = "" if metric is None or metric.time_to_closest_s is None else metric.time_to_closest_s
            record[f"{player}_swept_racket_intersection"] = False if metric is None else metric.swept_intersection
    if context is not None:
        for player in ("left", "right"):
            body = context.bodies.get(player)
            racket = context.rackets.get(player)
            wrist = None if body is None else body.right_wrist_xy
            record[f"{player}_right_wrist_x_px"] = "" if wrist is None else wrist[0]
            record[f"{player}_right_wrist_y_px"] = "" if wrist is None else wrist[1]
            record[f"{player}_body_confidence"] = 0 if body is None else body.confidence
            record[f"{player}_racket_status"] = TrackStatus.LOST.value if racket is None else racket.status.value
            record[f"{player}_racket_confidence"] = 0 if racket is None else racket.confidence
            record[f"{player}_racket_source"] = "" if racket is None or racket.pose is None else racket.pose.source
            if racket is not None and racket.pose is not None:
                for point_index, point_name in enumerate(("top", "bottom", "handle", "left", "right")):
                    point = racket.pose.keypoints_xy[point_index]
                    record[f"{player}_racket_{point_name}_x_px"] = point[0]
                    record[f"{player}_racket_{point_name}_y_px"] = point[1]
    return record


def draw_overlay(frame, candidates, tracked, trail, scene, events_now, multimodal=None, context=None):
    table_points = np.round(scene.table_polygon_xy).astype(np.int32)
    table_layer = frame.copy()
    cv2.fillPoly(table_layer, [table_points], (210, 160, 40))
    cv2.addWeighted(table_layer, .12, frame, .88, 0, frame)
    cv2.polylines(frame, [table_points], True, (210, 160, 40), 3, cv2.LINE_AA)
    if scene.net_polygon_xy is not None:
        net_points = np.round(scene.net_polygon_xy).astype(np.int32)
        cv2.polylines(frame, [net_points], True, (255, 70, 210), 3, cv2.LINE_AA)
        cv2.line(frame, tuple(net_points[0]), tuple(net_points[1]), (255, 255, 255), 2, cv2.LINE_AA)
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
        future_origin = max(tracked.timestamp_s, tracked.state.timestamp_s)
        future_times = [future_origin + i / 60 for i in range(1, 9)]
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
    if context is not None:
        colors = {"left": (255, 210, 40), "right": (65, 235, 80)}
        for player in ("left", "right"):
            body, racket = context.bodies.get(player), context.rackets.get(player)
            if body is not None and body.landmarks_xy is not None:
                quality = body.landmark_confidence
                for start, end in POSE_CONNECTIONS:
                    a, b = body.landmarks_xy[start], body.landmarks_xy[end]
                    reliable = (np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                                and (quality is None or min(quality[start], quality[end]) >= .12))
                    if reliable:
                        cv2.line(frame, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                                 colors[player], 2, cv2.LINE_AA)
                for index, point in enumerate(body.landmarks_xy):
                    if (np.all(np.isfinite(point))
                            and (quality is None or quality[index] >= .12)):
                        cv2.circle(frame, tuple(np.round(point).astype(int)), 3,
                                   colors[player], -1, cv2.LINE_AA)
            if body is not None and body.right_wrist_xy is not None:
                wrist_point = tuple(np.round(body.right_wrist_xy).astype(int))
                cv2.circle(frame, wrist_point, 11, (255, 45, 220), 3, cv2.LINE_AA)
            if racket is not None and racket.pose is not None:
                points = np.round(racket.pose.keypoints_xy).astype(int)
                head = points[[0, 4, 1, 3]]
                cv2.polylines(frame, [head], True, colors[player], 3, cv2.LINE_AA)
                cv2.line(frame, tuple(points[1]), tuple(points[2]), colors[player], 4, cv2.LINE_AA)
                for point in points:
                    cv2.circle(frame, tuple(point), 5, colors[player], -1, cv2.LINE_AA)
                if body is not None and body.right_wrist_xy is not None:
                    cv2.line(frame, tuple(np.round(body.right_wrist_xy).astype(int)), tuple(points[2]),
                             (255, 45, 220), 2, cv2.LINE_AA)
        panel_width, gap = 430, 20
        start_x, top = (frame.shape[1] - 2 * panel_width - gap) // 2, frame.shape[0] - 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, top), (start_x + 2 * panel_width + gap, frame.shape[0] - 8),
                      (10, 10, 10), -1)
        cv2.addWeighted(overlay, .72, frame, .28, 0, frame)
        for offset, player in enumerate(("left", "right")):
            racket = context.rackets.get(player)
            status = "LOST" if racket is None else racket.status.value
            confidence = 0 if racket is None else racket.confidence
            cv2.putText(frame, f"{player.upper()} racket {status} conf {confidence:.2f}",
                        (start_x + offset * (panel_width + gap) + 12, top + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, colors[player], 2, cv2.LINE_AA)
        if multimodal is not None:
            cv2.putText(frame, f"INTERACCION: {multimodal.mode.value}", (start_x + 12, top + 61),
                        cv2.FONT_HERSHEY_SIMPLEX, .58, (40, 240, 255), 2, cv2.LINE_AA)
            for rank, alternative in enumerate(multimodal.alternatives[1:3], 1):
                cv2.circle(frame, (int(round(alternative[0])), int(round(alternative[1]))),
                           8 + 3 * rank, (180, 80, 255), 1, cv2.LINE_AA)


def detect_event(observed_history, context_history, frame_index, scene, existing):
    if len(observed_history) < 3:
        return []
    a, b, c = observed_history[-3:]
    dt1, dt2 = b[0] - a[0], c[0] - b[0]
    if dt1 <= 0 or dt2 <= 0 or max(dt1, dt2) > .16:
        return []
    v1, v2 = (b[1] - a[1]) / dt1, (c[1] - b[1]) / dt2
    delta = float(np.linalg.norm(v2 - v1))
    events = []
    table_x = scene.table_polygon_xy[:, 0]
    near_surface = (table_x.min() - 25 <= b[1][0] <= table_x.max() + 25
                    and scene.close_to_table_surface(b[1], margin_px=25))
    if v1[1] > 120 and v2[1] < -80 and near_surface:
        confidence = float(np.clip((v1[1] - v2[1]) / 1200, .15, .9))
        events.append({"type": "possible_table_bounce", "timestamp_s": b[0], "frame": b[2],
                       "x_px": b[1][0], "y_px": b[1][1], "confidence": confidence})
    cosine = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-6))
    context = context_history.get(b[2])
    if context is not None and (delta >= 350 or cosine < .45):
        for metric in interaction_metrics(b[1], v1, context):
            racket = context.rackets.get(metric.player_id)
            if racket is None or racket.pose is None:
                continue
            model_ok = racket.status == TrackStatus.OBSERVED and racket.confidence >= .30 and metric.distance_px <= 145
            proxy_ok = (metric.distance_px <= 100 and metric.closing_speed_px_s >= 300
                        and metric.time_to_closest_s is not None and metric.time_to_closest_s <= .12)
            if not (model_ok or proxy_ok):
                continue
            confidence = float(np.clip(.45 * (1 - metric.distance_px / 170)
                                       + .35 * min(delta / 1200, 1) + .2 * racket.confidence, .1, .95))
            events.append({"type": "possible_racket_contact", "timestamp_s": b[0], "frame": b[2],
                           "x_px": b[1][0], "y_px": b[1][1], "confidence": confidence,
                           "actor": metric.player_id, "racket_distance_px": metric.distance_px,
                           "closing_speed_px_s": metric.closing_speed_px_s,
                           "trajectory_velocity_change_px_s": delta,
                           "reason": "direction_change_near_associated_racket"})
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
    expected_events = reference.get("events", [])
    event_frames = [int(event["frame"]) for event in expected_events]

    def canonical_visibility(label):
        label = str(label).lower()
        if "out_of_frame" in label:
            return "out_of_frame"
        if "occlud" in label:
            return "occluded"
        if "blur" in label:
            return "blurred"
        return "visible"
    for method in ("isolated", "temporal", "physical", "multimodal"):
        errors, observed_errors, predicted_errors, misses, false_positive_frames = [], [], [], [], []
        by_visibility = {name: {"points": 0, "true_positive": 0, "available": 0}
                         for name in ("visible", "blurred", "occluded", "out_of_frame")}
        outside_events = {"points": 0, "true_positive": 0, "available": 0}
        for point in reference["points"]:
            visibility = canonical_visibility(point.get("visibility", "visible"))
            by_visibility[visibility]["points"] += 1
            outside = all(abs(int(point["frame"]) - event_frame) > 3 for event_frame in event_frames)
            if outside:
                outside_events["points"] += 1
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
            by_visibility[visibility]["available"] += 1
            if outside:
                outside_events["available"] += 1
            error = float(np.hypot(float(x) - point["x_px"], float(y) - point["y_px"]))
            errors.append(error)
            (predicted_errors if source == "predicted" else observed_errors).append(error)
            if error > max(50, 2 * float(point["uncertainty_px"])):
                false_positive_frames.append(int(point["frame"]))
            else:
                by_visibility[visibility]["true_positive"] += 1
                if outside:
                    outside_events["true_positive"] += 1
        true_positive = len(errors) - len(false_positive_frames)

        def classification(values):
            precision = values["true_positive"] / max(values["available"], 1)
            recall = values["true_positive"] / max(values["points"], 1)
            return {**values, "precision": precision, "recall": recall,
                    "f1": 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)}
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
            "precision_in_annotated_points": true_positive / max(len(errors), 1),
            "recall_in_annotated_points": true_positive / len(reference["points"]),
            "f1_in_annotated_points": 2 * true_positive / max(len(errors) + len(reference["points"]), 1),
            "by_visibility": {name: classification(values) for name, values in by_visibility.items()},
            "outside_event_windows": classification(outside_events),
        }
    event_results = []
    recovered = 0
    for expected in expected_events:
        matches = [event for event in events if event["type"] == expected["type"] and
                   abs(int(event["frame"]) - int(expected["frame"])) <= int(expected["tolerance_frames"])]
        frame = int(expected["frame"])
        recovery = next((offset for offset in range(0, 3) if frame + offset < len(records)
                         and records[frame + offset]["multimodal_status"] in ("OBSERVED", "PREDICTED")), None)
        recovered += recovery is not None
        event_results.append({**expected, "detected": bool(matches),
                              "detected_frame": None if not matches else int(min(matches, key=lambda e: abs(e["frame"] - expected["frame"]))["frame"]),
                              "multimodal_recovered_within_2_frames": recovery is not None,
                              "recovery_offset_frames": recovery})
    return {"available": True, "reference": str(path.relative_to(ROOT)), "methods": methods,
            "events": event_results,
            "event_recovery_within_2_frames_pct": None if not expected_events else 100 * recovered / len(expected_events),
            "event_precision": "not estimable from sparse non-exhaustive event annotations",
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
    geometry_path = table_annotation_path(ROOT / "data/annotations/table_geometry", stem)
    geometry_source = "config/ball_tracking_scenes.json (approximate fallback)"
    net_polygon = None
    if geometry_path.exists():
        geometry = load_table_geometry(geometry_path, video_path=video)
        scene_config = {**scene_config, "table_polygon_xy": geometry["table_polygon_xy"].tolist(),
                        "calibration_status": "manual fixed-camera surface and net annotation"}
        net_polygon = geometry["net_polygon_xy"]
        geometry_source = str(geometry_path.relative_to(ROOT))
    detector_config = BallDetectorConfig.from_json(ROOT / "config/ball_detection_defaults.json")
    physical_params = BallPhysicalParams.from_json(ROOT / "config/ball_tracking_defaults.json")
    timestamps, timestamp_source = video_timestamps(video)
    background = estimate_static_background(video)
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ensure_output_layout(output_root)
    two_player_cache = artifact_path(output_root, "arrays", stem, "two_players", "pose_landmarks", ".npz")
    left_player_cache = artifact_path(output_root, "arrays", stem, "left_player", "pose_landmarks", ".npz")
    racketvision_cache = artifact_path(output_root, "arrays", stem, "racketvision", "cache", ".jsonl")
    # Container metadata can advertise one trailing packet that OpenCV cannot
    # decode. Align all modalities to their common, actually processed prefix.
    available_frames = len(timestamps)
    if two_player_cache.exists():
        with np.load(two_player_cache) as pose_cache:
            available_frames = min(available_frames, len(pose_cache["left_normalized"]),
                                   len(pose_cache["right_normalized"]))
    racketvision_frames = None
    if racketvision_cache.exists():
        racketvision_frames = load_cache(racketvision_cache)
        available_frames = min(available_frames, len(racketvision_frames))
    if available_frames < 3:
        cap.release()
        raise ValueError("fewer than three common frames across video modalities")
    if available_frames != len(timestamps):
        print(f"Aligning modalities to {available_frames} decodable frames (container reports {len(timestamps)})",
              flush=True)
        timestamps = timestamps[:available_frames]
    fps = 1 / np.median(np.diff(timestamps))
    scene = PlanarSceneCalibration(np.array(scene_config["table_polygon_xy"], float),
                                   pixels_per_m_vertical=scene_config["pixels_per_m_vertical"],
                                   net_polygon_xy=net_polygon)
    physical = ProjectedBallTracker(scene, physical_params, use_physics=True)
    temporal = ProjectedBallTracker(scene, physical_params, use_physics=False)
    variant_specs = {
        "balltrack_model": (False, False),
        "body_context": (False, False),
        "bounce_hypothesis": (True, False),
        "racket_priors": (False, True),
        "full_multimodal": (True, True),
    }
    variant_trackers = {
        name: InteractionAwareBallTracker(
            ProjectedBallTracker(scene, physical_params, use_physics=True), scene,
            enable_bounce=flags[0], enable_contact=flags[1],
        ) for name, flags in variant_specs.items()
    }
    wrist = load_wrist(stem, timestamps, width, height)
    body_frames, body_source = load_body_frames(two_player_cache, left_player_cache,
                                                 len(timestamps), width, height)
    if racketvision_frames is not None:
        racketvision_frames = racketvision_frames[:len(timestamps)]
        racketvision_source = str(racketvision_cache.relative_to(ROOT))
    else:
        racketvision_frames = [RacketVisionFrame(index, float(timestamp), (), ())
                               for index, timestamp in enumerate(timestamps)]
        racketvision_source = "unavailable; classical ball and wrist-forearm racket fallback active"
    racket_tracker = TwoRacketTracker()
    video_output = artifact_path(output_root, "videos", stem, "ball_tracking", "diagnostic", ".mp4")
    ok0, previous = cap.read()
    ok1, current = cap.read()
    if not (ok0 and ok1):
        raise ValueError("could not decode initial frames")
    records, observed_history, baseline_observed_history, events, trail = [], [], [], [], []
    overlay_payloads = []
    context_history = {}
    variant_frames = {name: [] for name in variant_trackers}
    variant_modes = {name: [] for name in variant_trackers}

    def consume(index, frame, candidates):
        nonlocal trail
        timestamp = float(timestamps[index])
        predicted = physical.predict_pixel(timestamp)
        isolated = candidates[0] if candidates else None
        temporal_result = temporal.step(timestamp, candidates)
        physical_result = physical.step(timestamp, candidates)
        # A free-flight model must not be propagated blindly through racket
        # impact. If the first missing observation occurs next to the main
        # player's right wrist, close the segment and seed a fresh trajectory.
        if physical_result.status == TrackStatus.PREDICTED and physical_result.missed_frames == 1 and baseline_observed_history:
            last_time, last_xy, last_frame = baseline_observed_history[-1]
            wrist_xy = wrist["xy"][last_frame]
            distance = float(np.linalg.norm(last_xy - wrist_xy)) if np.all(np.isfinite(wrist_xy)) else math.inf
            if distance <= 140 and wrist["quality"][last_frame] >= .3:
                contact = {"type": "possible_racket_contact", "timestamp_s": last_time, "frame": last_frame,
                           "x_px": last_xy[0], "y_px": last_xy[1],
                           "confidence": float(np.clip(.7 * (1 - distance / 180) + .3 * wrist["quality"][last_frame], .1, .9)),
                           "right_wrist_distance_px": distance,
                           "trajectory_velocity_change_px_s": None,
                           "reason": "track_break_near_right_wrist"}
                physical.reset_for_discontinuity()
                physical_result = physical.step(timestamp, candidates)
                trail = []
        if physical_result.observed_xy is not None:
            baseline_observed_history.append((timestamp, physical_result.observed_xy.copy(), index))
            baseline_observed_history[:] = baseline_observed_history[-8:]
        model_frame = racketvision_frames[index]
        rackets = racket_tracker.step(timestamp, model_frame.rackets, body_frames[index], width)
        empty_context = InteractionContext(timestamp)
        body_context = InteractionContext(timestamp, body_frames[index], {})
        full_context = InteractionContext(timestamp, body_frames[index], rackets)
        context_history[index] = full_context
        contexts = {
            "balltrack_model": empty_context,
            "body_context": body_context,
            "bounce_hypothesis": body_context,
            "racket_priors": full_context,
            "full_multimodal": full_context,
        }
        multimodal_results = {}
        for name, tracker in variant_trackers.items():
            multimodal_results[name] = tracker.step(timestamp, candidates,
                                                     model_frame.ball_candidates, contexts[name],
                                                     anchor=physical_result)
            variant_frames[name].append(multimodal_results[name].track)
            variant_modes[name].append(multimodal_results[name].mode)
        multimodal = multimodal_results["full_multimodal"]
        overlay_payloads.append((tuple(candidates), multimodal, full_context))
        record = frame_record(index, timestamp, candidates, isolated, temporal_result,
                              physical_result, multimodal, full_context)
        for name, result in multimodal_results.items():
            record[f"ablation_{name}_status"] = result.track.status.value
        records.append(record)
        events_now = []
        full_track = multimodal.track
        point = full_track.observed_xy if full_track.observed_xy is not None else full_track.predicted_xy
        if point is not None and full_track.status in (TrackStatus.OBSERVED, TrackStatus.PREDICTED):
            trail.append((point.copy(), full_track.status))
            trail[:] = trail[-35:]
        elif full_track.status == TrackStatus.LOST:
            trail = []
        if full_track.observed_xy is not None:
            observed_history.append((timestamp, full_track.observed_xy.copy(), index))
            observed_history[:] = observed_history[-8:]
            events_now = detect_event(observed_history, context_history, index, scene, events)
            events.extend(events_now)

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
    if len(records) != len(timestamps):
        raise AssertionError(f"processed {len(records)} records for {len(timestamps)} timestamps")

    smoothing = {}
    for name in variant_trackers:
        smoothed, corrections = smooth_event_gaps(variant_frames[name], variant_modes[name], scene)
        variant_frames[name] = smoothed
        smoothing[name] = corrections
        if name == "full_multimodal":
            correction_by_frame = {item["frame"]: item for item in corrections}
            for frame_index, tracked in enumerate(smoothed):
                records[frame_index]["multimodal_status"] = tracked.status.value
                records[frame_index]["multimodal_confidence"] = tracked.confidence
                for kind, point in (("observed", tracked.observed_xy), ("predicted", tracked.predicted_xy)):
                    records[frame_index][f"multimodal_{kind}_x_px"] = "" if point is None else point[0]
                    records[frame_index][f"multimodal_{kind}_y_px"] = "" if point is None else point[1]
                correction = correction_by_frame.get(frame_index)
                records[frame_index]["offline_smoothed"] = correction is not None
                records[frame_index]["smoothing_method"] = "" if correction is None else correction["method"]

    # Render only after the retrospective pass so the diagnostic video and the
    # exported per-frame metrics show the same corrected nine-frame trajectory.
    if make_overlay:
        overlay_cap = cv2.VideoCapture(str(video))
        writer = cv2.VideoWriter(str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (width, height))
        if not overlay_cap.isOpened() or not writer.isOpened():
            overlay_cap.release()
            writer.release()
            raise RuntimeError("could not open retrospective diagnostic video pass")
        overlay_trail = []
        events_by_frame = {}
        for event in events:
            events_by_frame.setdefault(int(event["frame"]), []).append(event)
        for frame_index, tracked in enumerate(variant_frames["full_multimodal"]):
            ok, frame = overlay_cap.read()
            if not ok:
                break
            point = tracked.observed_xy if tracked.observed_xy is not None else tracked.predicted_xy
            if point is not None and tracked.status in (TrackStatus.OBSERVED, TrackStatus.PREDICTED):
                overlay_trail.append((point.copy(), tracked.status))
                overlay_trail[:] = overlay_trail[-35:]
            elif tracked.status == TrackStatus.LOST:
                overlay_trail = []
            candidates, causal_multimodal, context = overlay_payloads[frame_index]
            displayed_multimodal = replace(causal_multimodal, track=tracked)
            annotated = frame.copy()
            draw_overlay(annotated, candidates, tracked, overlay_trail, scene,
                         events_by_frame.get(frame_index, []), displayed_multimodal, context)
            writer.write(annotated)
        overlay_cap.release()
        writer.release()
        if frame_index + 1 != len(timestamps):
            raise AssertionError(f"rendered {frame_index + 1} of {len(timestamps)} overlay frames")

    strokes = outgoing_records(events, records, wrist)
    manual_validation = evaluate_manual_reference(stem, records, events)
    model_comparison = evaluate_flight_models(stem, scene, timestamps)
    write_csv(artifact_path(output_root, "metrics", stem, "ball_tracking", "frames", ".csv"), records)
    write_csv(artifact_path(output_root, "events", stem, "ball_tracking", "events", ".csv"), events)
    write_csv(artifact_path(output_root, "datasets", stem, "ball_tracking", "strokes", ".csv"), strokes)
    counts = lambda prefix: {status.value: sum(row[prefix + "_status"] == status.value for row in records)
                             for status in TrackStatus}
    physical_counts, temporal_counts, multimodal_counts = counts("physical"), counts("temporal"), counts("multimodal")
    ablation = {}
    for name, frames in variant_frames.items():
        state_counts = {status.value: sum(frame.status == status for frame in frames) for status in TrackStatus}
        ablation[name] = {
            "state_counts": state_counts,
            "track_coverage_pct": 100 * (state_counts["OBSERVED"] + state_counts["PREDICTED"]) / len(frames),
            "lost_frames": state_counts["LOST"],
            "smoothing_corrections": len(smoothing[name]),
        }
    isolated_count = sum(row["isolated_detected"] for row in records)
    lost_reduction_pct = 100 * (physical_counts["LOST"] - multimodal_counts["LOST"]) / max(physical_counts["LOST"], 1)
    baseline_outside = manual_validation.get("methods", {}).get("physical", {}).get("outside_event_windows", {})
    multimodal_outside = manual_validation.get("methods", {}).get("multimodal", {}).get("outside_event_windows", {})
    precision_delta_pp = 100 * (multimodal_outside.get("precision", 0) - baseline_outside.get("precision", 0))
    event_recovery_pct = manual_validation.get("event_recovery_within_2_frames_pct")
    summary = {
        "scope": "real-video exploratory projected tracking; not ground truth validation",
        "video": str(video.relative_to(ROOT)), "difficulty": scene_config["difficulty"],
        "frames": len(records), "duration_s": float(timestamps[-1]), "timestamp_source": timestamp_source,
        "timing_fallback_used": timestamp_source.startswith("FALLBACK"),
        "detector_isolated": {"frames_with_candidates": isolated_count,
                              "coverage_pct": 100 * isolated_count / len(records),
                              "warning": "coverage includes visual distractors and is not track accuracy"},
        "racketvision_observations": {
            "balltrack_frames_with_candidates": sum(bool(frame.ball_candidates) for frame in racketvision_frames),
            "raw_frames_with_racket_detections": sum(bool(frame.rackets) for frame in racketvision_frames),
            "raw_racket_instances": sum(len(frame.rackets) for frame in racketvision_frames),
        },
        "racket_tracks": {
            player: {status.value: sum(row[f"{player}_racket_status"] == status.value for row in records)
                     for status in TrackStatus}
            for player in ("left", "right")
        },
        "temporal_constant_velocity": {"state_counts": temporal_counts,
                                       "track_coverage_pct": 100 * (temporal_counts["OBSERVED"] + temporal_counts["PREDICTED"]) / len(records)},
        "physics_informed_projected": {"state_counts": physical_counts,
            "observed_frames": physical_counts["OBSERVED"], "predicted_frames": physical_counts["PREDICTED"],
            "track_coverage_pct": 100 * (physical_counts["OBSERVED"] + physical_counts["PREDICTED"]) / len(records),
            "observed_fraction_within_track_pct": 100 * physical_counts["OBSERVED"] /
                max(physical_counts["OBSERVED"] + physical_counts["PREDICTED"], 1),
            "loss_transitions": sum(records[i - 1]["physical_status"] != "LOST" and records[i]["physical_status"] == "LOST"
                                    for i in range(1, len(records)))},
        "multimodal": {
            "state_counts": multimodal_counts,
            "track_coverage_pct": 100 * (multimodal_counts["OBSERVED"] + multimodal_counts["PREDICTED"]) / len(records),
            "interaction_mode_counts": {mode.value: sum(row["interaction_mode"] == mode.value for row in records)
                                        for mode in InteractionMode},
            "offline_smoothing_corrections": len(smoothing["full_multimodal"]),
            "racketvision_source": racketvision_source,
            "body_pose_source": body_source,
        },
        "ablation": {"baseline_classical_physics": {
            "state_counts": physical_counts,
            "track_coverage_pct": 100 * (physical_counts["OBSERVED"] + physical_counts["PREDICTED"]) / len(records),
            "lost_frames": physical_counts["LOST"],
        }, **ablation},
        "acceptance": {
            "lost_frame_reduction_pct": lost_reduction_pct,
            "lost_frame_reduction_target_pct": 30,
            "lost_frame_reduction_pass": lost_reduction_pct >= 30,
            "annotated_event_recovery_within_2_frames_pct": event_recovery_pct,
            "event_recovery_target_pct": 80,
            "event_recovery_pass": None if event_recovery_pct is None else event_recovery_pct >= 80,
            "outside_event_precision_delta_percentage_points": precision_delta_pp,
            "outside_event_precision_pass": precision_delta_pp >= -2,
            "common_visible_localization": "unchanged by construction: valid classical observations anchor the multimodal tracker",
            "racket_player_association_accuracy": "not evaluable until independent five-keypoint racket annotations exist",
            "racket_finetuning_decision": "deferred; pretrained recall cannot be measured from current annotations",
        },
        "events": {"possible_racket_contacts": sum(e["type"] == "possible_racket_contact" for e in events),
                   "possible_table_bounces": sum(e["type"] == "possible_table_bounce" for e in events)},
        "outgoing_strokes": strokes,
        "manual_validation": manual_validation,
        "flight_model_comparison": model_comparison,
        "player_kinematics_source": body_source,
        "geometry": {"table_corners_px": scene_config["table_polygon_xy"],
                     "net_polygon_px": None if net_polygon is None else net_polygon.tolist(),
                     "source": geometry_source,
                     "table_length_m_prior": 2.74, "pixels_per_m_vertical_prior": scene_config["pixels_per_m_vertical"],
                     "status": scene_config["calibration_status"]},
        "identifiability": {"ball_speed_px_s": "estimated when >=2 post-contact observations",
                            "metric_ball_speed_m_s": "not identifiable from current monocular approximate calibration",
                            "spin": "spin_not_identifiable", "magnus_used": False},
        "limitations": ["candidate coverage is not precision",
                        "table geometry remains approximate until its annotation JSON is manually confirmed",
                        "single view does not recover depth", "30 fps undersamples fast ball flight",
                        "events are hypotheses, not verified contacts", "foreground includes moving players",
                        "racket wrist-forearm proxies are explicitly lower confidence than RacketVision detections"],
    }
    artifact_path(output_root, "summaries", stem, "ball_tracking", "summary", ".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"clip": stem, "isolated": isolated_count, "temporal": temporal_counts,
                      "physical": physical_counts, "multimodal": multimodal_counts,
                      "events": summary["events"]}, ensure_ascii=False), flush=True)
    return summary


def write_report(output_root: Path, summaries):
    lines = ["# Primera prueba real de seguimiento de bola", "",
             "Resultados exploratorios. Observación visual, predicción y pérdida se conservan como estados distintos. "
             "La calibración es planar y aproximada: no se reporta velocidad métrica ni spin medido.", "",
             "| Video | Dificultad | Físico base | Multimodal | Perdidos base→fusión | Contactos / botes |",
             "|---|---|---:|---:|---:|---:|"]
    for s in summaries:
        p = s["physics_informed_projected"]
        m = s["multimodal"]
        lines.append(f"| {Path(s['video']).name} | {s['difficulty']} | {p['track_coverage_pct']:.1f}% | "
                     f"{m['track_coverage_pct']:.1f}% | {p['state_counts']['LOST']}→{m['state_counts']['LOST']} | "
                     f"{s['events']['possible_racket_contacts']} / {s['events']['possible_table_bounces']} |")
    for s in summaries:
        validation = s.get("manual_validation", {})
        if not validation.get("available"):
            continue
        lines += ["", f"## Auditoría manual: {Path(s['video']).name}", "",
                  "| Método | Cobertura | Mediana / P95 / máximo / RMSE (px) | Falsos positivos observables | Frames perdidos |",
                  "|---|---:|---:|---:|---:|"]
        for name, label in (("isolated", "Detector aislado"), ("temporal", "Continuidad temporal"),
                            ("physical", "Tracker físico proyectado"),
                            ("multimodal", "Fusión cuerpo-raquetas-bola")):
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
    lines += ["", "Los archivos `metrics/*__ball_tracking__frames.csv` y los MP4 de `videos/` permanecen locales. "
              "Los conteos no prueban exactitud: la auditoría "
              "manual es pequeña y debe ampliarse antes de ajustar umbrales o estudiar Magnus.", "",
              "Variables identificables por ahora: coordenadas 2D, estado observado/predicho, cobertura y velocidad aparente px/s "
              "si hay suficientes observaciones. No identificables: posición/velocidad 3D calibradas y spin; resultado de spin: "
              "`spin_not_identifiable`.", "",
              "Las detecciones de eventos continúan siendo hipótesis. Las anotaciones actuales no son exhaustivas, por lo que "
              "permiten medir recuperación alrededor de eventos conocidos, pero no precisión global de eventos ni PCK de raqueta."]
    artifact_path(output_root, "reports", "batch", "ball_tracking", "report", ".md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clips", nargs="*", default=DEFAULT_CLIPS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--manifest", type=Path, help="Reviewed rally manifest (source-bound assisted pipeline)")
    parser.add_argument("--interactive", action="store_true", help="Pause and inspect manual recovery episodes")
    parser.add_argument("--clip-id", action="append", default=[], help="Manifest clip ID; repeat for multiple clips")
    parser.add_argument("--interventions-dir", type=Path, help="Persist/replay source-bound manual corrections")
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    args = parser.parse_args()
    if args.manifest is not None:
        from seima_mocap.rally_pipeline import process_rally_manifest
        process_rally_manifest(ROOT, args.manifest, args.output_root, clip_ids=args.clip_id,
                               interactive=args.interactive, journal_root=args.interventions_dir,
                               make_overlay=not args.no_overlay, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                               detect_event=detect_event)
        return
    if args.interactive or args.clip_id or args.interventions_dir:
        parser.error("--interactive, --clip-id and --interventions-dir require --manifest")
    ensure_output_layout(args.output_root)
    summaries = [process(stem, args.output_root, not args.no_overlay) for stem in args.clips]
    write_report(args.output_root, summaries)


if __name__ == "__main__":
    main()
