"""Run source-bound rally views with optional interactive manual recovery."""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .assisted_tracking import replay_tracking, smooth_replay
from .ball_detection import BallDetectorConfig, detect_ball_candidates
from .ball_tracking import BallPhysicalParams
from .ball_video_tracking import PlanarSceneCalibration, TrackStatus
from .ball_inspector import inspect_ball
from .interaction_tracking import InteractionContext, TwoRacketTracker
from .manual_recovery import InterventionJournal
from .output_layout import artifact_path
from .pose_context import load_body_frames
from .racketvision_adapter import RacketVisionFrame, load_cache
from .rally_clips import clip_parts, local_model_frames, reviewed_geometry, validate_manifest
from .table_geometry import sha256_file
from .video_io import atomic_json, iter_frames, read_frame, resolve_video_tool, write_timed_video


def prepare_evidence(root, manifest, part, model_frames=None, body_frames=None):
    source = manifest["source"]
    video = Path(source["path"])
    start, end = part["start_frame"], part["end_frame"]
    absolute_times = np.asarray(source["pts_s"][start:end])
    times = absolute_times-absolute_times[0]
    shot = next(s for s in manifest["shots"] if s["id"] == part["shot_id"])
    geometry = reviewed_geometry(manifest, shot)
    scene = PlanarSceneCalibration(geometry["table_polygon_xy"], net_polygon_xy=geometry["net_polygon_xy"],
                                   pixels_per_m_vertical=manifest["scene_config"]["pixels_per_m_vertical"])
    if model_frames is None:
        cache = manifest["caches"].get("racketvision")
        model_frames = load_cache(cache["path"]) if cache else []
    local_models = local_model_frames(model_frames, start, end, absolute_times[0], source_times=source["pts_s"])
    while len(local_models) < len(times):
        index = len(local_models)
        local_models.append(RacketVisionFrame(index, float(times[index]), (), ()))
    if body_frames is None:
        pose = manifest["caches"].get("poses")
        body_frames, _ = load_body_frames(None if pose is None else pose["path"], None,
                                         source["frame_count"], source["width"], source["height"])
    bodies = body_frames[start:end]
    racket_tracker = TwoRacketTracker()
    contexts = [InteractionContext(float(time), bodies[i], racket_tracker.step(float(time), local_models[i].rackets,
                                                                               bodies[i], source["width"]))
                for i, time in enumerate(times)]
    # Background belongs to this reviewed fixed-camera shot, never to the long video.
    targets = sorted(set(np.linspace(shot["start_frame"], shot["end_frame"]-1, 17, dtype=int)))
    samples = [cv2.cvtColor(read_frame(video, int(i)), cv2.COLOR_BGR2GRAY) for i in targets]
    background = np.median(np.stack(samples), axis=0).astype(np.uint8)
    config = BallDetectorConfig.from_json(root / "config/ball_detection_defaults.json")
    # Derive the search area from this shot's reviewed table, not a stale long-video ROI.
    polygon = geometry["table_polygon_xy"]
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    span = hi[0]-lo[0]
    x0, y0 = np.maximum([0, 0], lo-[.65*span, .65*span])
    x1, y1 = np.minimum([source["width"]-1, source["height"]-1], hi+[.65*span, .25*span])
    roi = [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]
    evidence = [[] for _ in times]
    # One frame of visual context on each side, clipped to the same shot.
    first, last = max(shot["start_frame"], start-1), min(shot["end_frame"], end+1)
    previous = current = None
    for original_index, following in iter_frames(video, first, last):
        target = original_index-1
        if previous is not None and current is not None and start <= target < end:
            index = target-start
            evidence[index] = detect_ball_candidates(previous, current, following, float(times[index]), config,
                                                     roi_polygon_xy=roi,
                                                     background_gray=background)
            if index % 250 == 0:
                print(f"{part['id']}: visual evidence {index}/{len(times)}", flush=True)
        previous, current = current, following
    recovery_cache = {}
    # Long exposure streaks in this recording exceed the ordinary detector's
    # 90-pixel gate. Broaden only manual recovery evidence, never automatic runs.
    recovery_config = replace(config, max_dimension_px=max(config.max_dimension_px, int(260*source["width"]/2156)),
                              max_area_px=max(config.max_area_px, 6000*(source["width"]/2156)**2))

    def recovery_candidates(index):
        if index not in recovery_cache:
            original = start+index
            if not shot["start_frame"] < original < shot["end_frame"]-1:
                recovery_cache[index] = []
            else:
                frames = [read_frame(video, original+delta) for delta in (-1, 0, 1)]
                recovery_cache[index] = detect_ball_candidates(*frames, float(times[index]), recovery_config,
                                                               roi_polygon_xy=roi, background_gray=background, manual_recovery=True)
        return recovery_cache[index]

    return times, evidence, local_models, contexts, scene, recovery_candidates


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def result_rows(result, manifest, part, times):
    source = manifest["source"]
    output = []
    for i, (track, origin) in enumerate(zip(result.tracks, result.provenance)):
        row = {"clip_id": part["id"], "rally_id": part["rally_id"], "shot_id": part["shot_id"],
               "frame": i, "source_frame": part["start_frame"]+i,
               "timestamp_s": float(times[i]), "source_pts_s": source["pts_s"][part["start_frame"]+i],
               "status": track.status.value, "confidence": track.confidence,
               "observation_source": "" if track.selected_candidate is None else track.selected_candidate.observation.source,
               **{key: value for key, value in origin.items() if key != "protected"}}
        for kind, point in (("observed", track.observed_xy), ("predicted", track.predicted_xy)):
            row[f"{kind}_x_px"] = "" if point is None else float(point[0])
            row[f"{kind}_y_px"] = "" if point is None else float(point[1])
        output.append(row)
    return output


def annotated_frames(manifest, part, result):
    colors = {TrackStatus.OBSERVED: (60, 230, 80), TrackStatus.PREDICTED: (0, 160, 255),
              TrackStatus.LOST: (60, 60, 240), TrackStatus.UNINITIALIZED: (150, 150, 150)}
    geometry = reviewed_geometry(manifest, next(s for s in manifest["shots"] if s["id"] == part["shot_id"]))
    trail, prior_segment = [], None
    for i, (_, frame) in enumerate(iter_frames(Path(manifest["source"]["path"]), part["start_frame"],
                                                part["start_frame"]+len(result.tracks))):
        tracked, origin = result.tracks[i], result.provenance[i]
        if origin["segment_id"] != prior_segment or tracked.status == TrackStatus.LOST:
            trail = []
        prior_segment = origin["segment_id"]
        for polygon in (geometry["table_polygon_xy"], geometry["net_polygon_xy"]):
            cv2.polylines(frame, [np.round(polygon).astype(np.int32)], True, (220, 150, 40), 2)
        point = tracked.observed_xy if tracked.status == TrackStatus.OBSERVED else tracked.predicted_xy
        if point is not None and tracked.status in (TrackStatus.OBSERVED, TrackStatus.PREDICTED):
            p = tuple(np.round(point).astype(int))
            cv2.circle(frame, p, 10, colors[tracked.status], 2)
            trail.append(p)
            trail = trail[-25:]
        if len(trail) > 1:
            cv2.polylines(frame, [np.asarray(trail, np.int32)], False, colors[tracked.status], 2)
        if origin["manual_seed_kind"]:
            point = (int(round(origin["manual_seed_x_px"])), int(round(origin["manual_seed_y_px"])))
            cv2.drawMarker(frame, point, (255, 70, 220), cv2.MARKER_CROSS, 28, 3)
        lines = [f"{part['id']} | source frame {part['start_frame']+i} | {tracked.status.value}",
                 f"recovery {origin['recovery_state']} | manual seed {origin['manual_seed_id'][:8]} {origin['manual_seed_kind']}",
                 "GREEN=visual observation ORANGE=prediction MAGENTA=manual seed"]
        cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 1400), 105), (15, 15, 15), -1)
        for j, line in enumerate(lines):
            cv2.putText(frame, line, (12, 28+30*j), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 2)
        yield frame


def inspection_progress(manifest, parts, position, frontier):
    """Resolved timeline before the pending episode, not the navigation cursor.

    In interactive mode earlier parts have completed before reaching this one.
    Remaining seconds describe source video, not an estimate of human work time.
    """
    pts = manifest["source"]["pts_s"]
    source_end = manifest["source"]["end_pts_s"]
    durations = [(pts[p["end_frame"]] if p["end_frame"] < len(pts) else source_end)-pts[p["start_frame"]] for p in parts]
    part = parts[position]
    local = max(0, min(frontier, part["end_frame"]-part["start_frame"]))
    endpoint = part["start_frame"]+local
    elapsed = (pts[endpoint] if endpoint < len(pts) else source_end)-pts[part["start_frame"]]
    total = sum(durations)
    done = sum(durations[:position])+elapsed
    return {"clip_number": position+1, "clip_count": len(parts),
            "clip_percent": 100*elapsed/durations[position], "total_percent": 100*done/total,
            "clip_remaining_seconds": max(0., durations[position]-elapsed),
            "remaining_video_seconds": max(0., total-done), "other_clips_remaining": len(parts)-position-1}


def process_rally_manifest(root, manifest_path, output_root, *, clip_ids=(), interactive=False,
                           journal_root=None, make_overlay=True, ffmpeg=None, ffprobe=None, detect_event=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    validate_manifest(manifest)
    parts = clip_parts(manifest)
    if clip_ids:
        unknown = set(clip_ids)-{p["id"] for p in parts}
        if unknown:
            raise ValueError(f"Unknown or unaccepted clips: {sorted(unknown)}")
        parts = [p for p in parts if p["id"] in clip_ids]
    if not parts:
        raise ValueError("No accepted rallies. Run prepare_rally_clips.py --review first")
    # Complete all validation before opening any interactive windows.
    for part in parts:
        reviewed_geometry(manifest, next(s for s in manifest["shots"] if s["id"] == part["shot_id"]))
    ffmpeg, ffprobe = resolve_video_tool("ffmpeg", ffmpeg), resolve_video_tool("ffprobe", ffprobe)
    source = manifest["source"]
    model_cache = manifest["caches"].get("racketvision")
    models = load_cache(model_cache["path"]) if model_cache else []
    pose_cache = manifest["caches"].get("poses")
    bodies, _ = load_body_frames(None if pose_cache is None else pose_cache["path"], None,
                                 source["frame_count"], source["width"], source["height"])
    params = BallPhysicalParams.from_json(root / "config/ball_tracking_defaults.json")
    journals = Path(journal_root) if journal_root is not None else Path(manifest_path).parent / "interventions"
    pipeline = "ball_tracking_assisted"
    summaries = []
    for position, part in enumerate(parts):
        print(f"Preparando intercambio {position+1}/{len(parts)}: {part['id']} ({part['end_frame']-part['start_frame']} fotogramas)", flush=True)
        times, evidence, local_models, contexts, scene, recovery_candidates = prepare_evidence(root, manifest, part, models, bodies)
        shot = next(s for s in manifest["shots"] if s["id"] == part["shot_id"])
        identity = {"source_sha256": source["sha256"], "clip_id": part["id"],
                    "frame_range": [part["start_frame"], part["end_frame"]], "geometry_sha256": shot["geometry"]["sha256"],
                    "caches": {key: cache["sha256"] for key, cache in manifest["caches"].items()},
                    "time_map_sha256": hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest()}
        journal = InterventionJournal(journals / f"{part['id']}.json", identity, times.tolist(),
                                       list(range(part["start_frame"], part["end_frame"])),
                                       source["pts_s"][part["start_frame"]:part["end_frame"]], source["width"], source["height"])
        journal.save()
        paused = False
        while True:
            result = replay_tracking(times, evidence, local_models, contexts, scene, params, journal.actions,
                                     recovery_candidates=recovery_candidates)
            if not interactive or not result.requests:
                break
            request = result.requests[0]
            pending = journal.payload.get("pending_request")
            if pending and all(pending.get(k) == request.get(k) for k in ("episode_start", "trigger_frame", "reason")):
                request = {**request, "inspection_frame": pending["inspection_frame"]}
            request = {**request, "progress": inspection_progress(manifest, parts, position, request["episode_start"]),
                       "trail_segments": [p["segment_id"] for p in result.provenance]}
            action = inspect_ball(Path(source["path"]), journal.source_frames, times, request, result.tracks)
            if action["action"] == "pause":
                journal.payload["pending_request"] = {**{k: v for k, v in request.items() if k not in ("progress", "trail_segments")},
                                                      "inspection_frame": action["frame"]}
                journal.save(action["frame"])
                paused = True
                break
            journal.append(action)
        # CSV and video share the same retrospective result and manual barriers.
        result = smooth_replay(result, scene)
        name = Path(source["path"]).stem + "__" + part["id"]
        rows = result_rows(result, manifest, part, times)
        csv_path = artifact_path(output_root, "metrics", name, pipeline, "frames", ".csv")
        write_csv(csv_path, rows)
        events, history = [], []
        last_segment = None
        context_by_frame = dict(enumerate(contexts))
        if detect_event is not None:
            for i, (track, origin) in enumerate(zip(result.tracks, result.provenance)):
                if origin["segment_id"] != last_segment or origin["protected"]:
                    history = []
                last_segment = origin["segment_id"]
                if track.observed_xy is not None and not origin["protected"]:
                    history.append((track.timestamp_s, track.observed_xy, i))
                    history = history[-8:]
                    for event in detect_event(history, context_by_frame, i, scene, events):
                        event.update(clip_id=part["id"], source_frame=part["start_frame"]+event["frame"],
                                     source_pts_s=source["pts_s"][part["start_frame"]+event["frame"]],
                                     manual_seed_id=origin["manual_seed_id"], assisted_segment=origin["assisted_segment"])
                        events.append(event)
        atomic_json(artifact_path(output_root, "events", name, pipeline, "events", ".json"), {"events": events})
        summary = {"identity": identity, "status": "paused" if paused else "needs_inspection" if result.requests else "completed",
                   "geometry_provenance": reviewed_geometry(manifest, shot)["metadata"].get("review", {}),
                   "frames_processed": len(rows), "source_frames_expected": len(times), "ended_by_user": result.ended,
                   "manual_interventions": len(journal.actions),
                   "manual_visible_seeds": sum(a["action"] == "seed" and a["kind"] == "visible" for a in journal.actions),
                   "manual_estimated_seeds": sum(a["action"] == "seed" and a["kind"] == "estimated_occluded" for a in journal.actions),
                   "confirmed_recoveries": sum(row["recovery_state"] == "confirmed" and
                                               (i == 0 or rows[i-1]["recovery_state"] != "confirmed") for i, row in enumerate(rows)),
                   "automatic_observation_frames": sum(row["status"] == "OBSERVED" for row in rows),
                   "predicted_frames": sum(row["status"] == "PREDICTED" for row in rows),
                   "lost_frames": sum(row["status"] == "LOST" for row in rows),
                   "pending_inspections": result.requests, "journal": str(journal.path.resolve()),
                   "frame_metrics": str(csv_path.resolve()), "visual_identity_validation": "pending_human_review",
                   "cache_provenance": manifest["caches"], "timestamp_source": source["timestamp_source"]}
        if make_overlay and len(rows) >= 3:
            overlay = artifact_path(output_root, "videos", name, pipeline, "diagnostic", ".mp4")
            write_timed_video(annotated_frames(manifest, part, result), times[:len(rows)], overlay, ffmpeg=ffmpeg, ffprobe=ffprobe)
            summary["overlay"] = str(overlay.resolve())
        atomic_json(artifact_path(output_root, "summaries", name, pipeline, "summary", ".json"), summary)
        summaries.append(summary)
        print(f"{part['id']}: {summary['status']}; {len(rows)} frames; {len(journal.actions)} interventions", flush=True)
        if paused:
            break
    return summaries
