"""Reviewable rally proposals and immutable source-frame provenance."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np

from .racketvision_adapter import load_cache
from .table_geometry import load_table_geometry, sha256_file
from .video_io import atomic_json, bind_verified_source_metadata, export_clip, iter_frames, probe_video

SCHEMA = "seima.rallies.v1"


def activity_intervals(times, ball_active, racket_active, inactivity_s=1.5, padding_s=.5):
    times = np.asarray(times, dtype=float)
    if inactivity_s <= 0 or padding_s < 0 or len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid timing/proposal parameters")
    active = np.asarray(ball_active, dtype=bool) | np.asarray(racket_active, dtype=bool)
    if active.shape != times.shape:
        raise ValueError("Activity must match source frames")
    indices = np.flatnonzero(active)
    if not len(indices):
        return [(0, len(times))]  # explicitly remains a proposal, never auto-accepted
    groups = np.split(indices, np.flatnonzero(np.diff(times[indices]) > inactivity_s) + 1)
    result = []
    for group in groups:
        start = int(np.searchsorted(times, times[group[0]] - padding_s, side="left"))
        end = int(np.searchsorted(times, times[group[-1]] + padding_s, side="right"))
        if end - start >= 3:
            if result and start < result[-1][1]:
                # Share padding at a midpoint; never count source frames twice.
                midpoint = (start + result[-1][1]) // 2
                result[-1] = (result[-1][0], midpoint)
                start = midpoint
            result.append((start, end))
    return result or [(0, len(times))]


def spatial_camera_change(previous, current):
    """Catch same-palette camera cuts even when feature matching fails.

    Inputs are the 480x270 proposal thumbnails. Ignore logos and most players;
    require a large change distributed across the central scene, not a ball.
    This is a proposal signal, never an automatic geometry confirmation.
    """
    delta = np.abs(current[80:200, 110:400].astype(np.float32) -
                   previous[80:200, 110:400].astype(np.float32)).mean(axis=2)
    return float(delta.mean()) > 35 and float((delta > 25).mean()) > .55


def shot_proposals(video: Path, metadata: dict) -> list[dict]:
    """Sparse visual/background comparison. All cuts require human review."""
    times = np.asarray(metadata["pts_s"])
    targets = set(np.searchsorted(times, np.arange(times[0], times[-1], .5)).tolist())
    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    reference = previous = None
    previous_image = None
    boundaries = [(0, "source_start")]
    for index, frame in iter_frames(video, 0, len(times)):
        if index not in targets:
            continue
        small = cv2.resize(frame, (480, 270))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        mask = np.full(gray.shape, 255, np.uint8)
        mask[:55] = 0  # ignore screen-recording controls and broadcast logos
        mask[:, :110] = 0
        mask[:, 400:] = 0  # exclude the foreground players, retain table/background
        keys, descriptors = orb.detectAndCompute(gray, mask)
        hist = cv2.calcHist([small], [0, 1], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist, hist)
        reason = None
        if previous_image is not None and spatial_camera_change(previous_image, small):
            reason = "spatial_camera_cut"
        if previous is not None and cv2.compareHist(previous, hist, cv2.HISTCMP_BHATTACHARYYA) > .55:
            reason = "appearance_cut"
        if reference is not None and descriptors is not None and reference[1] is not None:
            pairs = matcher.knnMatch(reference[1], descriptors, k=2)
            matches = [a for pair in pairs if len(pair) == 2 for a, b in [pair] if a.distance < .7 * b.distance]
            if len(matches) >= 12:
                a = np.float32([reference[0][m.queryIdx].pt for m in matches])
                b = np.float32([keys[m.trainIdx].pt for m in matches])
                affine, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2)
                if affine is not None and inliers.sum() >= 10 and inliers.mean() > .55:
                    scale = float(np.hypot(affine[0, 0], affine[0, 1]))
                    movement = float(np.median(np.linalg.norm(a[inliers[:, 0] == 1] - b[inliers[:, 0] == 1], axis=1)))
                    if abs(scale - 1) > .03 or movement > 5:
                        reason = "background_motion_or_zoom"
        if reason and index - boundaries[-1][0] >= 3:
            boundaries.append((index, reason))
            reference = None
        if reference is None:
            reference = (keys, descriptors)
        previous = hist
        previous_image = small
    return [{"id": f"shot_{i+1:03d}", "start_frame": start,
             "end_frame": boundaries[i+1][0] if i+1 < len(boundaries) else len(times),
             "reason": reason, "reviewed": False, "geometry": None}
            for i, (start, reason) in enumerate(boundaries)]


def source_caches(root: Path, video: Path, video_hash: str, metadata: dict) -> dict:
    """Bind original caches, never pretend they were inferred on exported clips."""
    stem = video.stem
    arrays = root / "data/processed/arrays"
    result = {}
    racket = arrays / f"{stem}__racketvision__cache.jsonl"
    if racket.exists():
        manifest = json.loads(racket.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if manifest.get("video_sha256") != video_hash:
            raise ValueError("RacketVision cache belongs to a different source video")
        frames = load_cache(racket)
        if len(frames) > metadata["frame_count"]:
            raise ValueError("RacketVision frame coverage exceeds the source")
        source_times = np.asarray(metadata["pts_s"][:len(frames)]) - metadata["pts_s"][0]
        cached_times = np.asarray([f.timestamp_s for f in frames])
        clock = "source_pts"
        if not np.allclose(cached_times, source_times, atol=1e-4, rtol=0):
            capture = cv2.VideoCapture(str(video))
            fps = capture.get(cv2.CAP_PROP_FPS)
            capture.release()
            if fps <= 0 or not np.allclose(cached_times, np.arange(len(frames))/fps, atol=1e-4, rtol=0):
                raise ValueError("Unrecognized RacketVision clock; cannot establish source frame mapping")
            clock = "legacy_frame_over_container_fps; replaced_by_source_pts_via_contiguous_frame_index"
        result["racketvision"] = {"path": str(racket.resolve()), "sha256": sha256_file(racket),
                                  "manifest_sha256": sha256_file(racket.with_suffix(".manifest.json")),
                                  "frames": len(frames), "provenance": "source_video_sha256",
                                  "clock": clock, "max_original_clock_error_s": float(np.max(abs(cached_times-source_times))) if len(frames) else 0.}
    pose = arrays / f"{stem}__two_players__pose_landmarks.npz"
    summary = root / f"data/processed/summaries/{stem}__two_players__summary.json"
    if pose.exists():
        if not summary.exists():
            raise ValueError("Legacy pose cache requires its original source summary")
        info = json.loads(summary.read_text(encoding="utf-8"))["video"]
        source_path = Path(info["input"].replace("\\", "/"))
        if not source_path.is_absolute():
            source_path = root / source_path
        if source_path.resolve() != video.resolve() or (info["width"], info["height"]) != (metadata["width"], metadata["height"]):
            raise ValueError("Pose source summary does not match the video")
        with np.load(pose) as rows:
            count = len(rows["left_normalized"])
            if len(rows["right_normalized"]) != count or count != info["frames"] or count > metadata["frame_count"]:
                raise ValueError("Pose frame coverage mismatch")
        result["poses"] = {"path": str(pose.resolve()), "sha256": sha256_file(pose), "frames": count,
                           "provenance": "legacy_source_path_dimensions_count; bound_to_video_hash_at_preparation",
                           "summary_path": str(summary.resolve()), "summary_sha256": sha256_file(summary)}
    return result


def activity_density(times, activity, window_s=.8, minimum_fraction=.3):
    """Ignore isolated detections; use source-time occupancy, not frame/FPS counts."""
    times = np.asarray(times, dtype=float)
    if window_s <= 0 or not 0 < minimum_fraction <= 1:
        raise ValueError("Invalid activity density parameters")
    left = np.searchsorted(times, times-window_s/2)
    right = np.searchsorted(times, times+window_s/2)
    weights = np.r_[np.diff(times), np.median(np.diff(times))]
    occupied = np.r_[0., np.cumsum(np.asarray(activity)*weights)]
    duration = np.r_[0., np.cumsum(weights)]
    return (occupied[right]-occupied[left])/np.maximum(1e-9, duration[right]-duration[left]) >= minimum_fraction


def cached_activity(caches: dict, metadata: dict, roi=None, *, activity_window_s=.8, activity_fraction=.3):
    n = metadata["frame_count"]
    times = np.asarray(metadata["pts_s"])
    ball = np.zeros(n, bool)
    racket = np.zeros(n, bool)
    scale = metadata["width"]/2156
    if "racketvision" in caches:
        frames = load_cache(caches["racketvision"]["path"])
        for i in range(1, len(frames)):
            dt = times[i] - times[i-1]
            for current in frames[i].ball_candidates:
                in_roi = roi is None or cv2.pointPolygonTest(np.asarray(roi, np.float32), tuple(current.pixel_xy), False) >= 0
                if current.confidence >= .45 and in_roi and any(
                    prior.confidence >= .45 and 500*scale <= np.linalg.norm(current.pixel_xy - prior.pixel_xy) / dt <= 6000*scale
                    for prior in frames[i-1].ball_candidates):
                    ball[i] = True
            for current in frames[i].rackets:
                if current.confidence >= .5 and any(
                    prior.confidence >= .5 and 1200*scale <= np.linalg.norm(current.handle_xy - prior.handle_xy) / dt <= 4500*scale
                    for prior in frames[i-1].rackets):
                    racket[i] = True
    if "poses" in caches:
        with np.load(caches["poses"]["path"]) as data:
            for side in ("left", "right"):
                rows = data[f"{side}_normalized"][:, 16]
                xy = rows[:, :2] * [metadata["width"], metadata["height"]]
                quality = np.min(rows[:, 3:5], axis=1) >= .5
                speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(times[:len(rows)])
                racket[1:len(rows)] |= quality[:-1] & quality[1:] & (speed >= 1200*scale) & (speed <= 4500*scale)
    return (activity_density(times, ball, activity_window_s, activity_fraction),
            activity_density(times, racket, activity_window_s, activity_fraction))


def prepare_manifest(root: Path, video: Path, destination: Path, *, inactivity_s=1.5, padding_s=.5,
                     ffprobe=None, activity_window_s=.8, activity_fraction=.3) -> dict:
    if destination.exists():
        raise FileExistsError(f"Already prepared; use --review or --export: {destination}")
    metadata = probe_video(video, ffprobe)
    digest = sha256_file(video)
    caches = source_caches(root, video, digest, metadata)
    scenes = json.loads((root / "config/ball_tracking_scenes.json").read_text(encoding="utf-8"))
    if video.stem not in scenes:
        raise ValueError("Source requires explicit ROI and planar scene configuration")
    ball, racket = cached_activity(caches, metadata, scenes[video.stem]["roi_polygon_xy"],
                                   activity_window_s=activity_window_s, activity_fraction=activity_fraction)
    intervals = activity_intervals(metadata["pts_s"], ball, racket, inactivity_s, padding_s)
    result = {"schema_version": SCHEMA, "source": {"path": str(video.resolve()), "sha256": digest, **metadata},
              "caches": caches, "scene_config": scenes[video.stem],
              "parameters": {"inactivity_s": inactivity_s, "padding_s": padding_s,
                             "activity_window_s": activity_window_s, "activity_fraction": activity_fraction,
                             "ball_confidence": .45, "ball_min_speed_px_s_at_2156_width": 500,
                             "racket_confidence": .5, "racket_min_speed_px_s_at_2156_width": 1200},
              "shots": shot_proposals(video, metadata),
              "rallies": [{"id": f"rally_{i+1:03d}", "start_frame": a, "end_frame": b, "accepted": False}
                          for i, (a, b) in enumerate(intervals)], "clips": [],
              "proposal_evidence": {"ball_active_frames": int(ball.sum()), "racket_active_frames": int(racket.sum())}}
    atomic_json(destination, result)
    return result


def validate_manifest(manifest: dict, *, check_files=True):
    if manifest.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported rally manifest")
    source = manifest["source"]
    n = source["frame_count"]
    pts = np.asarray(source["pts_s"])
    if len(pts) != n or not np.isfinite(pts).all() or np.any(np.diff(pts) <= 0):
        raise ValueError("Invalid source-frame time map")
    cursor = 0
    for shot in manifest["shots"]:
        if shot["start_frame"] != cursor or not cursor < shot["end_frame"] <= n:
            raise ValueError("Shots must partition source frames")
        cursor = shot["end_frame"]
    if cursor != n:
        raise ValueError("Shots must cover the complete source")
    end = 0
    for rally in sorted(manifest["rallies"], key=lambda row: row["start_frame"]):
        a, b = rally["start_frame"], rally["end_frame"]
        if not end <= a < b <= n or b-a < 3:
            raise ValueError("Rallies must be nonoverlapping and at least three frames")
        end = b
    for key in ("shots", "rallies"):
        if len({r["id"] for r in manifest[key]}) != len(manifest[key]):
            raise ValueError("Duplicate interval identifiers")
    if check_files:
        if sha256_file(source["path"]) != source["sha256"]:
            raise ValueError("Source video changed")
        bind_verified_source_metadata(source["path"], source)
        for cache in manifest["caches"].values():
            if sha256_file(cache["path"]) != cache["sha256"]:
                raise ValueError("Source cache changed")
            if "manifest_sha256" in cache and sha256_file(Path(cache["path"]).with_suffix(".manifest.json")) != cache["manifest_sha256"]:
                raise ValueError("Source cache manifest changed")
            if "summary_sha256" in cache and sha256_file(cache["summary_path"]) != cache["summary_sha256"]:
                raise ValueError("Source pose summary changed")


def clip_parts(manifest: dict, *, accepted_only=True):
    parts = []
    for rally in manifest["rallies"]:
        if accepted_only and not rally["accepted"]:
            continue
        count = 0
        for shot in manifest["shots"]:
            a, b = max(rally["start_frame"], shot["start_frame"]), min(rally["end_frame"], shot["end_frame"])
            if b > a:
                if b - a < 3:
                    raise ValueError("Shot boundary creates a part shorter than three frames; adjust review")
                count += 1
                parts.append({"id": f"{rally['id']}_part_{count:02d}", "rally_id": rally["id"],
                              "shot_id": shot["id"], "start_frame": a, "end_frame": b})
    return parts


def reviewed_geometry(manifest: dict, shot: dict) -> dict:
    if not shot["reviewed"] or not shot.get("geometry"):
        raise ValueError(f"Review table/net at start, middle and end of {shot['id']} first")
    geometry = shot["geometry"]
    if sha256_file(geometry["path"]) != geometry["sha256"]:
        raise ValueError("Reviewed geometry changed; review again")
    loaded = load_table_geometry(geometry["path"], video_path=manifest["source"]["path"])
    metadata = loaded["metadata"]
    if metadata["source"].get("frame_range") != [shot["start_frame"], shot["end_frame"]]:
        raise ValueError("Geometry belongs to a different source interval")
    if metadata["review"].get("checked_frames") != sorted(set([
            shot["start_frame"], (shot["start_frame"] + shot["end_frame"] - 1)//2, shot["end_frame"]-1])):
        raise ValueError("Three-frame geometry review incomplete")
    return loaded


def export_manifest(manifest_path: Path, output: Path, *, ffmpeg=None, ffprobe=None):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    parts = clip_parts(manifest)
    if not parts:
        raise ValueError("No accepted rallies to export; open --review first")
    for part in parts:
        shot = next(s for s in manifest["shots"] if s["id"] == part["shot_id"])
        reviewed_geometry(manifest, shot)
    source = manifest["source"]
    for part in parts:
        video = output / "videos" / f"{Path(source['path']).stem}__rallies__{part['id']}.mp4"
        metadata = export_clip(Path(source["path"]), video, part["start_frame"], part["end_frame"], source,
                               ffmpeg=ffmpeg, ffprobe=ffprobe)
        a, b = part["start_frame"], part["end_frame"]
        part.update({"path": str(video.resolve()), "sha256": sha256_file(video),
                     "source_frames": list(range(a, b)), "source_pts_s": source["pts_s"][a:b],
                     "local_pts_s": metadata["pts_s"], "audio": "omitted; analysis video",
                     "geometry": next(s["geometry"] for s in manifest["shots"] if s["id"] == part["shot_id"])})
        manifest["clips"] = [p for p in manifest["clips"] if p["id"] != part["id"]] + [part]
        atomic_json(manifest_path, manifest)
        print(f"Exported and verified {part['id']}: {b-a} source frames", flush=True)
    return manifest


def local_model_frames(frames, start, end, origin_time, source_times=None):
    """Rebase views, retaining source cache identity in the enclosing manifest."""
    result = []
    for index, frame in enumerate(frames[start:end]):
        time = (frame.timestamp_s if source_times is None else source_times[start+index]) - origin_time
        result.append(replace(frame, frame_index=index, timestamp_s=time,
                              ball_candidates=tuple(replace(o, timestamp_s=time) for o in frame.ball_candidates),
                              rackets=tuple(replace(o, timestamp_s=time) for o in frame.rackets)))
    return result
