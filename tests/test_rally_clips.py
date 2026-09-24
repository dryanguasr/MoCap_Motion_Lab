from copy import deepcopy

import numpy as np
import pytest

from seima_mocap.rally_clips import activity_density, activity_intervals, clip_parts, local_model_frames, validate_manifest
from seima_mocap.rally_review import edit_intervals
from seima_mocap.racketvision_adapter import RacketVisionFrame
from seima_mocap.ball_tracking import BallObservation
from seima_mocap.video_io import export_clip, iter_frames, probe_video, read_frame, resolve_video_tool, write_timed_video


def test_same_palette_camera_cut_does_not_require_feature_matches():
    from seima_mocap.rally_clips import spatial_camera_change
    first = np.zeros((270, 480, 3), np.uint8)
    first[:, :240] = 180
    changed = np.flip(first, axis=1).copy()
    assert spatial_camera_change(first, changed)
    ball = first.copy()
    ball[120:124, 200:204] = 255
    assert not spatial_camera_change(first, ball)


def manifest():
    return {"schema_version": "seima.rallies.v1", "source": {"frame_count": 30, "pts_s": (np.arange(30)*.02).tolist()},
            "rallies": [{"id": "r1", "start_frame": 3, "end_frame": 27, "accepted": True}],
            "shots": [{"id": "s1", "start_frame": 0, "end_frame": 15, "reviewed": True, "geometry": "old"},
                      {"id": "s2", "start_frame": 15, "end_frame": 30, "reviewed": True, "geometry": "old"}],
            "clips": [], "caches": {}}


def test_ball_loss_does_not_split_rally_while_rackets_remain_active():
    t = np.arange(100)*.1
    ball = np.zeros(100, bool)
    ball[2:10] = True
    racket = np.zeros(100, bool)
    racket[2:50] = True
    racket[80:95] = True
    intervals = activity_intervals(t, ball, racket)
    assert len(intervals) == 2
    assert intervals[0][0] == 0 and intervals[0][1] >= 54


def test_activity_density_ignores_isolated_detections_and_uses_presentation_duration():
    times = np.arange(100)*.02
    signal = np.zeros(100, bool)
    signal[40] = True
    assert not activity_density(times, signal).any()
    signal[40:70] = True
    assert activity_density(times, signal)[55]


def test_rally_crossing_shots_keeps_identity_with_distinct_parts():
    parts = clip_parts(manifest())
    assert len(parts) == 2
    assert {p["rally_id"] for p in parts} == {"r1"}
    assert [(p["start_frame"], p["end_frame"]) for p in parts] == [(3, 15), (15, 27)]


def test_editor_transaction_rejects_bad_intervals_without_changing_original():
    original = manifest()
    before = deepcopy(original)
    with pytest.raises(ValueError):
        edit_intervals(original, "start", 29)
    assert original == before
    changed = edit_intervals(original, "split", 12)
    assert len(changed["rallies"]) == 2
    assert not any(r["accepted"] for r in changed["rallies"])
    joined = edit_intervals(changed, "merge", 12)
    assert len(joined["rallies"]) == 1


def test_camera_boundary_changes_invalidate_reviewed_geometry():
    value = edit_intervals(manifest(), "shot_split", 9)
    assert len(value["shots"]) == 3
    assert not value["shots"][0]["reviewed"] and value["shots"][0]["geometry"] is None
    merged = edit_intervals(value, "shot_merge", 10)
    assert len(merged["shots"]) == 2
    assert not merged["shots"][0]["reviewed"]


def test_manifest_rejects_shot_holes_and_overlapping_rallies():
    value = manifest()
    value["shots"][1]["start_frame"] = 16
    with pytest.raises(ValueError, match="partition"):
        validate_manifest(value, check_files=False)
    value = manifest()
    value["rallies"].append({"id": "r2", "start_frame": 20, "end_frame": 29, "accepted": False})
    with pytest.raises(ValueError, match="nonoverlapping"):
        validate_manifest(value, check_files=False)


def test_cache_view_rebases_nested_timestamps_without_mutating_source():
    frames = [RacketVisionFrame(i, i*.04, (BallObservation(i*.04, np.array([i, 4]), .8),), ()) for i in range(5)]
    local = local_model_frames(frames, 2, 5, .08)
    assert [f.frame_index for f in local] == [0, 1, 2]
    assert local[0].timestamp_s == local[0].ball_candidates[0].timestamp_s == 0
    assert frames[2].timestamp_s == .08
    corrected = local_model_frames(frames, 2, 5, .09, source_times=[.01, .04, .09, .11, .16])
    np.testing.assert_allclose([f.timestamp_s for f in corrected], [0., .02, .07])
    assert corrected[1].ball_candidates[0].timestamp_s == corrected[1].timestamp_s


def test_vfr_export_and_overlay_preserve_last_frame_and_irregular_pts(tmp_path):
    try:
        ffmpeg, ffprobe = resolve_video_tool("ffmpeg"), resolve_video_tool("ffprobe")
    except FileNotFoundError:
        pytest.skip("FFmpeg/FFprobe not installed")
    times = np.array([0, .017, .050, .080, .097, .14, .19])
    frames = [np.full((120, 160, 3), i*30, np.uint8) for i in range(len(times))]
    video = tmp_path / "original.mp4"
    write_timed_video(iter(frames), times, video, ffmpeg=ffmpeg, ffprobe=ffprobe)
    metadata = probe_video(video, ffprobe)
    np.testing.assert_allclose(metadata["pts_s"], times, atol=2e-5, rtol=0)
    # Every random access and nonzero-start decode must identify the same image
    # as sequential decoding, even when frame/average-FPS points elsewhere.
    sequential = [frame for _, frame in iter_frames(video, 0, len(times))]
    for index in range(len(times)):
        assert abs(float(read_frame(video, index).mean())-float(sequential[index].mean())) < 3
    decoded = [frame for _, frame in iter_frames(video, 2, 7)]
    assert all(abs(float(a.mean())-float(b.mean())) < 3 for a, b in zip(decoded, sequential[2:]))
    destination = tmp_path / "part.mp4"
    result = export_clip(video, destination, 2, 7, metadata, ffmpeg=ffmpeg, ffprobe=ffprobe)
    assert result["frame_count"] == 5
    np.testing.assert_allclose(result["pts_s"], times[2:]-times[2], atol=2e-5, rtol=0)
