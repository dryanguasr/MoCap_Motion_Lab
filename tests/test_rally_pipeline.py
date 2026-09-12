import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from seima_mocap.rally_clips import export_manifest
from seima_mocap.rally_pipeline import process_rally_manifest
from seima_mocap.table_geometry import TABLE_POINT_ORDER, NET_POINT_ORDER, sha256_file
from seima_mocap.video_io import atomic_json, probe_video, resolve_video_tool, write_timed_video


def fixture_manifest(tmp_path):
    try:
        resolve_video_tool("ffmpeg")
        resolve_video_tool("ffprobe")
    except FileNotFoundError:
        pytest.skip("FFmpeg/FFprobe not installed")
    video = tmp_path / "input.mp4"
    times = np.arange(35)*.02
    frames = [np.zeros((180, 320, 3), np.uint8) for _ in times]
    write_timed_video(iter(frames), times, video)
    source = {"path": str(video), "sha256": sha256_file(video), **probe_video(video)}
    annotation = {"schema_version": "seima.table-geometry.v1", "video_sha256": source["sha256"],
                  "source": {"frame_range": [0, 35]}, "review": {"checked_frames": [0, 17, 34]},
                  "table_surface": {"point_order": list(TABLE_POINT_ORDER), "points": [
                      {"name": name, "xy": xy} for name, xy in zip(TABLE_POINT_ORDER, [[60, 100], [260, 100], [280, 150], [40, 150]])]},
                  "net": {"point_order": list(NET_POINT_ORDER), "points": [
                      {"name": name, "xy": xy} for name, xy in zip(NET_POINT_ORDER, [[80, 90], [240, 90], [240, 115], [80, 115]])]}}
    geometry_path = tmp_path / "geometry.json"
    atomic_json(geometry_path, annotation)
    value = {"schema_version": "seima.rallies.v1", "source": source, "caches": {},
             "scene_config": {"pixels_per_m_vertical": 100, "roi_polygon_xy": [[0, 0], [319, 0], [319, 179], [0, 179]]},
             "shots": [{"id": "s1", "start_frame": 0, "end_frame": 35, "reviewed": True,
                        "geometry": {"path": str(geometry_path), "sha256": sha256_file(geometry_path)}}],
             "rallies": [{"id": "r1", "start_frame": 0, "end_frame": 35, "accepted": True}], "clips": []}
    destination = tmp_path / "manifest.json"
    atomic_json(destination, value)
    return destination


def test_reviewed_manifest_export_process_pause_and_replay(tmp_path):
    manifest_path = fixture_manifest(tmp_path)
    root = Path(__file__).resolve().parents[1]
    exported = export_manifest(manifest_path, tmp_path / "exports")
    assert exported["clips"][0]["source_frames"] == list(range(35))
    initial = process_rally_manifest(root, manifest_path, tmp_path / "out", make_overlay=True)
    assert initial[0]["status"] == "needs_inspection"
    assert initial[0]["frames_processed"] == 26
    journal_path = Path(initial[0]["journal"])
    data = json.loads(journal_path.read_text())
    data["actions"] = [{"action": "end", "id": "manual_end", "frame": 34, "source_frame": 34,
                        "source_pts_s": exported["source"]["pts_s"][34], "timestamp_s": .68,
                        "episode_start": 0, "trigger_frame": 25}]
    atomic_json(journal_path, data)
    replayed = process_rally_manifest(root, manifest_path, tmp_path / "out", make_overlay=False)
    assert replayed[0]["status"] == "completed" and replayed[0]["ended_by_user"]
    assert replayed[0]["frames_processed"] == 35
    first_hash = sha256_file(replayed[0]["frame_metrics"])
    again = process_rally_manifest(root, manifest_path, tmp_path / "out", make_overlay=False)
    assert sha256_file(again[0]["frame_metrics"]) == first_hash


def test_geometry_and_source_changes_block_processing(tmp_path):
    manifest_path = fixture_manifest(tmp_path)
    value = json.loads(manifest_path.read_text())
    geometry_path = Path(value["shots"][0]["geometry"]["path"])
    geometry_path.write_text(geometry_path.read_text()+" ")
    with pytest.raises(ValueError, match="geometry changed"):
        export_manifest(manifest_path, tmp_path / "out")
    value["shots"][0]["reviewed"] = False
    atomic_json(manifest_path, value)
    with pytest.raises(ValueError, match="Review table"):
        export_manifest(manifest_path, tmp_path / "out")


def test_cli_rejects_interactive_without_manifest(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/process_ball_tracking_real.py"
    spec = importlib.util.spec_from_file_location("process_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", [str(path), "--interactive"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


def test_interactive_pause_restores_inspection_cursor(tmp_path, monkeypatch):
    import seima_mocap.rally_pipeline as pipeline
    manifest_path = fixture_manifest(tmp_path)
    root = Path(__file__).resolve().parents[1]
    requests = []
    def inspector(video, frames, times, request, tracks):
        requests.append(request)
        return {"action": "pause", "frame": 12}
    monkeypatch.setattr(pipeline, "inspect_ball", inspector)
    first = pipeline.process_rally_manifest(root, manifest_path, tmp_path/'out', interactive=True, make_overlay=False)
    assert first[0]["status"] == "paused"
    second = pipeline.process_rally_manifest(root, manifest_path, tmp_path/'out', interactive=True, make_overlay=False)
    assert second[0]["status"] == "paused"
    assert requests[1]["inspection_frame"] == 12
    assert requests[1]["progress"] == requests[0]["progress"]


def test_progress_uses_vfr_duration_and_pending_frontier():
    from seima_mocap.rally_pipeline import inspection_progress
    value = {"source": {"pts_s": [0, .1, .3, .4, .7, 1.], "end_pts_s": 1.2}}
    parts = [{"start_frame": 0, "end_frame": 3}, {"start_frame": 3, "end_frame": 6}]
    p = inspection_progress(value, parts, 1, 1)
    assert p["clip_number"] == 2 and p["other_clips_remaining"] == 0
    assert p["remaining_video_seconds"] == pytest.approx(.5)
    assert p["total_percent"] == pytest.approx(100*.7/1.2)
    assert inspection_progress(value, parts, 1, 3)["total_percent"] == pytest.approx(100)
