import json

import pytest

from seima_mocap.racketvision_adapter import SCHEMA_VERSION, load_cache, write_cache


def test_cache_round_trip(tmp_path):
    path = tmp_path / "cache.jsonl"
    frames = [{
        "frame": 0,
        "timestamp_s": 0.0,
        "ball_candidates": [{"x_px": 10, "y_px": 20, "confidence": .8}],
        "rackets": [{
            "bbox_xyxy": [1, 2, 30, 50],
            "keypoints_xy": [[10, 3], [10, 20], [10, 45], [4, 10], [16, 10]],
            "keypoint_scores": [.9] * 5,
            "confidence": .85,
        }],
    }]
    manifest = write_cache(path, iter(frames), {"device": "cpu"})
    loaded = load_cache(path, expected_frames=1)
    assert manifest.exists()
    assert loaded[0].ball_candidates[0].source == "racketvision_balltrack"
    assert loaded[0].rackets[0].point("handle").tolist() == [10, 45]


def test_cache_rejects_non_contiguous_frames(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "frame": 2,
                                "timestamp_s": 0, "ball_candidates": [], "rackets": []}) + "\n")
    with pytest.raises(ValueError, match="contiguous"):
        load_cache(path)


def test_cache_rejects_tampering_when_manifest_exists(tmp_path):
    path = tmp_path / "cache.jsonl"
    write_cache(path, iter([{"frame": 0, "timestamp_s": 0, "ball_candidates": [], "rackets": []}]), {})
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_cache(path)
