from pathlib import Path

import pytest

from seima_mocap.output_layout import CATEGORIES, artifact_path, ensure_output_layout


def test_layout_creates_one_directory_per_artifact_type(tmp_path: Path):
    ensure_output_layout(tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == set(CATEGORIES)


def test_artifact_name_is_flat_and_self_identifying(tmp_path: Path):
    path = artifact_path(tmp_path, "videos", "clip 01", "left/player", "annotated", "mp4")
    assert path == tmp_path / "videos" / "clip_01__left_player__annotated.mp4"


def test_artifact_path_rejects_unknown_category(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown output category"):
        artifact_path(tmp_path, "misc", "clip", "pose", "result", ".bin")
