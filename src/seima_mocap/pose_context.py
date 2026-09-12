"""Load synchronized two-player pose caches into interaction contexts."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .interaction_tracking import BodyPoseFrame


RIGHT_SHOULDER = 12
RIGHT_ELBOW = 14
RIGHT_WRIST = 16


def _body_from_row(player: str, row: np.ndarray, width: int, height: int, source: str) -> BodyPoseFrame:
    valid = np.isfinite(row[:, :2]).all(axis=1)
    quality = np.minimum(row[:, 3], row[:, 4])
    reliable = valid & (quality >= .15)
    points = row[reliable, :2] * [width, height]
    bbox = None if len(points) < 4 else np.r_[points.min(axis=0), points.max(axis=0)]

    def landmark(index: int):
        return row[index, :2] * [width, height] if valid[index] and quality[index] >= .1 else None

    confidence = 0.0 if not reliable.any() else float(np.nanmean(quality[reliable]))
    landmarks_xy = row[:, :2] * [width, height]
    landmarks_xy[~valid] = np.nan
    return BodyPoseFrame(player, landmark(RIGHT_WRIST), landmark(RIGHT_ELBOW),
                         landmark(RIGHT_SHOULDER), bbox, confidence, source,
                         landmarks_xy=landmarks_xy, landmark_confidence=quality)


def load_body_frames(two_player_cache: str | Path | None, left_player_cache: str | Path | None,
                     frame_count: int, width: int, height: int) -> tuple[list[dict[str, BodyPoseFrame]], str]:
    rows = {"left": np.full((frame_count, 33, 5), np.nan),
            "right": np.full((frame_count, 33, 5), np.nan)}
    source = "unavailable"
    two_player_cache = None if two_player_cache is None else Path(two_player_cache)
    left_player_cache = None if left_player_cache is None else Path(left_player_cache)
    if two_player_cache is not None and two_player_cache.exists():
        with np.load(two_player_cache) as data:
            for player in rows:
                key = f"{player}_normalized"
                if key in data:
                    count = min(frame_count, len(data[key]))
                    rows[player][:count] = data[key][:count]
        source = str(two_player_cache)
    elif left_player_cache is not None and left_player_cache.exists():
        with np.load(left_player_cache) as data:
            count = min(frame_count, len(data["normalized"]))
            rows["left"][:count] = data["normalized"][:count]
        source = str(left_player_cache) + " (left-only fallback)"
    frames = []
    for index in range(frame_count):
        frames.append({player: _body_from_row(player, rows[player][index], width, height, source)
                       for player in ("left", "right")})
    return frames, source
