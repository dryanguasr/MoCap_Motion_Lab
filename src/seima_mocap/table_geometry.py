"""Validated table-surface and net annotations for fixed-camera clips."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


SCHEMA_VERSION = "seima.table-geometry.v1"
TABLE_POINT_ORDER = ("far_left", "far_right", "near_right", "near_left")
NET_POINT_ORDER = ("top_left", "top_right", "base_right", "base_left")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _points(section: dict, names: tuple[str, ...]) -> np.ndarray:
    if tuple(section.get("point_order", ())) != names:
        raise ValueError(f"point_order must be {list(names)}")
    records = section.get("points", ())
    if len(records) != len(names) or tuple(item.get("name") for item in records) != names:
        raise ValueError(f"points must be named and ordered as {list(names)}")
    points = np.asarray([item["xy"] for item in records], dtype=float)
    if points.shape != (len(names), 2) or not np.all(np.isfinite(points)):
        raise ValueError("geometry points must be finite pixel pairs")
    return points


def load_table_geometry(path: str | Path, *, video_path: str | Path | None = None) -> dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported table geometry schema in {path}")
    table = _points(payload["table_surface"], TABLE_POINT_ORDER)
    net = _points(payload["net"], NET_POINT_ORDER)
    if not cv2.isContourConvex(np.round(table).astype(np.int32)) or abs(cv2.contourArea(table.astype(np.float32))) < 500:
        raise ValueError("table surface must be a non-degenerate convex quadrilateral")
    if abs(cv2.contourArea(net.astype(np.float32))) < 20:
        raise ValueError("net must be a non-degenerate quadrilateral")
    if video_path is not None:
        expected = payload.get("video_sha256")
        if expected and expected != sha256_file(video_path):
            raise ValueError(f"table geometry does not match video bytes: {path}")
    return {"table_polygon_xy": table, "net_polygon_xy": net, "metadata": payload}


def annotation_path(root: str | Path, stem: str) -> Path:
    return Path(root) / f"{stem}__table_geometry.json"
