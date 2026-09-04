"""Interactively annotate one fixed table surface and its net once per clip.

Left click the eight requested points in order. Shift+click marks an occluded
estimate. U/right-click undoes, R restarts, S saves, and Esc/Q exits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seima_mocap.table_geometry import (NET_POINT_ORDER, SCHEMA_VERSION, TABLE_POINT_ORDER,
                                        annotation_path, sha256_file)


LABELS = tuple(("table", name) for name in TABLE_POINT_ORDER) + tuple(("net", name) for name in NET_POINT_ORDER)
COLORS = {"table": (40, 225, 255), "net": (255, 70, 210)}


def median_frame(video: Path, count: int = 21) -> tuple[np.ndarray, list[int]]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    targets = np.linspace(0, total - 1, min(count, total), dtype=int).tolist()
    frames = []
    # Seeking to the sparse reference frames avoids decoding an entire long clip
    # before the interactive window becomes available.
    for index in targets:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if len(frames) < 3:
        raise ValueError(f"could not decode enough frames from {video}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8), targets


def render(image: np.ndarray, points: list[dict], stem: str, scale: float) -> np.ndarray:
    canvas = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    overlay = canvas.copy()
    for section, offset in (("table", 0), ("net", 4)):
        subset = points[offset:offset + 4]
        if len(subset) >= 2:
            xy = np.round(np.asarray([item["xy"] for item in subset]) * scale).astype(np.int32)
            cv2.polylines(overlay, [xy], len(subset) == 4, COLORS[section], 3, cv2.LINE_AA)
            if len(subset) == 4:
                cv2.fillPoly(overlay, [xy], COLORS[section])
    cv2.addWeighted(overlay, .22, canvas, .78, 0, canvas)
    for index, item in enumerate(points):
        point = tuple(np.round(np.asarray(item["xy"]) * scale).astype(int))
        color = COLORS[LABELS[index][0]]
        cv2.circle(canvas, point, 7, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"{index + 1}:{LABELS[index][1]}", (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    next_label = "complete - press S to save" if len(points) == len(LABELS) else f"NEXT {len(points)+1}: {LABELS[len(points)][0]} {LABELS[len(points)][1]}"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 82), (15, 15, 15), -1)
    cv2.putText(canvas, stem, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (245, 245, 245), 2)
    cv2.putText(canvas, next_label, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, .58, (40, 235, 255), 2)
    cv2.putText(canvas, "click=visible  shift+click=occluded  U/right=undo  R=reset  S=save  Q=quit",
                (15, 76), cv2.FONT_HERSHEY_SIMPLEX, .42, (210, 210, 210), 1)
    return canvas


def annotate(video: Path, output_root: Path, preview_root: Path) -> bool:
    image, sampled = median_frame(video)
    scale = min(1.0, 1400 / image.shape[1], 820 / image.shape[0])
    points: list[dict] = []
    window = f"SEIMA table geometry - {video.stem}"

    def mouse(event, x, y, flags, _parameter):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(LABELS):
            points.append({"xy": [round(x / scale, 2), round(y / scale, 2)],
                           "visibility": "estimated_occluded" if flags & cv2.EVENT_FLAG_SHIFTKEY else "visible"})
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse)
    saved = False
    while True:
        cv2.imshow(window, render(image, points, video.stem, scale))
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("u"), 8) and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key == ord("s") and len(points) == len(LABELS):
            output_root.mkdir(parents=True, exist_ok=True)
            preview_root.mkdir(parents=True, exist_ok=True)
            table_records = [{"name": name, **points[index]} for index, name in enumerate(TABLE_POINT_ORDER)]
            net_records = [{"name": name, **points[index + 4]} for index, name in enumerate(NET_POINT_ORDER)]
            payload = {
                "schema_version": SCHEMA_VERSION,
                "video": str(video), "video_sha256": sha256_file(video),
                "source": {"method": "manual_on_temporal_median", "sampled_frames": sampled},
                "table_surface": {"point_order": list(TABLE_POINT_ORDER), "points": table_records,
                                  "dimensions_m": {"length": 2.74, "width": 1.525}},
                "net": {"point_order": list(NET_POINT_ORDER), "points": net_records,
                        "nominal_height_m": 0.1525},
                "review": {"status": "manual_confirmed"},
            }
            destination = annotation_path(output_root, video.stem)
            destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            preview = render(image, points, video.stem, 1.0)
            cv2.imwrite(str(preview_root / f"{video.stem}__table_geometry__preview.png"), preview)
            print(destination)
            saved = True
            break
    cv2.destroyWindow(window)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/annotations/table_geometry"))
    parser.add_argument("--preview-root", type=Path, default=Path("data/processed/diagnostics"))
    args = parser.parse_args()
    for video in args.videos:
        if not annotate(video, args.output_root, args.preview_root):
            break


if __name__ == "__main__":
    main()
