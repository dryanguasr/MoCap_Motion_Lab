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


def median_frame(video: Path, count: int = 21, frame_range=None) -> tuple[np.ndarray, list[int]]:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = (0, total) if frame_range is None else frame_range
    targets = np.linspace(start, end - 1, min(count, end-start), dtype=int).tolist()
    frames = []
    if frame_range is not None:
        from seima_mocap.video_io import read_frame
        cap.release()
        frames = [read_frame(video, int(index)) for index in targets]
        if len(frames) < 3:
            raise ValueError(f"could not decode enough frames from {video}")
        return np.median(np.stack(frames), axis=0).astype(np.uint8), targets
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


def annotate(video: Path, output_root: Path, preview_root: Path, *, frame_range=None,
             output_stem=None, initial_geometry=None, guided=False) -> bool:
    image, sampled = median_frame(video, frame_range=frame_range)
    scale = min(1.0, 1400 / image.shape[1], 820 / image.shape[0])
    points: list[dict] = []
    if initial_geometry is not None:
        points = [{"xy": p["xy"], "visibility": p.get("visibility", "visible")}
                  for section in ("table_surface", "net") for p in initial_geometry[section]["points"]]
    checked = set()
    draft = output_root / f"{output_stem or video.stem}__review_draft.json"
    if guided and draft.exists():
        previous = json.loads(draft.read_text(encoding="utf-8"))
        if previous.get("frame_range") == list(frame_range) and previous.get("video_sha256") == sha256_file(video):
            points = previous["points"]
            checked = set(previous["checked"])
    references = {}
    if frame_range is not None:
        from seima_mocap.video_io import read_frame
        a, b = frame_range
        for key, index in zip((ord("1"), ord("2"), ord("3")), (a, (a+b-1)//2, b-1)):
            references[key] = (index, read_frame(video, index))
    display = image
    window = f"SEIMA table geometry - {video.stem}"
    selected_point = None
    needs_verification = guided and len(points) == len(LABELS)
    guidance = ""

    def save_draft():
        if guided:
            from seima_mocap.video_io import atomic_json
            atomic_json(draft, {"frame_range": list(frame_range), "video_sha256": sha256_file(video),
                                "points": points, "checked": sorted(checked)})

    def mouse(event, x, y, flags, _parameter):
        nonlocal selected_point
        if guided and event == cv2.EVENT_LBUTTONDOWN and len(points) == len(LABELS) and 82 <= y < image.shape[0]*scale:
            if selected_point is None:
                selected_point = int(np.argmin([np.linalg.norm(np.asarray(p['xy'])*scale-[x, y]) for p in points]))
            else:
                points[selected_point] = {"xy": [round(x/scale, 2), round(y/scale, 2)],
                                          "visibility": "estimated_occluded" if flags & cv2.EVENT_FLAG_SHIFTKEY else "visible"}
                selected_point = None
                checked.clear()
            return
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(LABELS) and 82 <= y < image.shape[0]*scale and 0 <= x < image.shape[1]*scale:
            points.append({"xy": [round(x / scale, 2), round(y / scale, 2)],
                           "visibility": "estimated_occluded" if flags & cv2.EVENT_FLAG_SHIFTKEY else "visible"})
            checked.clear()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            checked.clear()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    saved = False
    while True:
        if needs_verification:
            from seima_mocap.guided_review import ask
            from seima_mocap.video_io import probe_video
            times = np.asarray(probe_video(video)["pts_s"])
            overlay = {section: {"points": points[offset:offset+4]} for section, offset in (("table_surface", 0), ("net", 4))}
            for label, (index, reference) in zip(("inicio", "centro", "final"), references.values()):
                if index in checked:
                    continue
                action, _ = ask(video, times, index, index+1, "Mesa y malla: comprobar " + label,
                                "Amarillo: bordes de mesa. Rosa: malla. Las lineas coinciden con la imagen?",
                                [("yes", "Si, coinciden"), ("fix", "Corregir puntos")], overlay=overlay)
                if action == "quit":
                    save_draft()
                    cv2.destroyWindow(window)
                    return False
                if action == "fix":
                    display = reference
                    checked.clear()
                    break
                checked.add(index)
                save_draft()
            needs_verification = False
            if len(checked) == 3:
                # Reuse the existing validated, atomic persistence below.
                key = ord("s")
            else:
                key = -1
        else:
            key = -1
        canvas = render(display, points, video.stem, scale)
        if guided:
            names = ("esquina lejana izquierda de mesa", "esquina lejana derecha de mesa", "esquina cercana derecha de mesa", "esquina cercana izquierda de mesa",
                     "extremo superior izquierdo de malla", "extremo superior derecho de malla", "base derecha de malla", "base izquierda de malla")
            prompt = ("Clic en " + names[len(points)]) if len(points) < 8 else (
                "Clic en la posicion correcta de " + names[selected_point] if selected_point is not None else "Para corregir: clic en un punto dibujado, luego en su posicion correcta.")
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 82), (15, 15, 15), -1)
            for row, text in enumerate(("Mesa y malla: corregir solo los puntos necesarios", prompt,
                                        guidance or "S = comprobar y continuar | Shift+clic = punto oculto estimado | Q = guardar progreso y salir")):
                cv2.putText(canvas, text, (12, 23+row*25), cv2.FONT_HERSHEY_SIMPLEX, .48, (240, 240, 240), 1)
        elif references:
            cv2.putText(canvas, f"1=inicio 2=centro 3=final 0=mediana | revisados {len(checked)}/3",
                        (15, canvas.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        cv2.imshow(window, canvas)
        if key != ord("s"):
            key = cv2.waitKey(20) & 0xFF
            if guided and key == ord("s") and len(points) == 8 and len(checked) != 3:
                selected_point = None
                needs_verification = True
                continue
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord("q")):
            save_draft()
            break
        if key in references:
            index, display = references[key]
            checked.add(index)
        if key == ord("0"):
            display = image
        if key in (ord("u"), 8) and points:
            points.pop()
            checked.clear()
        elif key == ord("r"):
            points.clear()
            checked.clear()
        elif key == ord("s") and len(points) == len(LABELS) and (not references or len(checked) == 3):
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
            if frame_range is not None:
                payload["source"]["frame_range"] = list(frame_range)
                payload["review"]["checked_frames"] = sorted(checked)
            from seima_mocap.video_io import atomic_json
            destination = annotation_path(output_root, output_stem or video.stem)
            # Use the same validation as the consumer before accepting a save.
            from seima_mocap.table_geometry import load_table_geometry
            temporary = destination.with_suffix(".validation.json")
            atomic_json(temporary, payload)
            try:
                load_table_geometry(temporary, video_path=video)
            except ValueError as error:
                print(f"Geometria invalida: {error}", flush=True)
                guidance = "Geometria invalida. Corrija los puntos; S vuelve a comprobar."
                checked.clear()
                continue
            finally:
                temporary.unlink(missing_ok=True)
            atomic_json(destination, payload)
            if guided:
                draft.unlink(missing_ok=True)
            preview = render(image, points, video.stem, 1.0)
            cv2.imwrite(str(preview_root / f"{output_stem or video.stem}__table_geometry__preview.png"), preview)
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
