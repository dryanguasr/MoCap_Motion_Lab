"""Local source-video editor. Mutations remain proposals until explicitly accepted."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import uuid

import cv2
import numpy as np

from .rally_clips import validate_manifest
from .table_geometry import annotation_path, sha256_file
from .video_io import atomic_json, read_frame


def review_manifest(path: Path, root: Path, annotate):
    """Default entry point: guided questions. Free editing is explicitly opt-in."""
    from .guided_review import review_manifest as guided
    return guided(path, root, annotate)


def edit_intervals(manifest, action, frame, selected=0):
    """Pure transactional editor; errors leave the original manifest untouched."""
    value = deepcopy(manifest)
    rallies = value["rallies"]
    item = rallies[selected] if rallies else None
    n = value["source"]["frame_count"]
    if action == "add":
        occupied_end = min([r["start_frame"] for r in rallies if r["start_frame"] > frame] + [n])
        rallies.append({"id": "rally_" + uuid.uuid4().hex[:8], "start_frame": frame,
                        "end_frame": min(frame+3, occupied_end), "accepted": False})
    elif action == "shot_split":
        shot = next(s for s in value["shots"] if s["start_frame"] <= frame < s["end_frame"])
        if min(frame-shot["start_frame"], shot["end_frame"]-frame) < 3:
            raise ValueError("Deje al menos tres fotogramas a cada lado")
        tail = {**shot, "id": "shot_" + uuid.uuid4().hex[:8], "start_frame": frame,
                "reviewed": False, "geometry": None, "reason": "manual_camera_boundary"}
        shot.update(end_frame=frame, reviewed=False, geometry=None)
        value["shots"].append(tail)
        value["shots"].sort(key=lambda s: s["start_frame"])
    elif action == "shot_merge":
        index = next(i for i, s in enumerate(value["shots"]) if s["start_frame"] <= frame < s["end_frame"])
        if index == 0:
            raise ValueError("No hay toma anterior")
        removed = value["shots"].pop(index)
        value["shots"][index-1].update(end_frame=removed["end_frame"], reviewed=False, geometry=None)
    elif item is None:
        raise ValueError("Agregue un intervalo primero")
    elif action == "start":
        item.update(start_frame=frame, accepted=False)
    elif action == "end":
        item.update(end_frame=frame+1, accepted=False)
    elif action == "delete":
        rallies.pop(selected)
    elif action == "split":
        tail = {**item, "id": "rally_" + uuid.uuid4().hex[:8], "start_frame": frame, "accepted": False}
        item.update(end_frame=frame, accepted=False)
        rallies.append(tail)
    elif action == "merge":
        ordered = sorted(rallies, key=lambda row: row["start_frame"])
        index = ordered.index(item)
        if index+1 == len(ordered):
            raise ValueError("No hay intercambio siguiente")
        following = ordered[index+1]
        item.update(end_frame=following["end_frame"], accepted=False)
        rallies.remove(following)
    elif action == "accept":
        item["accepted"] = not item["accepted"]
    else:
        raise ValueError("Unknown editor action")
    rallies.sort(key=lambda row: row["start_frame"])
    value["clips"] = []  # old exports are retained on disk but no longer authoritative
    validate_manifest(value, check_files=False)
    return value


def review_manifest_advanced(path: Path, root: Path, annotate):
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value)
    source = value["source"]
    video = Path(source["path"])
    times = np.asarray(source["pts_s"])
    frame = min(value.get("review_cursor", 0), len(times)-1)
    selected = 0
    playing = False
    speed = .5
    message = "Revise limites y geometria; ENTER acepta solo el intercambio seleccionado"
    window = "SEIMA - Revision de intercambios"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    dirty_frame, raw = -1, None
    try:
        while True:
            if dirty_frame != frame:
                raw = read_frame(video, frame)
                dirty_frame = frame
            scale = min(1., 1400/raw.shape[1], 740/raw.shape[0])
            picture = cv2.resize(raw, None, fx=scale, fy=scale)
            canvas = cv2.copyMakeBorder(picture, 130, 75, 0, 0, cv2.BORDER_CONSTANT)
            selected = min(selected, max(0, len(value["rallies"])-1))
            item = value["rallies"][selected] if value["rallies"] else None
            shot = next(s for s in value["shots"] if s["start_frame"] <= frame < s["end_frame"])
            text = f"{item['id']} [{item['start_frame']},{item['end_frame']}) {'ACEPTADO' if item['accepted'] else 'PROPUESTA'}" if item else "Sin intervalos"
            lines = [f"Frame {frame}/{len(times)-1} | {times[frame]:.3f}s | {speed}x | {shot['id']} mesa {'OK' if shot['reviewed'] else 'PENDIENTE'}",
                     text, "ESPACIO play | A/D frame | J/L -/+1s | V velocidad | N/P intervalo siguiente/anterior",
                     "I/O inicio/final | C agregar | X eliminar | T dividir | M unir siguiente | ENTER aceptar",
                     "B corte de camara | H unir toma anterior | G revisar mesa (1/2/3) | S guardar | Q salir"]
            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (10, 22+i*23), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
            cv2.putText(canvas, message[:160], (10, canvas.shape[0]-47), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 220, 255), 1)
            width = canvas.shape[1]-20
            for rally in value["rallies"]:
                x0, x1 = (10+int(width*rally[k]/len(times)) for k in ("start_frame", "end_frame"))
                cv2.rectangle(canvas, (x0, canvas.shape[0]-30), (x1, canvas.shape[0]-14),
                              (40, 180, 40) if rally["accepted"] else (100, 100, 180), -1)
            x = 10+int(width*frame/len(times))
            cv2.line(canvas, (x, canvas.shape[0]-35), (x, canvas.shape[0]-8), (0, 255, 255), 2)
            cv2.imshow(window, canvas)
            delay = max(1, int(1000*(times[min(frame+1, len(times)-1)]-times[frame])/speed)) if playing else 30
            key = cv2.waitKeyEx(delay)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord("q")):
                break
            try:
                if key == ord(" "):
                    playing = not playing
                elif key in (ord("a"), ord("d"), ord("j"), ord("l")):
                    playing = False
                    frame = (frame + (-1 if key == ord("a") else 1)) if key in (ord("a"), ord("d")) else int(np.searchsorted(times, times[frame] + (-1 if key == ord("j") else 1)))
                elif key == ord("v"):
                    speed = {.25: .5, .5: 1., 1.: .25}[speed]
                elif key in (ord("n"), ord("p")) and value["rallies"]:
                    selected = (selected+(1 if key == ord("n") else -1)) % len(value["rallies"])
                    frame = value["rallies"][selected]["start_frame"]
                    playing = False
                elif key == ord("g"):
                    playing = False
                    initial_path = Path(shot["geometry"]["path"]) if shot.get("geometry") else annotation_path(root / "data/annotations/table_geometry", video.stem)
                    initial = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.exists() else None
                    stem = video.stem + "__" + shot["id"]
                    directory = path.parent / "geometry"
                    if annotate(video, directory, root / "data/processed/diagnostics",
                                frame_range=(shot["start_frame"], shot["end_frame"]), output_stem=stem, initial_geometry=initial):
                        geometry = annotation_path(directory, stem)
                        shot.update(reviewed=True, geometry={"path": str(geometry.resolve()), "sha256": sha256_file(geometry)})
                        value["clips"] = []
                        atomic_json(path, value)
                elif key == ord("s"):
                    atomic_json(path, value)
                    message = "Guardado"
                else:
                    actions = {ord("i"): "start", ord("o"): "end", ord("c"): "add", ord("x"): "delete",
                               ord("t"): "split", ord("m"): "merge", 13: "accept", ord("b"): "shot_split", ord("h"): "shot_merge"}
                    if key in actions:
                        value = edit_intervals(value, actions[key], frame, selected)
                        atomic_json(path, value)
                        message = "Cambio guardado"
                        playing = False
                if playing:
                    frame += 1
                    if frame >= len(times):
                        playing = False
                frame = int(np.clip(frame, 0, len(times)-1))
            except (ValueError, OSError) as error:
                message = str(error)
                playing = False
    finally:
        value["review_cursor"] = frame
        atomic_json(path, value)
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
