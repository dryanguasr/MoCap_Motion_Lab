"""Question-driven review; every persisted decision resolves a specific uncertainty."""
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np

from .rally_review import edit_intervals
from .rally_clips import validate_manifest
from .table_geometry import annotation_path, sha256_file
from .video_io import atomic_json, read_frame


def pending_checks(value):
    """Camera boundaries first, then unresolved rallies, then required geometry."""
    checks = [("camera", s["id"]) for s in value["shots"][1:]
              if not s.get("boundary_confirmed")]
    checks += [("rally", r["id"]) for r in value["rallies"] if not r["accepted"]]
    checks += [("geometry", s["id"]) for s in value["shots"] if not s["reviewed"] and any(
        r["accepted"] and r["start_frame"] < s["end_frame"] and r["end_frame"] > s["start_frame"]
        for r in value["rallies"])]
    return checks


def ask(video, times, start, end, title, instruction, choices, *, frame=None, overlay=None):
    """Mouse buttons and a clickable timeline; returns action and original frame."""
    frame = start if frame is None else frame
    window = "SEIMA - Revision guiada"
    state = {"action": None, "buttons": [], "width": 1000, "playing": False}

    def mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for x0, y0, x1, y1, action in state["buttons"]:
            if x0 <= x <= x1 and y0 <= y <= y1:
                state["action"] = action
                return
        if 90 <= y <= 110:
            state["seek"] = start + round(np.clip((x-15)/(state["width"]-30), 0, 1)*(end-start-1))

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    raw_index, raw = -1, None
    try:
        while True:
            if raw_index != frame:
                raw = read_frame(video, frame)
                raw_index = frame
            picture = raw.copy()
            if overlay:
                for section, color in (("table_surface", (40, 225, 255)), ("net", (255, 70, 210))):
                    xy = np.round([p["xy"] for p in overlay[section]["points"]]).astype(np.int32)
                    cv2.polylines(picture, [xy], True, color, 3)
            scale = min(1., 1100/picture.shape[1], 580/picture.shape[0])
            picture = cv2.resize(picture, None, fx=scale, fy=scale)
            width = max(1000, picture.shape[1])
            canvas = np.zeros((picture.shape[0]+255, width, 3), dtype=np.uint8)
            canvas[120:120+picture.shape[0], :picture.shape[1]] = picture
            for i, text in enumerate((title, instruction, f"Fotograma {frame} | {times[frame]:.3f} s | Barra: ir a otro momento")):
                cv2.putText(canvas, text, (15, 25+i*27), cv2.FONT_HERSHEY_SIMPLEX, .53, (240, 240, 240), 1)
            cv2.line(canvas, (15, 100), (width-15, 100), (110, 110, 110), 5)
            x = 15+round((width-30)*(frame-start)/max(1, end-start-1))
            cv2.circle(canvas, (x, 100), 7, (0, 220, 255), -1)
            controls = [("play", "Pausa" if state["playing"] else "Reproducir lento"),
                        ("back", "Anterior frame"), ("forward", "Siguiente frame"), ("quit", "Guardar y salir")]
            state["buttons"] = []
            state["width"] = width
            for row, items in enumerate((controls, choices)):
                cell = width//len(items)
                for col, (action, label) in enumerate(items):
                    x0, y0 = col*cell+5, picture.shape[0]+135+row*53
                    cv2.rectangle(canvas, (x0, y0), (x0+cell-10, y0+43), (65, 65, 65), -1)
                    cv2.putText(canvas, label, (x0+8, y0+27), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
                    state["buttons"].append((x0, y0, x0+cell-10, y0+43, action))
            cv2.imshow(window, canvas)
            delay = max(1, round(2000*(times[min(frame+1, end-1)]-times[frame]))) if state["playing"] else 30
            key = cv2.waitKeyEx(delay)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord("q")):
                return "quit", frame
            action, state["action"] = state["action"], None
            if action == "quit" or action in dict(choices):
                return action, frame
            if action == "play" or key == 32:
                state["playing"] = not state["playing"]
            if action in ("back", "forward") or key in (ord("a"), ord("d")):
                frame += -1 if action == "back" or key == ord("a") else 1
                state["playing"] = False
            if "seek" in state:
                frame = state.pop("seek")
                state["playing"] = False
            if state["playing"]:
                frame += 1
                if frame >= end:
                    frame = start
            frame = int(np.clip(frame, start, end-1))
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass


def review_manifest(path, root, annotate):
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(value)
    video = Path(value["source"]["path"])
    times = np.asarray(value["source"]["pts_s"])
    message = ""
    while pending_checks(value):
        kind, identity = pending_checks(value)[0]
        items = value["rallies" if kind == "rally" else "shots"]
        item = next(s for s in items if s["id"] == identity)
        a, b = item["start_frame"], item["end_frame"]
        cursor = value.get("guided_cursor", {})
        resume = cursor.get("frame", a) if cursor.get("id") == identity else a
        if kind == "camera":
            lo, hi = max(0, int(np.searchsorted(times, times[a]-1))), min(len(times), int(np.searchsorted(times, times[a]+1))+1)
            action, frame = ask(video, times, lo, hi, "Posible cambio de camara: " + identity,
                                message or "Reproduzca el fragmento. Hay un corte, zoom o desplazamiento de la mesa?",
                                [("yes", "Si, separar tomas"), ("no", "No, misma toma")], frame=int(np.clip(resume, lo, hi-1)))
            if action == "yes":
                action, frame = ask(video, times, lo, hi, "Precisar cambio de camara",
                                    "Busque el PRIMER fotograma del nuevo encuadre. Use Anterior/Siguiente frame.",
                                    [("set", "La nueva toma empieza aqui")], frame=a)
                previous = value["shots"][value["shots"].index(item)-1]
                if action == "set" and min(frame-previous["start_frame"], b-frame) >= 3:
                    if frame != a:
                        previous.update(end_frame=frame, reviewed=False, geometry=None)
                        item.update(start_frame=frame, reviewed=False, geometry=None)
                        value["clips"] = []
                    item["boundary_confirmed"] = True
                elif action != "quit":
                    message = "Deje al menos tres fotogramas en cada toma."
            elif action == "no":
                value = edit_intervals(value, "shot_merge", a)
        elif kind == "rally":
            action, frame = ask(video, times, a, b, "Intercambio pendiente: " + identity,
                                message or "Compruebe que contiene un solo intercambio completo, con margen al inicio y final.",
                                [("accept", "Esta completo"), ("adjust", "Corregir limites"),
                                 ("split", "Hay varios"), ("merge", "Sigue en el proximo"), ("delete", "No hay juego")], frame=int(np.clip(resume, a, b-1)))
            selected = value["rallies"].index(item)
            if action == "adjust":
                # Allow extension into adjacent gaps, but never across another rally.
                lo = value["rallies"][selected-1]["end_frame"] if selected else 0
                hi = value["rallies"][selected+1]["start_frame"] if selected+1 < len(value["rallies"]) else len(times)
                for boundary, prompt, initial in (("start", "Busque el PRIMER fotograma que quiere conservar, con margen antes del juego.", a),
                                                  ("end", "Busque el ULTIMO fotograma que quiere conservar, con margen despues del juego.", b-1)):
                    action, frame = ask(video, times, lo, hi, "Corregir " + ("inicio" if boundary == "start" else "final"),
                                        prompt, [("set", "Usar este fotograma")], frame=initial)
                    if action == "quit":
                        break
                    # Commit both bounds together to allow moving the entire interval.
                    if boundary == "start":
                        new_start = frame
                    elif frame-new_start >= 2:
                        candidate = deepcopy(value)
                        candidate["rallies"][selected].update(start_frame=new_start, end_frame=frame+1, accepted=False)
                        candidate["clips"] = []
                        validate_manifest(candidate, check_files=False)
                        value = candidate
                    else:
                        message = "Limites invalidos: conserve al menos tres fotogramas. Intente de nuevo."
            elif action == "split":
                action, frame = ask(video, times, a, b, "Separar intercambios",
                                    "Busque el inicio del segundo intercambio. Este fotograma iniciara el nuevo clip.",
                                    [("set", "Separar aqui")], frame=frame)
                if action == "set":
                    if min(frame-a, b-frame) >= 3:
                        value = edit_intervals(value, "split", frame, selected)
                    else:
                        message = "El corte necesita al menos tres fotogramas a cada lado."
            elif action in ("accept", "delete"):
                value = edit_intervals(value, action, frame, selected)
            elif action == "merge":
                if selected+1 < len(value["rallies"]):
                    value = edit_intervals(value, "merge", frame, selected)
                else:
                    message = "No hay otro clip propuesto. Use Corregir limites para ampliar este."
        else:
            initial_path = Path(item["geometry"]["path"]) if item.get("geometry") else annotation_path(root / "data/annotations/table_geometry", video.stem)
            initial = json.loads(initial_path.read_text(encoding="utf-8")) if initial_path.exists() else None
            stem = video.stem + "__" + identity
            directory = path.parent / "geometry"
            action, frame = "continue", a
            if annotate(video, directory, root / "data/processed/diagnostics", frame_range=(a, b),
                        output_stem=stem, initial_geometry=initial, guided=True):
                geometry = annotation_path(directory, stem)
                item.update(reviewed=True, geometry={"path": str(geometry.resolve()), "sha256": sha256_file(geometry)})
                value["clips"] = []
            else:
                action = "quit"
        value["guided_cursor"] = {"id": identity, "frame": frame}
        atomic_json(path, value)
        if action == "quit":
            return False
    print("Revision terminada. Los intercambios aceptados estan listos para exportar.", flush=True)
    return True
