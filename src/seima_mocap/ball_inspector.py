"""Local OpenCV inspection UI; never owns tracker or journal state."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .manual_recovery import display_to_source
from .video_io import read_frame


def zoom_bounds(point, width, height, size=90):
    extent_x, extent_y = min(size, width), min(size, height)
    x = int(np.clip(point[0]-extent_x//2, 0, width-extent_x))
    y = int(np.clip(point[1]-extent_y//2, 0, height-extent_y))
    return x, y, extent_x, extent_y


def inspector_click(x, y, width, height, scale, top, zoom_display=None, zoom_source=None):
    if zoom_display is not None:
        zx, zy, zw, zh = zoom_display
        if zx <= x < zx+zw and zy <= y < zy+zh:
            sx, sy, sw, sh = zoom_source
            return float(sx+(x-zx)*sw/zw), float(sy+(y-zy)*sh/zh)
    return display_to_source(x, y, scale, width, height, top)


def continuous_trails(tracks, index, segments=()):
    """Never connect across missing observations or manual restarts."""
    result, run = [], []
    previous_segment = None
    for i in range(max(0, index-30), min(index+1, len(tracks))):
        point = tracks[i].observed_xy
        segment = segments[i] if i < len(segments) else 0
        if point is None or (previous_segment is not None and segment != previous_segment):
            if len(run) > 1:
                result.append(run)
            run = []
        if point is not None:
            run.append(point)
        previous_segment = segment
    if len(run) > 1:
        result.append(run)
    return result


def inspect_ball(video: Path, source_frames, times, request, tracks):
    index = request.get("inspection_frame", request["episode_start"])
    chosen = None
    cursor = None
    kind = "estimated_occluded" if request.get("kind") == "estimated_occluded" else "visible"
    window = "SEIMA - Recuperar bola"
    first = read_frame(video, source_frames[index])
    height, width = first.shape[:2]
    scale = min(1., 1000/width, 500/height)
    top = 100
    zoom_display = zoom_source = None
    play = False
    error = ""
    mode = "locate"
    buttons = []
    pending_action = None

    def mouse(event, x, y, flags, _):
        nonlocal chosen, cursor, kind, pending_action, play
        if event == cv2.EVENT_LBUTTONDOWN:
            for x0, y0, x1, y1, action in buttons:
                if x0 <= x < x1 and y0 <= y < y1:
                    pending_action = action
                    return
        point = inspector_click(x, y, width, height, scale, top, zoom_display, zoom_source)
        if event == cv2.EVENT_MOUSEMOVE and point is not None:
            # Keep enlargement fixed while interacting inside it.
            if zoom_display is None or not (zoom_display[0] <= x < zoom_display[0]+zoom_display[2] and zoom_display[1] <= y < zoom_display[1]+zoom_display[3]):
                cursor = point
        if event == cv2.EVENT_LBUTTONDOWN and point is not None:
            chosen = point
            play = False
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                kind = "estimated_occluded"
        if event == cv2.EVENT_RBUTTONDOWN:
            chosen = None

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    cached_index, raw = index, first
    try:
        while True:
            if index != cached_index:
                raw = read_frame(video, source_frames[index])
                cached_index = index
                chosen = None
            picture = cv2.resize(raw, None, fx=scale, fy=scale)
            canvas_width = max(1000, picture.shape[1]+270)
            canvas = np.zeros((top+max(270, picture.shape[0])+235, canvas_width, 3), np.uint8)
            canvas[top:top+picture.shape[0], :picture.shape[1]] = picture
            for trail in continuous_trails(tracks, index, request.get("trail_segments", ())):
                points = np.round(np.asarray(trail)*scale+[0, top]).astype(np.int32)
                cv2.polylines(canvas, [points], False, (60, 220, 80), 2)
            last = request.get("last_observed_frame")
            if last is not None and last < len(tracks) and tracks[last].observed_xy is not None:
                point = np.round(tracks[last].observed_xy*scale+[0, top]).astype(int)
                cv2.circle(canvas, tuple(point), 9, (0, 160, 255), 2)
            if chosen is not None:
                point = np.round(np.asarray(chosen)*scale+[0, top]).astype(int)
                cv2.drawMarker(canvas, tuple(point), (255, 70, 220), cv2.MARKER_CROSS, 22, 2)
            zoom_display = zoom_source = None
            if cursor is None:
                cursor = (width/2, height/2)
            if cursor is not None:
                zoom_source = zoom_bounds(cursor, width, height)
                sx, sy, sw, sh = zoom_source
                extent = 250
                zx, zy = picture.shape[1]+10, top+20
                zoom_display = (zx, zy, extent, extent)
                canvas[zy:zy+extent, zx:zx+extent] = cv2.resize(raw[sy:sy+sh, sx:sx+sw], (extent, extent))
                cv2.rectangle(canvas, (zx, zy), (zx+extent, zy+extent), (255, 70, 220), 2)
                cv2.putText(canvas, "Lupa: tambien puede hacer clic", (zx, top+12),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, (230, 230, 230), 1)
                if chosen is not None and sx <= chosen[0] < sx+sw and sy <= chosen[1] < sy+sh:
                    p = (int(zx+(chosen[0]-sx)*extent/sw), int(zy+(chosen[1]-sy)*extent/sh))
                    cv2.drawMarker(canvas, p, (255, 70, 220), cv2.MARKER_CROSS, 15, 1)
            reason = {"manual_recovery_timeout": "No pude recuperar la bola desde el punto anterior.",
                      "clip_ended_before_recovery": "El clip termino antes de confirmar la recuperacion."}.get(
                          request['reason'], "Necesito ayuda para localizar la bola.")
            prompt = ("1. Haga clic sobre la bola en la imagen o en la lupa." if chosen is None else
                      "2. Revise la cruz rosa. Si esta bien, pulse Confirmar y continuar.")
            if chosen is None and kind == "estimated_occluded":
                prompt = "1. Haga clic donde estima que esta la bola oculta. Puede ajustar el punto despues."
            if mode == "options":
                prompt = "Si no ve la bola, puede estimar su posicion o dejar un tramo sin seguimiento."
            elif mode == "skip":
                prompt = "Busque el ULTIMO fotograma que quiere omitir. Ese tramo quedara marcado como perdido."
            elif mode == "end":
                prompt = "Busque el ULTIMO fotograma del intercambio. El procesamiento terminara ahi."
            lines = [reason, error or prompt,
                     f"Tiempo del clip: {times[index]:.2f} s | Fotograma {index+1} de {len(times)}",
                     "Rosa: su punto manual | Naranja: ultima bola observada | Verde: trayectoria anterior"]
            for line_index, line in enumerate(lines):
                cv2.putText(canvas, line, (10, 20+line_index*23), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1)
            rows = [[("back", "Anterior frame"), ("forward", "Siguiente frame"),
                     ("earlier", "Retroceder 0,2 s"), ("later", "Avanzar 0,2 s"),
                     ("play", "Pausar" if play else "Reproducir lento")]]
            if mode == "options":
                rows.append([("estimate", "Estimar posicion"), ("skip", "Omitir un tramo"), ("end", "Terminar intercambio")])
            elif mode in ("skip", "end"):
                rows.append([("apply_"+mode, "Omitir hasta este frame" if mode == "skip" else "Terminar en este frame"),
                             ("cancel", "Volver a localizar bola")])
            elif chosen is not None:
                rows.append([("confirm", "Confirmar y continuar"), ("undo", "Corregir punto"),
                             ("type", "Visible (cambiar tipo)" if kind == "visible" else "Estimada (cambiar tipo)")])
            else:
                rows.append([("options", "No encuentro la bola"),
                             ("type", "Visible (cambiar tipo)" if kind == "visible" else "Estimada (cambiar tipo)")])
            footer = [("quit", "Guardar y salir")]
            if mode != "locate" or chosen is not None:
                footer.insert(0, ("cancel" if mode != "locate" else "options", "Volver" if mode != "locate" else "Otras opciones"))
            rows.append(footer)
            buttons = []
            bottom = canvas.shape[0]-175
            progress = request.get("progress")
            if progress:
                text = (f"Intercambio {progress['clip_number']}/{progress['clip_count']}: {progress['clip_percent']:.1f}% | "
                        f"Total recorrido: {progress['total_percent']:.1f}% | Faltan {progress['remaining_video_seconds']:.1f} s de video")
                cv2.putText(canvas, text, (10, bottom-33), cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 240, 240), 1)
                cv2.rectangle(canvas, (10, bottom-24), (canvas_width-10, bottom-16), (60, 60, 60), -1)
                cv2.rectangle(canvas, (10, bottom-24), (10+int((canvas_width-20)*progress['total_percent']/100), bottom-16), (70, 180, 80), -1)
                cv2.putText(canvas, f"Este clip: faltan {progress['clip_remaining_seconds']:.1f} s; despues, {progress['other_clips_remaining']} intercambios. Tiempo de trabajo: depende de las perdidas.",
                            (10, bottom-3), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 220, 220), 1)
            for row, choices in enumerate(rows):
                cell = canvas_width//len(choices)
                for col, (action, label) in enumerate(choices):
                    x0, y0 = col*cell+5, bottom+row*54
                    cv2.rectangle(canvas, (x0, y0), (x0+cell-10, y0+43), (65, 65, 65), -1)
                    cv2.putText(canvas, label, (x0+8, y0+27), cv2.FONT_HERSHEY_SIMPLEX, .46, (255, 255, 255), 1)
                    buttons.append((x0, y0, x0+cell-10, y0+43, action))
            cv2.imshow(window, canvas)
            delay = max(1, int(2000*(times[min(index+1, len(times)-1)]-times[index]))) if play else 30
            key = cv2.waitKeyEx(delay)
            action, pending_action = pending_action, None
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord("q")) or action == "quit":
                return {"action": "pause", "frame": index}
            if action in ("options", "skip", "end", "cancel", "estimate"):
                mode = "locate" if action in ("cancel", "estimate") else action
                play = False
                error = ""
                if action == "estimate":
                    kind = "estimated_occluded"
                    chosen = None
            if action == "type":
                kind = "estimated_occluded" if kind == "visible" else "visible"
            mapping = {"back": ord("a"), "forward": ord("d"), "earlier": ord("j"), "later": ord("l"),
                       "play": ord(" "), "confirm": 13, "undo": ord("u")}
            key = mapping.get(action, key)
            if key in (ord("v"), ord("e")):
                kind = "visible" if key == ord("v") else "estimated_occluded"
            elif key == ord("u"):
                chosen = None
            elif key == 13 and chosen is not None and mode == "locate":
                return {"action": "seed", "frame": index, "xy": list(chosen), "kind": kind,
                        "episode_start": request["episode_start"], "trigger_frame": request["trigger_frame"]}
            elif action in ("apply_skip", "apply_end"):
                if index < request["episode_start"]:
                    error = "Seleccione un frame posterior a la perdida"
                else:
                    return {"action": "skip" if action == "apply_skip" else "end", "frame": index,
                            "episode_start": request["episode_start"], "trigger_frame": request["trigger_frame"]}
            elif key in (ord("s"), ord("f")):
                mode = "skip" if key == ord("s") else "end"
                play = False
            elif key == ord(" "):
                play = not play
            elif key in (ord("a"), ord("d"), ord("j"), ord("l")):
                play = False
                if key in (ord("a"), ord("d")):
                    index += -1 if key == ord("a") else 1
                else:
                    index = int(np.searchsorted(times, times[index]+(-.2 if key == ord("j") else .2)))
            if play:
                index += 1
            index = int(np.clip(index, 0, len(times)-1))
            if index == len(times)-1:
                play = False
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
