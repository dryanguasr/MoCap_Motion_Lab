import json
import importlib.util
from pathlib import Path

import numpy as np

from seima_mocap import guided_review as review


def fixture_manifest():
    return {"source": {"path": "source.mp4", "frame_count": 40, "pts_s": (np.arange(40)*.02).tolist()},
            "rallies": [{"id": "r1", "start_frame": 3, "end_frame": 35, "accepted": False}],
            "shots": [{"id": "s1", "start_frame": 0, "end_frame": 20, "reviewed": True, "geometry": None},
                      {"id": "s2", "start_frame": 20, "end_frame": 40, "reviewed": True, "geometry": None}],
            "clips": []}


def test_pending_skips_resolved_decisions_and_unused_geometry():
    value = fixture_manifest()
    assert review.pending_checks(value) == [("camera", "s2"), ("rally", "r1")]
    value["shots"][1]["boundary_confirmed"] = True
    value["rallies"][0]["accepted"] = True
    value["shots"][0]["reviewed"] = False
    assert review.pending_checks(value) == [("geometry", "s1")]
    value["rallies"] = []
    assert review.pending_checks(value) == []


def test_camera_confirmation_is_saved_and_not_repeated_on_resume(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(fixture_manifest()))
    monkeypatch.setattr(review, "validate_manifest", lambda *a, **kw: None)
    answers = iter([("yes", 20), ("set", 20), ("quit", 10)])
    monkeypatch.setattr(review, "ask", lambda *a, **kw: next(answers))
    assert review.review_manifest(path, tmp_path, None) is False
    value = json.loads(path.read_text())
    assert value["shots"][1]["boundary_confirmed"]
    assert value["guided_cursor"] == {"id": "r1", "frame": 10}
    calls = []

    def resumed(*args, **kwargs):
        calls.append((args[4], kwargs["frame"]))
        return "quit", 10

    monkeypatch.setattr(review, "ask", resumed)
    review.review_manifest(path, tmp_path, None)
    assert calls == [("Intercambio pendiente: r1", 10)]


def test_closing_mid_adjustment_keeps_original_bounds(tmp_path, monkeypatch):
    value = fixture_manifest()
    value["shots"][1]["boundary_confirmed"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    monkeypatch.setattr(review, "validate_manifest", lambda *a, **kw: None)
    answers = iter([("adjust", 3), ("set", 1), ("quit", 38)])
    monkeypatch.setattr(review, "ask", lambda *a, **kw: next(answers))
    assert review.review_manifest(path, tmp_path, None) is False
    assert json.loads(path.read_text())["rallies"] == value["rallies"]


def test_prompt_mouse_selects_button_and_original_frame(monkeypatch):
    cv = review.cv2
    callback = {}
    monkeypatch.setattr(review, "read_frame", lambda *args: np.zeros((100, 200, 3), np.uint8))
    monkeypatch.setattr(cv, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(cv, "imshow", lambda *a: None)
    monkeypatch.setattr(cv, "getWindowProperty", lambda *a: 1)
    monkeypatch.setattr(cv, "setMouseCallback", lambda w, fn: callback.update(fn=fn))

    def click(delay):
        callback["fn"](cv.EVENT_LBUTTONDOWN, 100, 300, 0, None)
        return -1

    monkeypatch.setattr(cv, "waitKeyEx", click)
    assert review.ask(Path("test.mp4"), np.arange(20)*.02, 5, 10, "Title", "Instruction",
                      [("yes", "Si")], frame=7) == ("yes", 7)


def test_geometry_resumes_after_last_confirmed_image(tmp_path, monkeypatch):
    from seima_mocap import video_io
    from seima_mocap.table_geometry import TABLE_POINT_ORDER, NET_POINT_ORDER
    script = Path(__file__).resolve().parents[1] / "scripts/annotate_table_geometry.py"
    spec = importlib.util.spec_from_file_location("guided_geometry_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"test source identity")
    img = np.zeros((180, 320, 3), np.uint8)
    monkeypatch.setattr(module, "median_frame", lambda *a, **kw: (img, [0, 2, 4]))
    monkeypatch.setattr(video_io, "read_frame", lambda *a: img)
    monkeypatch.setattr(video_io, "probe_video", lambda *a: {"pts_s": [0, .02, .04, .06, .08]})
    for name in ("namedWindow", "setMouseCallback", "imshow", "destroyWindow"):
        monkeypatch.setattr(module.cv2, name, lambda *a: None)
    monkeypatch.setattr(module.cv2, "getWindowProperty", lambda *a: 1)
    initial = {"table_surface": {"points": [{"name": n, "xy": xy} for n, xy in zip(TABLE_POINT_ORDER,
               [[60, 100], [260, 100], [280, 150], [40, 150]])]},
               "net": {"points": [{"name": n, "xy": xy} for n, xy in zip(NET_POINT_ORDER,
               [[80, 90], [240, 90], [240, 115], [80, 115]])]}}
    answers = iter([("yes", 0), ("quit", 2)])
    monkeypatch.setattr(review, "ask", lambda *a, **kw: next(answers))
    kwargs = dict(frame_range=(0, 5), initial_geometry=initial, guided=True)
    assert not module.annotate(video, tmp_path, tmp_path, **kwargs)
    draft = tmp_path / "source__review_draft.json"
    assert json.loads(draft.read_text())["checked"] == [0]
    visited = []

    def accept(*args, **kwargs):
        visited.append(args[2])
        return "yes", args[2]

    monkeypatch.setattr(review, "ask", accept)
    assert module.annotate(video, tmp_path, tmp_path, **kwargs)
    assert visited == [2, 4]
    assert not draft.exists()
