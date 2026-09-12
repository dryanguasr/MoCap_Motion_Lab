import numpy as np
import pytest

from seima_mocap import ball_inspector as ui


def run_ui(monkeypatch, steps, *, episode=0):
    state = {"labels": {}}
    original_text = ui.cv2.putText
    monkeypatch.setattr(ui, "read_frame", lambda *args: np.zeros((300, 500, 3), np.uint8))
    for name in ("namedWindow", "destroyWindow", "imshow"):
        monkeypatch.setattr(ui.cv2, name, lambda *args: None)
    monkeypatch.setattr(ui.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(ui.cv2, "setMouseCallback", lambda name, fn: state.update(mouse=fn))

    def text(image, label, position, *args):
        state["labels"][label] = position
        return original_text(image, label, position, *args)

    monkeypatch.setattr(ui.cv2, "putText", text)
    actions = iter(steps)

    def wait(delay):
        step = next(actions)
        if isinstance(step, tuple):
            x, y = step
        else:
            assert step in state["labels"], f"Button not visible: {step}"
            x, y = state["labels"][step]
            x, y = x+3, y-8
        state["mouse"](ui.cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        state["labels"].clear()
        return -1

    monkeypatch.setattr(ui.cv2, "waitKeyEx", wait)
    return ui.inspect_ball(None, [50, 51, 52], np.array([0, .02, .04]),
                           {"episode_start": episode, "trigger_frame": 2, "reason": "lost"}, [])


def test_estimated_seed_entirely_with_mouse(monkeypatch):
    result = run_ui(monkeypatch, ["No encuentro la bola", "Estimar posicion", (100, 160), "Confirmar y continuar"])
    assert result["kind"] == "estimated_occluded"
    assert result["xy"] == [100, 60]
    assert result["action"] == "seed"


@pytest.mark.parametrize("option,confirm,expected", [
    ("Omitir un tramo", "Omitir hasta este frame", "skip"),
    ("Terminar intercambio", "Terminar en este frame", "end")])
def test_skip_end_require_endpoint_confirmation(monkeypatch, option, confirm, expected):
    result = run_ui(monkeypatch, ["No encuentro la bola", option, "Siguiente frame", confirm])
    assert result["action"] == expected
    assert result["frame"] == 1


def test_navigation_discards_point_and_pause_does_not_confirm_it(monkeypatch):
    result = run_ui(monkeypatch, [(100, 160), "Siguiente frame", "No encuentro la bola", "Guardar y salir"])
    assert result == {"action": "pause", "frame": 1}


def test_click_pauses_playback_before_confirming(monkeypatch):
    result = run_ui(monkeypatch, ["Reproducir lento", (100, 160), "Confirmar y continuar"])
    assert result["frame"] == 1
    assert result["xy"] == [100, 60]


def test_trail_does_not_bridge_losses_or_manual_seeds():
    from types import SimpleNamespace
    tracks = [SimpleNamespace(observed_xy=p) for p in [(1, 1), (2, 2), None, (5, 5), (6, 6), (100, 100), (101, 101)]]
    assert ui.continuous_trails(tracks, 6, [0, 0, 0, 0, 0, 1, 1]) == [
        [(1, 1), (2, 2)], [(5, 5), (6, 6)], [(100, 100), (101, 101)]]
