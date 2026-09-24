from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "process_two_player_video.py"
    spec = importlib.util.spec_from_file_location("seima_two_player_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_side_selection_prefers_foreground_player_over_background_bystander():
    module = _module()
    track = module.Track("left", center=np.array([0.12, 0.36]), area=0.02)
    background = module.Detection(0, np.array([0.10, 0.34]), 0.009, 0.98, 0.99, 0.98, 1.0, 0.49)
    athlete = module.Detection(1, np.array([0.24, 0.62]), 0.16, 0.86, 0.96, 0.84, 0.95, 0.94)

    selected = module.select_side_detection([background, athlete], track)

    assert selected is athlete
