import numpy as np

from seima_mocap.pose_context import load_body_frames


def test_loads_both_players_and_maps_right_arm_to_pixels(tmp_path):
    left = np.full((2, 33, 5), np.nan)
    right = np.full((2, 33, 5), np.nan)
    for values, x in ((left, .25), (right, .75)):
        for index in range(33):
            values[:, index] = [x, .5, 0, .9, .8]
    path = tmp_path / "pose.npz"
    np.savez_compressed(path, left_normalized=left, right_normalized=right)
    frames, source = load_body_frames(path, None, 2, 200, 100)
    assert frames[0]["left"].right_wrist_xy.tolist() == [50, 50]
    assert frames[0]["right"].right_wrist_xy.tolist() == [150, 50]
    assert source == str(path)
