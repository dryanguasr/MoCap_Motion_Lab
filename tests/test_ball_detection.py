import cv2
import numpy as np

from seima_mocap.ball_detection import BallDetectorConfig, detect_ball_candidates


def blank():
    frame = np.full((180, 320, 3), 45, np.uint8)
    cv2.circle(frame, (270, 30), 8, (245, 245, 245), -1)  # static distractor
    return frame


def test_detects_moving_ball_and_rejects_static_bright_blob():
    frames = [blank() for _ in range(3)]
    for frame, center in zip(frames, ((90, 90), (110, 86), (130, 84))):
        cv2.circle(frame, center, 5, (235, 235, 235), -1)
    candidates = detect_ball_candidates(*frames, 0.1, BallDetectorConfig())
    assert candidates
    best = min(candidates, key=lambda c: np.linalg.norm(c.observation.pixel_xy - [110, 86]))
    assert np.linalg.norm(best.observation.pixel_xy - [110, 86]) < 3
    assert all(np.linalg.norm(c.observation.pixel_xy - [270, 30]) > 10 for c in candidates)
    assert "motion" in best.reason_scores


def test_detects_motion_blur_and_reports_vector():
    frames = [blank() for _ in range(3)]
    for frame, ends in zip(frames, (((60, 100), (76, 100)), ((90, 100), (120, 100)), ((135, 100), (151, 100)))):
        cv2.line(frame, *ends, (240, 240, 240), 4)
    candidates = detect_ball_candidates(*frames, 0.1, BallDetectorConfig())
    best = min(candidates, key=lambda c: np.linalg.norm(c.observation.pixel_xy - [105, 100]))
    assert best.elongation > 2
    assert best.observation.blur_vector_px is not None
    assert abs(best.observation.blur_vector_px[0]) > abs(best.observation.blur_vector_px[1])


def test_roi_and_prediction_are_soft_evidence():
    frames = [blank() for _ in range(3)]
    for frame, center in zip(frames, ((90, 90), (110, 86), (130, 84))):
        cv2.circle(frame, center, 5, (235, 235, 235), -1)
    outside = detect_ball_candidates(*frames, 0.1, BallDetectorConfig(),
                                     roi_polygon_xy=((150, 0), (319, 0), (319, 179), (150, 179)))
    assert all(c.observation.pixel_xy[0] >= 150 for c in outside)
    candidates = detect_ball_candidates(*frames, 0.1, BallDetectorConfig(), predicted_pixel_xy=np.array([110, 86]))
    best = min(candidates, key=lambda c: np.linalg.norm(c.observation.pixel_xy - [110, 86]))
    assert best.reason_scores["prediction_proximity"] > 0.99
