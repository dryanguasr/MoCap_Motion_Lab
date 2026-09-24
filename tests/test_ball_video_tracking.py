import cv2
import numpy as np

from seima_mocap.ball_detection import BallCandidate
from seima_mocap.ball_tracking import BallObservation, BallPhysicalParams
from seima_mocap.ball_video_tracking import PlanarSceneCalibration, ProjectedBallTracker, TrackStatus


def candidate(t, xy, confidence=.9):
    obs = BallObservation(t, np.array(xy, float), confidence, size_px=7)
    return BallCandidate(obs, (int(xy[0]), int(xy[1]), 7, 7), 30, .8, 1, 230, 10, 100, 80, 80,
                         {"motion": .8, "bright_unsaturated": .8, "contrast": .8})


def calibration():
    return PlanarSceneCalibration(np.array([[20, 100], [294, 100], [300, 130], [14, 130]], float),
                                  pixels_per_m_vertical=100)


def test_planar_mapping_round_trip_and_table_proximity():
    scene = calibration()
    pixel = np.array([120., 70.])
    np.testing.assert_allclose(scene.project(scene.image_to_plane(pixel)), pixel)
    assert scene.close_to_table_surface([150, 110])
    assert not scene.close_to_table_surface([150, 20], margin_px=10)


def test_requires_three_coherent_observations_then_predicts_and_loses():
    tracker = ProjectedBallTracker(calibration(), BallPhysicalParams(drag_coefficient=0), max_prediction_frames=2)
    statuses = []
    for i, xy in enumerate(((60, 70), (80, 68), (100, 67))):
        statuses.append(tracker.step(i / 30, [candidate(i / 30, xy)]).status)
    assert statuses == [TrackStatus.UNINITIALIZED, TrackStatus.UNINITIALIZED, TrackStatus.OBSERVED]
    assert tracker.step(3 / 30, []).status == TrackStatus.PREDICTED
    assert tracker.step(4 / 30, []).status == TrackStatus.PREDICTED
    assert tracker.step(5 / 30, []).status == TrackStatus.LOST


def test_physical_and_constant_velocity_predictions_differ():
    scene = calibration()
    physical = ProjectedBallTracker(scene, BallPhysicalParams(drag_coefficient=0), use_physics=True)
    temporal = ProjectedBallTracker(scene, BallPhysicalParams(drag_coefficient=0), use_physics=False)
    for tracker in (physical, temporal):
        for i, xy in enumerate(((60, 70), (80, 68), (100, 67))):
            tracker.step(i / 30, [candidate(i / 30, xy)])
    p = physical.step(3 / 30, []).predicted_xy
    c = temporal.step(3 / 30, []).predicted_xy
    assert p[1] > c[1]  # gravity moves the image prediction down


def test_explicit_discontinuity_does_not_continue_old_prediction():
    tracker = ProjectedBallTracker(calibration(), BallPhysicalParams(drag_coefficient=0))
    for i, xy in enumerate(((60, 70), (80, 68), (100, 67))):
        tracker.step(i / 30, [candidate(i / 30, xy)])
    assert tracker.state is not None
    tracker.reset_for_discontinuity()
    assert tracker.state is None
    result = tracker.step(3 / 30, [candidate(3 / 30, (130, 75))])
    assert result.status == TrackStatus.LOST
    assert result.predicted_xy is None
