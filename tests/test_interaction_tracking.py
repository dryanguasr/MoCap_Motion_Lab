import numpy as np

from seima_mocap.ball_detection import BallCandidate
from seima_mocap.ball_tracking import BallObservation
from seima_mocap.ball_video_tracking import PlanarSceneCalibration, TrackFrame, TrackStatus
from seima_mocap.interaction_tracking import (
    BodyPoseFrame,
    InteractionContext,
    InteractionMode,
    RacketPoseObservation,
    RacketTrackFrame,
    associate_rackets_to_players,
    fuse_ball_candidates,
    infer_interaction_mode,
    interaction_metrics,
    smooth_event_gaps,
)


def scene():
    return PlanarSceneCalibration(np.array([[20, 100], [294, 100], [300, 130], [14, 130]], float),
                                  pixels_per_m_vertical=100)


def racket(t, center, confidence=.9):
    cx, cy = center
    points = np.array([[cx, cy - 20], [cx, cy + 15], [cx, cy + 55],
                       [cx - 15, cy], [cx + 15, cy]], float)
    return RacketPoseObservation(t, [cx - 20, cy - 25, cx + 20, cy + 60], points,
                                 np.full(5, confidence), confidence)


def body(player, wrist, bbox):
    wrist = np.asarray(wrist, float)
    return BodyPoseFrame(player, wrist, wrist + [0, 45], wrist + [0, 90], bbox, .9, "test")


def candidate(t, xy, confidence=.8, source="classical"):
    observation = BallObservation(t, np.asarray(xy, float), confidence, 7, source=source)
    return BallCandidate(observation, (int(xy[0]), int(xy[1]), 7, 7), 30, .8, 1,
                         230, 10, 100, 80, 80, {"motion": .8, "contrast": .8})


def test_two_rackets_are_globally_associated_to_nearest_right_wrists():
    bodies = {"left": body("left", (80, 140), [10, 20, 150, 220]),
              "right": body("right", (260, 140), [190, 20, 319, 220])}
    observations = [racket(0, (250, 100)), racket(0, (90, 100))]
    matches = associate_rackets_to_players(observations, bodies, None, 320)
    assert matches["left"][0].head_center_xy[0] < 160
    assert matches["right"][0].head_center_xy[0] > 160


def test_body_suppression_is_disabled_next_to_racket():
    bodies = {"left": body("left", (80, 140), [20, 20, 300, 240])}
    racket_track = RacketTrackFrame("left", 0, TrackStatus.OBSERVED, racket(0, (85, 95)), confidence=.9)
    context = InteractionContext(0, bodies, {"left": racket_track})
    near = fuse_ball_candidates([candidate(0, (85, 95))], [], context)[0]
    far = fuse_ball_candidates([candidate(0, (260, 220))], [], context)[0]
    assert near.reason_scores["body_suppression_factor"] == 1
    assert far.reason_scores["body_suppression_factor"] < 1


def test_model_and_classical_candidates_merge_and_keep_provenance():
    context = InteractionContext(0)
    model = BallObservation(0, np.array([103., 99.]), .75, 8, source="racketvision_balltrack")
    fused = fuse_ball_candidates([candidate(0, (100, 100))], [model], context)
    assert len(fused) == 1
    assert "racketvision_balltrack" in fused[0].observation.source
    assert fused[0].reason_scores["source_agreement"] > .9


def test_closing_racket_opens_contact_pending_before_collision():
    track = RacketTrackFrame("left", 0, TrackStatus.OBSERVED, racket(0, (100, 100)),
                             np.array([200., 0.]), .9)
    context = InteractionContext(0, rackets={"left": track})
    metrics = interaction_metrics(np.array([160., 100.]), np.array([-400., 0.]), context)
    assert metrics[0].closing_speed_px_s > 0
    mode, _ = infer_interaction_mode(np.array([160., 100.]), np.array([-400., 0.]), context, scene())
    assert mode == InteractionMode.CONTACT_PENDING


def test_short_bounce_occlusion_is_smoothed_but_remains_predicted():
    frames = [
        TrackFrame(0, TrackStatus.OBSERVED, np.array([100., 90.]), None, None, .9, 0),
        TrackFrame(1 / 30, TrackStatus.LOST, None, np.array([120., 105.]), None, 0, 1),
        TrackFrame(2 / 30, TrackStatus.OBSERVED, np.array([140., 92.]), None, None, .9, 0),
    ]
    modes = [InteractionMode.BOUNCE_PENDING] * 3
    smoothed, corrections = smooth_event_gaps(frames, modes, scene())
    assert smoothed[1].status == TrackStatus.PREDICTED
    assert smoothed[1].observed_xy is None
    assert smoothed[1].predicted_xy is not None
    assert corrections[0]["method"] == "piecewise_quadratic_bounce"
