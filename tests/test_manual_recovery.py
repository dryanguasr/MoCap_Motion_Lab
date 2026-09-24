import json

import numpy as np
import pytest

from seima_mocap.assisted_tracking import replay_tracking, smooth_replay
from seima_mocap.ball_detection import BallCandidate
from seima_mocap.ball_tracking import BallObservation, BallPhysicalParams
from seima_mocap.ball_video_tracking import PlanarSceneCalibration, ProjectedBallTracker, TrackFrame, TrackStatus
from seima_mocap.interaction_tracking import InteractionAwareBallTracker, InteractionContext, InteractionMode
from seima_mocap.manual_recovery import InterventionJournal, LossEpisodes, ManualRecovery, ManualSeed
from seima_mocap.ball_inspector import inspector_click, zoom_bounds
from seima_mocap.racketvision_adapter import RacketVisionFrame


def scene():
    return PlanarSceneCalibration(np.array([[20, 100], [294, 100], [300, 130], [14, 130]], float), pixels_per_m_vertical=100)


def candidate(t, x, y=60, confidence=.9):
    return BallCandidate(BallObservation(t, np.array([x, y], float), confidence, size_px=7),
                         (int(x), int(y), 7, 7), 30, .8, 1, 230, 10, 100, 80, 80,
                         {"motion": .8, "bright_unsaturated": .8, "contrast": .8})


def recovery():
    return ManualRecovery(InteractionAwareBallTracker(ProjectedBallTracker(scene(), BallPhysicalParams()), scene()))


@pytest.mark.parametrize("kind", ["visible", "estimated_occluded"])
def test_seed_has_no_velocity_and_needs_two_subsequent_visual_detections(kind):
    recover = recovery()
    recover.begin(ManualSeed(0, 0., (60., 60.), kind))
    initial = recover.step(0, [candidate(0, 60)])
    assert initial.observed_xy is None and initial.state is None
    assert recover.step(.02, [candidate(.02, 80)]).state is None
    confirmed = recover.step(.04, [candidate(.04, 100)])
    assert recover.state == "confirmed" and confirmed.status == TrackStatus.OBSERVED
    np.testing.assert_allclose(confirmed.observed_xy, [100, 60])
    assert confirmed.state.velocity_m_s[0] > 0


def test_false_click_does_not_initialize_on_unrelated_distractor():
    recover = recovery()
    recover.begin(ManualSeed(0, 0., (20., 20.)))
    for i in range(1, 13):
        tracked = recover.step(i*.02, [candidate(i*.02, 400+i*20, 250)])
        assert tracked.observed_xy is None
    assert recover.step(.26, []).status == TrackStatus.LOST
    assert recover.state == "failed"


def test_occluded_seed_allows_larger_position_uncertainty():
    statuses = []
    for kind in ("visible", "estimated_occluded"):
        recover = recovery()
        recover.begin(ManualSeed(0, 0, (60., 60.), kind))
        recover.step(.02, [candidate(.02, 80, 120)])
        recover.step(.04, [candidate(.04, 100, 120)])
        statuses.append(recover.state)
    assert statuses == ["pending", "confirmed"]


def test_guided_detector_keeps_present_blurred_ball_when_exposure_is_duplicated():
    import cv2
    from seima_mocap.ball_detection import BallDetectorConfig, detect_ball_candidates
    background = np.zeros((180, 320), np.uint8)
    current = np.zeros((180, 320, 3), np.uint8)
    cv2.ellipse(current, (140, 70), (60, 5), 0, 0, 360, (230, 230, 230), -1)
    next_frame = np.zeros_like(current)
    cv2.ellipse(next_frame, (220, 100), (60, 5), 0, 0, 360, (230, 230, 230), -1)
    config = BallDetectorConfig(max_dimension_px=260, max_area_px=6000)
    standard = detect_ball_candidates(current.copy(), current, next_frame, .02, config, background_gray=background)
    guided = detect_ball_candidates(current.copy(), current, next_frame, .02, config, background_gray=background, manual_recovery=True)
    assert not standard
    assert guided and np.linalg.norm(guided[0].observation.pixel_xy-[140, 70]) < 2
    assert guided[0].observation.source == "manual_guided_visual_blob"


def test_loss_episode_starts_at_first_missing_but_waits_until_lost():
    times = np.arange(40)*.02
    monitor = LossEpisodes(times)
    observed = TrackFrame(0, TrackStatus.OBSERVED, np.array([10, 10]), None, None, .9, 0)
    assert monitor.step(0, observed) is None
    predicted = TrackFrame(.02, TrackStatus.PREDICTED, None, np.array([20, 20]), None, .5, 1)
    assert monitor.step(1, predicted) is None
    lost = TrackFrame(.04, TrackStatus.LOST, None, None, None, 0, 4)
    request = monitor.step(2, lost)
    assert request["episode_start"] == 1 and request["trigger_frame"] == 2
    assert monitor.step(3, lost) is None
    monitor.step(4, observed)
    assert monitor.step(5, lost)["episode_start"] == 5


def test_uninitialized_timeout_and_reset_after_skip():
    times = np.arange(60)*.02
    monitor = LossEpisodes(times, start_frame=20)
    track = TrackFrame(0, TrackStatus.UNINITIALIZED, None, None, None, 0, 0)
    assert all(monitor.step(i, track) is None for i in range(20, 45))
    assert monitor.step(46, track)["episode_start"] == 20


def journal(tmp_path):
    return InterventionJournal(tmp_path / "manual.json", {"source": "hash", "clip": "one"},
                               [0., .1, .2, .3], [50, 51, 52, 53], [1., 1.1, 1.2, 1.3], 320, 180)


def test_journal_replay_identity_and_transactional_edit_history(tmp_path):
    log = journal(tmp_path)
    action = {"action": "seed", "frame": 2, "episode_start": 1, "trigger_frame": 3,
              "kind": "visible", "xy": [40., 50.]}
    log.append(action)
    reloaded = journal(tmp_path)
    assert reloaded.actions == log.actions
    assert reloaded.actions[0]["source_frame"] == 52
    reloaded.append({**action, "frame": 1})
    assert len(reloaded.actions) == 1 and len(reloaded.payload["history"][-1]["superseded"]) == 1
    with pytest.raises(ValueError, match="different source"):
        InterventionJournal(log.path, {"source": "other"}, [], [], [], 320, 180)


def test_journal_rejects_tampered_source_mapping(tmp_path):
    log = journal(tmp_path)
    log.append({"action": "skip", "frame": 2, "episode_start": 0, "trigger_frame": 1})
    data = json.loads(log.path.read_text())
    data["actions"][0]["source_frame"] = 999
    log.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mapping"):
        journal(tmp_path)


def test_scaled_and_zoom_clicks_map_to_source_and_ignore_toolbar():
    assert inspector_click(50, 130, 1000, 600, .5, 100) == (100., 60.)
    assert inspector_click(50, 30, 1000, 600, .5, 100) is None
    assert inspector_click(425, 325, 1000, 600, .5, 100, (400, 300, 100, 100), (100, 200, 50, 50)) == (112.5, 212.5)
    assert zoom_bounds((999, 599), 1000, 600) == (910, 510, 90, 90)


def inputs(n=20):
    times = np.arange(n)*.02
    return times, [[] for _ in times], [RacketVisionFrame(i, t, (), ()) for i, t in enumerate(times)], [InteractionContext(t) for t in times]


def seed_action(frame=0, xy=(60., 60.), trigger=0):
    return {"id": "manual1", "action": "seed", "frame": frame, "episode_start": 0,
            "trigger_frame": trigger, "kind": "visible", "xy": xy}


def test_manual_priority_survives_conflicting_classic_anchor_and_replay_is_identical():
    times, evidence, models, contexts = inputs(7)
    for i in range(7):
        evidence[i] = [candidate(times[i], 500+i*20, confidence=.99), candidate(times[i], 60+i*20, confidence=.8)]
    first = replay_tracking(times, evidence, models, contexts, scene(), BallPhysicalParams(), [seed_action()])
    second = replay_tracking(times, evidence, models, contexts, scene(), BallPhysicalParams(), [seed_action()])
    assert not first.requests
    assert first.tracks[0].state is None
    assert first.tracks[2].observed_xy[0] == 100
    assert first.tracks[3].observed_xy[0] < 300
    assert [t.status for t in first.tracks] == [t.status for t in second.tracks]
    assert first.provenance == second.provenance
    for a, b in zip(first.tracks, second.tracks):
        if a.observed_xy is not None:
            np.testing.assert_array_equal(a.observed_xy, b.observed_xy)


def test_recovery_timeout_is_not_silenced_by_replayed_old_trigger():
    times, evidence, models, contexts = inputs(50)
    result = replay_tracking(times, evidence, models, contexts, scene(), BallPhysicalParams(), [seed_action(trigger=40)])
    assert result.requests[0]["reason"] == "manual_recovery_timeout"
    assert len(result.tracks) == 14


def test_future_seed_keeps_omitted_gap_and_smoothing_cannot_overwrite_it():
    times, evidence, models, contexts = inputs(30)
    for i in range(11, 30):
        evidence[i] = [candidate(times[i], 60+(i-10)*20)]
    result = replay_tracking(times, evidence, models, contexts, scene(), BallPhysicalParams(), [seed_action(10, trigger=5)])
    smoothed = smooth_replay(result, scene())
    assert all(t.status == TrackStatus.LOST for t in smoothed.tracks[:11])
    assert all(t.observed_xy is None for t in smoothed.tracks[:11])
    assert result.provenance[10]["manual_seed_kind"] == "visible"
    assert result.tracks[12].status == TrackStatus.OBSERVED


def test_end_action_stops_clip_and_skip_does_not_repeat_resolved_loss():
    times, evidence, models, contexts = inputs(60)
    skip = {"id": "skip", "action": "skip", "frame": 35, "episode_start": 0, "trigger_frame": 25}
    result = replay_tracking(times, evidence, models, contexts, scene(), BallPhysicalParams(), [skip])
    assert not result.requests  # less than .5 s remains after skip
    end = {**skip, "action": "end"}
    result = replay_tracking(times, evidence, models, contexts, scene(), BallPhysicalParams(), [end])
    assert result.ended and len(result.tracks) == 36


def test_seed_near_last_frame_reports_unconfirmed_recovery():
    times, evidence, models, contexts = inputs(6)
    result = replay_tracking(times, evidence, models, contexts, scene(), BallPhysicalParams(), [seed_action(5, trigger=4)])
    assert result.requests[0]["reason"] == "clip_ended_before_recovery"


def test_inspector_navigation_kind_and_click_confirmation_without_desktop(monkeypatch):
    import cv2
    import seima_mocap.ball_inspector as ui
    monkeypatch.setattr(ui, "read_frame", lambda *args: np.zeros((300, 500, 3), np.uint8))
    callback = {}
    for name in ("namedWindow", "imshow", "destroyWindow"):
        monkeypatch.setattr(ui.cv2, name, lambda *args: None)
    monkeypatch.setattr(ui.cv2, "getWindowProperty", lambda *args: 1.)
    monkeypatch.setattr(ui.cv2, "setMouseCallback", lambda name, fn: callback.update(mouse=fn))
    keys = iter([ord("d"), ord("e"), 13])
    def wait(_):
        key = next(keys)
        if key == 13:
            callback["mouse"](cv2.EVENT_LBUTTONDOWN, 100, 160, 0, None)
        return key
    monkeypatch.setattr(ui.cv2, "waitKeyEx", wait)
    action = ui.inspect_ball(None, [50, 51, 52], np.array([0., .02, .04]),
                             {"episode_start": 0, "trigger_frame": 2, "reason": "lost"}, [])
    assert action["frame"] == 1 and action["xy"] == [100., 60.]
    assert action["kind"] == "estimated_occluded"


def test_inspector_close_returns_pause_and_does_not_invent_a_seed(monkeypatch):
    import seima_mocap.ball_inspector as ui
    monkeypatch.setattr(ui, "read_frame", lambda *args: np.zeros((300, 500, 3), np.uint8))
    for name in ("namedWindow", "imshow", "destroyWindow", "setMouseCallback"):
        monkeypatch.setattr(ui.cv2, name, lambda *args: None)
    monkeypatch.setattr(ui.cv2, "waitKeyEx", lambda *args: -1)
    monkeypatch.setattr(ui.cv2, "getWindowProperty", lambda *args: 0.)
    action = ui.inspect_ball(None, [50, 51, 52], np.array([0., .02, .04]),
                             {"episode_start": 0, "trigger_frame": 2, "reason": "lost"}, [])
    assert action == {"action": "pause", "frame": 0}
