from dataclasses import replace

import numpy as np
import pytest

from seima_mocap.stroke_guidance import StrokeGuide, StrokePriorFrame, StrokeRegion
from seima_mocap.interaction_tracking import BodyPoseFrame, InteractionContext
from seima_mocap.assisted_tracking import replay_tracking
from seima_mocap.ball_tracking import BallPhysicalParams
from seima_mocap.racketvision_adapter import RacketVisionFrame
from test_manual_recovery import candidate, scene, seed_action


def context(t, x):
    return InteractionContext(t, {"left": BodyPoseFrame("left", np.array([x, 60]), confidence=.9)})


def test_missing_pose_leaves_candidates_unchanged_and_no_region():
    prior = StrokeGuide(scene().table_polygon_xy).step(InteractionContext(0))
    source = candidate(0, 100)
    result = prior.apply([source])[0]
    assert prior.fallback == "no_pose_baseline_unchanged"
    assert prior.weight([999, 999]) == 1
    assert result.observation.confidence == source.observation.confidence
    np.testing.assert_array_equal(result.observation.pixel_xy, source.observation.pixel_xy)
    tied = [candidate(0, 150), candidate(0, 100)]
    assert [c.observation.pixel_xy[0] for c in prior.apply(tied)] == [150, 100]


def test_prior_never_invents_or_excludes_detections_and_preserves_geometry():
    region = StrokeRegion("left", "contact_hypothesis", (50, 60), (20, 20), .8, "wrist")
    prior = StrokePriorFrame(0, (region,), "soft_full_candidate_pool", .25)
    source = [candidate(0, 50), candidate(0, 900)]
    result = prior.apply(source)
    assert len(result) == 2 and prior.apply([]) == []
    assert prior.contains([50, 60]) and not prior.contains([900, 60])
    assert 0 < result[1].observation.confidence < source[1].observation.confidence
    assert result[0].observation.confidence == source[0].observation.confidence
    assert all(r.bbox_xywh == s.bbox_xywh for r, s in zip(result, source))
    assert all(r.observation.source == s.observation.source for r, s in zip(result, source))


def test_causal_phases_and_large_timestamp_gap_does_not_invent_velocity():
    guide = StrokeGuide(scene().table_polygon_xy)
    assert guide.step(context(0, 30)).regions[0].phase == "uncertain"
    assert guide.step(context(.02, 40)).regions[0].phase == "contact_hypothesis"
    assert guide.step(context(.04, 40)).regions[0].phase == "outgoing_hypothesis"
    assert guide.step(context(.5, 60)).regions[0].phase == "uncertain"
    with pytest.raises(ValueError):
        guide.step(context(.5, 61))


def test_future_pose_changes_cannot_change_prior_prefix():
    inputs = [context(0, 30), context(.02, 40), context(.04, 40)]
    a, b = StrokeGuide(scene().table_polygon_xy), StrokeGuide(scene().table_polygon_xy)
    prefix_a = [a.step(c).record() for c in inputs]
    prefix_b = [b.step(c).record() for c in inputs]
    a.step(context(.06, 90)); b.step(context(.06, 900))
    assert prefix_a == prefix_b


def test_identity_hook_reproduces_unmodified_engine_and_manual_seed_priority():
    times = np.arange(10)*.02
    evidence = [[candidate(t, 60+i*20)] for i, t in enumerate(times)]
    models = [RacketVisionFrame(i, t, (), ()) for i, t in enumerate(times)]
    contexts = [InteractionContext(t) for t in times]
    args = (times, evidence, models, contexts, scene(), BallPhysicalParams(), [seed_action()])
    baseline = replay_tracking(*args)
    modified = replay_tracking(*args, candidate_prior=lambda i, cs: cs)
    assert modified.requests == baseline.requests
    assert modified.provenance == baseline.provenance
    assert modified.tracks[0].state is None  # no velocity invented at manual seed
    for a, b in zip(baseline.tracks, modified.tracks):
        assert a.status == b.status
        if a.observed_xy is not None:
            np.testing.assert_array_equal(a.observed_xy, b.observed_xy)
