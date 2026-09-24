import numpy as np

from seima_mocap.ball_tracking import (
    BallObservation,
    BallPhysicalParams,
    BallState,
    WorldBallObservation,
    associate_observations,
    fit_initial_state_world,
    predict_trajectory,
    propagate_state,
)


def test_gravity_only_matches_closed_form():
    params = BallPhysicalParams(drag_coefficient=0.0)
    initial = BallState(
        position_m=np.array([0.0, 0.0, 1.0]),
        velocity_m_s=np.array([2.0, -1.0, 3.0]),
        timestamp_s=0.0,
    )
    state = propagate_state(initial, 0.4, params)
    expected_position = np.array([
        0.8,
        -0.4,
        1.0 + 3.0 * 0.4 - 0.5 * params.gravity_m_s2 * 0.4**2,
    ])
    expected_velocity = np.array([2.0, -1.0, 3.0 - params.gravity_m_s2 * 0.4])
    np.testing.assert_allclose(state.position_m, expected_position, atol=1e-9)
    np.testing.assert_allclose(state.velocity_m_s, expected_velocity, atol=1e-9)


def test_drag_reduces_horizontal_speed():
    initial = BallState(np.array([0.0, 0.0, 1.0]), np.array([12.0, 0.0, 0.0]), 0.0)
    no_drag = propagate_state(initial, 0.2, BallPhysicalParams(drag_coefficient=0.0))
    with_drag = propagate_state(initial, 0.2, BallPhysicalParams(drag_coefficient=0.47))
    assert 0.0 < with_drag.velocity_m_s[0] < no_drag.velocity_m_s[0]
    assert with_drag.position_m[0] < no_drag.position_m[0]


def test_prediction_continues_without_observations():
    params = BallPhysicalParams(drag_coefficient=0.0)
    initial = BallState(np.array([0.0, 0.0, 0.5]), np.array([3.0, 0.0, 2.0]), 1.0)
    states = predict_trajectory(initial, [1.05, 1.10, 1.15], params)
    assert [state.timestamp_s for state in states] == [1.05, 1.10, 1.15]
    assert states[2].position_m[0] > states[1].position_m[0] > states[0].position_m[0]


def test_association_prefers_candidate_consistent_with_prediction():
    state = BallState(np.array([1.0, 2.0, 3.0]), np.zeros(3), 0.1)
    projector = lambda position: np.array([100.0 * position[0], 100.0 * position[2]])
    observations = [
        BallObservation(0.1, np.array([102.0, 301.0]), confidence=0.60),
        BallObservation(0.1, np.array([115.0, 300.0]), confidence=0.95),
        BallObservation(0.1, np.array([240.0, 300.0]), confidence=1.00),
    ]
    match = associate_observations(state, observations, projector, gate_px=60.0, sigma_px=10.0)
    assert match is not None
    np.testing.assert_allclose(match.observation.pixel_xy, [102.0, 301.0])
    assert match.residual_px < 3.0


def test_world_fit_recovers_initial_state_from_synthetic_flight():
    params = BallPhysicalParams(drag_coefficient=0.0)
    truth = BallState(
        position_m=np.array([0.25, -0.10, 0.55]),
        velocity_m_s=np.array([7.5, 1.2, 3.8]),
        timestamp_s=0.0,
    )
    times = np.linspace(0.0, 0.35, 10)
    observations = [
        WorldBallObservation(t, propagate_state(truth, float(t), params).position_m)
        for t in times
    ]
    guess = BallState(
        position_m=truth.position_m + np.array([0.08, -0.05, 0.06]),
        velocity_m_s=truth.velocity_m_s + np.array([-1.0, 0.6, -0.8]),
        timestamp_s=0.0,
    )
    result = fit_initial_state_world(observations, guess, params)
    assert result.rmse_m < 1e-5
    np.testing.assert_allclose(result.state.position_m, truth.position_m, atol=1e-4)
    np.testing.assert_allclose(result.state.velocity_m_s, truth.velocity_m_s, atol=1e-4)


def test_physical_parameters_load_from_json():
    params = BallPhysicalParams.from_json("config/ball_tracking_defaults.json")
    assert params.mass_kg == 0.0027
    assert params.radius_m == 0.02
    assert params.magnus_enabled is False
