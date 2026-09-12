"""Physics-informed state propagation and observation association for table-tennis balls.

This module is intentionally independent of OpenCV and MediaPipe. It provides:
- a 3-D ball state;
- gravity + quadratic drag + optional Magnus acceleration;
- RK4 propagation with constant spin over short flight windows;
- image-space observation structures and prediction gating;
- robust fitting of an initial state from metric 3-D observations.

The model is exploratory. Spin inferred only from trajectory should be treated as
an effective parameter unless identifiability is demonstrated independently.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence
import json
import math

import numpy as np


Array = np.ndarray
Projector = Callable[[Array], Array]


@dataclass(frozen=True)
class BallPhysicalParams:
    """Physical/empirical parameters for free-flight propagation.

    Coordinate convention: z is vertical and positive upwards.
    ``magnus_enabled`` is False by default. When enabled, the lift coefficient is
    computed from the spin parameter r*|omega_perp|/|v| and clipped explicitly.
    """

    mass_kg: float = 0.0027
    radius_m: float = 0.020
    air_density_kg_m3: float = 1.225
    drag_coefficient: float = 0.47
    gravity_m_s2: float = 9.81
    magnus_enabled: bool = False
    magnus_lift_slope: float = 1.0
    max_lift_coefficient: float = 0.6

    def __post_init__(self) -> None:
        if self.mass_kg <= 0 or self.radius_m <= 0 or self.air_density_kg_m3 <= 0:
            raise ValueError("mass, radius and air density must be positive")
        if self.drag_coefficient < 0 or self.gravity_m_s2 <= 0:
            raise ValueError("drag must be non-negative and gravity positive")
        if self.magnus_lift_slope < 0 or self.max_lift_coefficient < 0:
            raise ValueError("Magnus coefficients must be non-negative")

    @property
    def cross_section_m2(self) -> float:
        return math.pi * self.radius_m**2

    @classmethod
    def from_json(cls, path: str | Path) -> "BallPhysicalParams":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


@dataclass(frozen=True)
class BallState:
    position_m: Array
    velocity_m_s: Array
    timestamp_s: float
    spin_rad_s: Array | None = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=float)
        velocity = np.asarray(self.velocity_m_s, dtype=float)
        spin = np.zeros(3, dtype=float) if self.spin_rad_s is None else np.asarray(self.spin_rad_s, dtype=float)
        if position.shape != (3,) or velocity.shape != (3,) or spin.shape != (3,):
            raise ValueError("position, velocity and spin must be 3-vectors")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(spin)):
            raise ValueError("ball state must contain finite values")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "velocity_m_s", velocity)
        object.__setattr__(self, "spin_rad_s", spin)
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))


@dataclass(frozen=True)
class BallObservation:
    """A visual candidate in image coordinates.

    ``pixel_xy`` is an observation. It must never be rendered or stored as if it
    were a physically predicted position. ``blur_vector_px`` can encode streak
    orientation/length when a detector can estimate it.
    """

    timestamp_s: float
    pixel_xy: Array
    confidence: float
    size_px: float | None = None
    blur_vector_px: Array | None = None
    source: str = "vision"

    def __post_init__(self) -> None:
        pixel = np.asarray(self.pixel_xy, dtype=float)
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError("pixel_xy must be a finite 2-vector")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        blur = None if self.blur_vector_px is None else np.asarray(self.blur_vector_px, dtype=float)
        if blur is not None and (blur.shape != (2,) or not np.all(np.isfinite(blur))):
            raise ValueError("blur_vector_px must be a finite 2-vector")
        object.__setattr__(self, "pixel_xy", pixel)
        object.__setattr__(self, "blur_vector_px", blur)
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))


@dataclass(frozen=True)
class CandidateMatch:
    observation: BallObservation
    predicted_pixel_xy: Array
    residual_px: float
    score: float


@dataclass(frozen=True)
class WorldBallObservation:
    timestamp_s: float
    position_m: Array
    confidence: float = 1.0

    def __post_init__(self) -> None:
        p = np.asarray(self.position_m, dtype=float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("position_m must be a finite 3-vector")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be in (0,1]")
        object.__setattr__(self, "position_m", p)
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))


@dataclass(frozen=True)
class StateFitResult:
    state: BallState
    params: BallPhysicalParams
    rmse_m: float
    iterations: int
    converged: bool


def _acceleration(velocity: Array, spin: Array, params: BallPhysicalParams) -> Array:
    speed = float(np.linalg.norm(velocity))
    gravity = np.array([0.0, 0.0, -params.gravity_m_s2])
    if speed < 1e-12:
        return gravity

    dynamic = 0.5 * params.air_density_kg_m3 * params.cross_section_m2 * speed
    drag = -(dynamic * params.drag_coefficient / params.mass_kg) * velocity
    acceleration = gravity + drag

    if params.magnus_enabled and np.linalg.norm(spin) > 1e-12:
        cross = np.cross(spin, velocity)
        cross_norm = float(np.linalg.norm(cross))
        if cross_norm > 1e-12:
            omega_parallel = np.dot(spin, velocity) / max(speed**2, 1e-12) * velocity
            omega_perp = spin - omega_parallel
            spin_parameter = params.radius_m * float(np.linalg.norm(omega_perp)) / max(speed, 1e-12)
            cl = min(params.max_lift_coefficient, params.magnus_lift_slope * spin_parameter)
            lift_force = 0.5 * params.air_density_kg_m3 * params.cross_section_m2 * cl * speed**2
            acceleration += (lift_force / params.mass_kg) * (cross / cross_norm)
    return acceleration


def _rk4_step(position: Array, velocity: Array, spin: Array, dt: float, params: BallPhysicalParams) -> tuple[Array, Array]:
    def deriv(pos: Array, vel: Array) -> tuple[Array, Array]:
        del pos
        return vel, _acceleration(vel, spin, params)

    k1_p, k1_v = deriv(position, velocity)
    k2_p, k2_v = deriv(position + 0.5 * dt * k1_p, velocity + 0.5 * dt * k1_v)
    k3_p, k3_v = deriv(position + 0.5 * dt * k2_p, velocity + 0.5 * dt * k2_v)
    k4_p, k4_v = deriv(position + dt * k3_p, velocity + dt * k3_v)
    next_p = position + dt / 6.0 * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
    next_v = velocity + dt / 6.0 * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
    return next_p, next_v


def propagate_state(
    state: BallState,
    target_timestamp_s: float,
    params: BallPhysicalParams,
    max_step_s: float = 1.0 / 500.0,
) -> BallState:
    """Propagate a state forward; spin is held constant over the flight segment."""
    target = float(target_timestamp_s)
    if target < state.timestamp_s:
        raise ValueError("target timestamp must not precede the state")
    if max_step_s <= 0:
        raise ValueError("max_step_s must be positive")
    duration = target - state.timestamp_s
    if duration == 0:
        return state
    steps = max(1, int(math.ceil(duration / max_step_s)))
    dt = duration / steps
    p, v = state.position_m.copy(), state.velocity_m_s.copy()
    for _ in range(steps):
        p, v = _rk4_step(p, v, state.spin_rad_s, dt, params)
    return BallState(p, v, target, state.spin_rad_s)


def predict_trajectory(
    state: BallState,
    timestamps_s: Sequence[float],
    params: BallPhysicalParams,
) -> list[BallState]:
    result: list[BallState] = []
    for timestamp in timestamps_s:
        result.append(propagate_state(state, float(timestamp), params))
    return result


def associate_observations(
    predicted_state: BallState,
    observations: Sequence[BallObservation],
    projector: Projector,
    gate_px: float = 80.0,
    sigma_px: float = 20.0,
    timestamp_tolerance_s: float = 0.02,
) -> CandidateMatch | None:
    """Choose the candidate most compatible with the projected physical prediction."""
    if gate_px <= 0 or sigma_px <= 0 or timestamp_tolerance_s < 0:
        raise ValueError("invalid association thresholds")
    predicted = np.asarray(projector(predicted_state.position_m), dtype=float)
    if predicted.shape != (2,) or not np.all(np.isfinite(predicted)):
        raise ValueError("projector must return a finite 2-vector")

    best: CandidateMatch | None = None
    for observation in observations:
        if abs(observation.timestamp_s - predicted_state.timestamp_s) > timestamp_tolerance_s:
            continue
        residual = float(np.linalg.norm(observation.pixel_xy - predicted))
        if residual > gate_px:
            continue
        likelihood = math.exp(-0.5 * (residual / sigma_px) ** 2)
        score = float(observation.confidence * likelihood)
        candidate = CandidateMatch(observation, predicted.copy(), residual, score)
        if best is None or candidate.score > best.score:
            best = candidate
    return best


def _huber_weight(norm: float, delta: float) -> float:
    return 1.0 if norm <= delta else delta / max(norm, 1e-12)


def fit_initial_state_world(
    observations: Sequence[WorldBallObservation],
    initial_state: BallState,
    params: BallPhysicalParams,
    *,
    fit_drag_coefficient: bool = False,
    huber_delta_m: float = 0.03,
    max_iterations: int = 25,
    tolerance: float = 1e-7,
) -> StateFitResult:
    """Robust Gauss-Newton/IRLS fit from metric 3-D observations.

    This function is intentionally *not* a monocular pixel-to-3D solver. It is a
    numerical core for calibrated/multiview observations or synthetic tests.
    Monocular data must first be reconciled with camera geometry/constraints.
    Huber weights are held fixed inside each Gauss-Newton step (IRLS), avoiding
    derivatives of the robust weighting function from destabilizing the fit.
    """
    if len(observations) < 3:
        raise ValueError("at least three observations are required")
    ordered = sorted(observations, key=lambda item: item.timestamp_s)
    if ordered[0].timestamp_s < initial_state.timestamp_s:
        raise ValueError("observations cannot precede initial_state")
    if huber_delta_m <= 0 or max_iterations <= 0:
        raise ValueError("invalid fitting settings")

    theta = np.r_[initial_state.position_m, initial_state.velocity_m_s]
    if fit_drag_coefficient:
        theta = np.r_[theta, math.log(max(params.drag_coefficient, 1e-6))]

    def decode(values: Array) -> tuple[BallState, BallPhysicalParams]:
        state = BallState(values[:3], values[3:6], initial_state.timestamp_s, initial_state.spin_rad_s)
        local_params = params
        if fit_drag_coefficient:
            cd = float(np.clip(math.exp(values[6]), 0.02, 2.0))
            local_params = replace(params, drag_coefficient=cd)
        return state, local_params

    def errors(values: Array) -> list[Array]:
        state, local_params = decode(values)
        return [
            propagate_state(state, observation.timestamp_s, local_params).position_m - observation.position_m
            for observation in ordered
        ]

    def residual(values: Array, robust_weights: Array) -> Array:
        pieces = []
        for observation, error, robust_weight in zip(ordered, errors(values), robust_weights):
            weight = math.sqrt(observation.confidence * robust_weight)
            pieces.append(weight * error)
        return np.concatenate(pieces)

    damping = 1e-3
    converged = False
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        current_errors = errors(theta)
        robust_weights = np.array([
            _huber_weight(float(np.linalg.norm(error)), huber_delta_m)
            for error in current_errors
        ])
        r = residual(theta, robust_weights)
        cost = float(r @ r)
        jacobian = np.empty((len(r), len(theta)))
        for j in range(len(theta)):
            step = 1e-5 * max(1.0, abs(theta[j]))
            shifted = theta.copy()
            shifted[j] += step
            jacobian[:, j] = (residual(shifted, robust_weights) - r) / step
        system = jacobian.T @ jacobian + damping * np.eye(len(theta))
        gradient = jacobian.T @ r
        try:
            delta = np.linalg.solve(system, -gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(system, -gradient, rcond=None)[0]
        candidate = theta + delta
        new_cost = float(residual(candidate, robust_weights) @ residual(candidate, robust_weights))
        if new_cost < cost:
            theta = candidate
            damping = max(damping / 3.0, 1e-9)
            if np.linalg.norm(delta) < tolerance:
                converged = True
                break
        else:
            damping = min(damping * 10.0, 1e9)

    fitted_state, fitted_params = decode(theta)
    final_errors = errors(theta)
    rmse = float(math.sqrt(np.mean([np.linalg.norm(error) ** 2 for error in final_errors])))
    return StateFitResult(fitted_state, fitted_params, rmse, iterations, converged)
