"""Projected image-space lifecycle around the existing physical ball core.

The monocular calibration is deliberately planar/approximate.  It is useful for
prediction gating, but its metric state must not be reported as calibrated 3-D.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import cv2
import numpy as np

from .ball_detection import BallCandidate
from .ball_tracking import BallPhysicalParams, BallState, associate_observations, propagate_state


class TrackStatus(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    OBSERVED = "OBSERVED"
    PREDICTED = "PREDICTED"
    LOST = "LOST"


@dataclass(frozen=True)
class PlanarSceneCalibration:
    table_polygon_xy: np.ndarray
    table_length_m: float = 2.74
    pixels_per_m_vertical: float = 250.0
    net_polygon_xy: np.ndarray | None = None

    def __post_init__(self):
        polygon = np.asarray(self.table_polygon_xy, dtype=float)
        if polygon.shape != (4, 2):
            raise ValueError("table polygon must be far-left, far-right, near-right, near-left")
        if self.table_length_m <= 0 or self.pixels_per_m_vertical <= 0:
            raise ValueError("scene scales must be positive")
        object.__setattr__(self, "table_polygon_xy", polygon)
        if self.net_polygon_xy is not None:
            net = np.asarray(self.net_polygon_xy, dtype=float)
            if net.shape != (4, 2) or not np.all(np.isfinite(net)):
                raise ValueError("net polygon must contain four finite pixel pairs")
            object.__setattr__(self, "net_polygon_xy", net)

    @property
    def left_center(self):
        return (self.table_polygon_xy[0] + self.table_polygon_xy[3]) / 2

    @property
    def right_center(self):
        return (self.table_polygon_xy[1] + self.table_polygon_xy[2]) / 2

    @property
    def pixels_per_m_horizontal(self):
        return float(abs(self.right_center[0] - self.left_center[0]) / self.table_length_m)

    @property
    def center_x(self):
        return float((self.left_center[0] + self.right_center[0]) / 2)

    def surface_y(self, pixel_x: float) -> float:
        left, right = self.left_center, self.right_center
        fraction = float(np.clip((pixel_x - left[0]) / max(right[0] - left[0], 1e-6), -0.5, 1.5))
        return float(left[1] + fraction * (right[1] - left[1]))

    def tabletop_y(self, pixel_x: float) -> float:
        """Image y of the far/top table edge used only as a contact prior."""
        left, right = self.table_polygon_xy[0], self.table_polygon_xy[1]
        fraction = float(np.clip((pixel_x - left[0]) / max(right[0] - left[0], 1e-6), -0.25, 1.25))
        return float(left[1] + fraction * (right[1] - left[1]))

    def table_near_y(self, pixel_x: float) -> float:
        """Image y of the near table edge at ``pixel_x``."""
        left, right = self.table_polygon_xy[3], self.table_polygon_xy[2]
        fraction = float(np.clip((pixel_x - left[0]) / max(right[0] - left[0], 1e-6), -0.25, 1.25))
        return float(left[1] + fraction * (right[1] - left[1]))

    def surface_bounds_y(self, pixel_x: float) -> tuple[float, float]:
        values = sorted((self.tabletop_y(pixel_x), self.table_near_y(pixel_x)))
        return float(values[0]), float(values[1])

    def image_to_plane(self, pixel_xy: np.ndarray) -> np.ndarray:
        u, v = np.asarray(pixel_xy, dtype=float)
        x = (u - self.center_x) / self.pixels_per_m_horizontal
        z = (self.surface_y(u) - v) / self.pixels_per_m_vertical
        return np.array([x, 0.0, z])

    def project(self, world_xyz: np.ndarray) -> np.ndarray:
        x, _, z = np.asarray(world_xyz, dtype=float)
        u = self.center_x + x * self.pixels_per_m_horizontal
        v = self.surface_y(u) - z * self.pixels_per_m_vertical
        return np.array([u, v])

    def close_to_table_surface(self, pixel_xy: np.ndarray, margin_px: float = 40.0) -> bool:
        distance = cv2.pointPolygonTest(self.table_polygon_xy.astype(np.float32),
                                        tuple(np.asarray(pixel_xy, dtype=float)), True)
        return bool(distance >= -margin_px)


@dataclass
class TrackFrame:
    timestamp_s: float
    status: TrackStatus
    observed_xy: np.ndarray | None
    predicted_xy: np.ndarray | None
    state: BallState | None
    confidence: float
    missed_frames: int
    selected_candidate: BallCandidate | None = None


@dataclass
class _Seed:
    candidates: list[BallCandidate]
    score: float


class ProjectedBallTracker:
    """Initialize from coherent detections, then alternate observation/prediction."""

    def __init__(self, calibration: PlanarSceneCalibration, params: BallPhysicalParams,
                 *, use_physics: bool = True, max_prediction_frames: int = 4,
                 base_gate_px: float = 75.0):
        self.calibration = calibration
        self.params = params
        self.use_physics = use_physics
        self.max_prediction_frames = max_prediction_frames
        self.base_gate_px = base_gate_px
        self.status = TrackStatus.UNINITIALIZED
        self.state: BallState | None = None
        self.missed = 0
        self.seeds: list[_Seed] = []
        self.last_observed_world: np.ndarray | None = None
        self.last_observed_pixel: np.ndarray | None = None
        self.last_observed_time: float | None = None
        self.slow_observation_count = 0
        self.ever_initialized = False

    def _predict_state(self, timestamp_s: float) -> BallState | None:
        if self.state is None or timestamp_s < self.state.timestamp_s:
            return None
        if self.use_physics:
            return propagate_state(self.state, timestamp_s, self.params)
        dt = timestamp_s - self.state.timestamp_s
        return BallState(self.state.position_m + dt * self.state.velocity_m_s,
                         self.state.velocity_m_s, timestamp_s, self.state.spin_rad_s)

    def predict_pixel(self, timestamp_s: float) -> np.ndarray | None:
        state = self._predict_state(timestamp_s)
        return None if state is None else self.calibration.project(state.position_m)

    def reset_for_discontinuity(self) -> None:
        """End a free-flight segment before a possible impact/reinitialization."""
        self.state = None
        self.missed = 0
        self.seeds.clear()
        self.last_observed_world = None
        self.last_observed_pixel = None
        self.last_observed_time = None
        self.slow_observation_count = 0
        self.status = TrackStatus.LOST if self.ever_initialized else TrackStatus.UNINITIALIZED

    @staticmethod
    def _seed_residual(items: Sequence[BallCandidate]) -> float:
        if len(items) < 3:
            return 0.0
        t = np.array([c.observation.timestamp_s for c in items])
        xy = np.array([c.observation.pixel_xy for c in items])
        prediction = np.column_stack([np.polyval(np.polyfit(t - t[0], xy[:, d], 1), t - t[0]) for d in range(2)])
        return float(np.sqrt(np.mean(np.sum((xy - prediction) ** 2, axis=1))))

    def _advance_seeds(self, candidates: Sequence[BallCandidate], required_observations: int = 3) -> _Seed | None:
        if required_observations not in (2, 3):
            raise ValueError("required_observations must be 2 or 3")
        candidates = [candidate for candidate in candidates
                      if candidate.reason_scores.get("motion", 0) >= 0.08
                      and (candidate.reason_scores.get("bright_unsaturated", 0) >= 0.25
                           or candidate.reason_scores.get("contrast", 0) >= 0.35)]
        proposals: list[_Seed] = []
        for candidate in candidates[:20]:
            evidence = candidate.observation.confidence + 0.6 * candidate.reason_scores.get("motion", 0)
            proposals.append(_Seed([candidate], evidence))
            for seed in self.seeds:
                last = seed.candidates[-1].observation
                dt = candidate.observation.timestamp_s - last.timestamp_s
                if not 0 < dt <= 0.14:
                    continue
                displacement = float(np.linalg.norm(candidate.observation.pixel_xy - last.pixel_xy))
                speed = displacement / dt
                if not 70 <= speed <= 6000:
                    continue
                if last.size_px and candidate.observation.size_px:
                    ratio = candidate.observation.size_px / last.size_px
                    if not 0.25 <= ratio <= 4.0:
                        continue
                extended = (seed.candidates + [candidate])[-5:]
                residual = self._seed_residual(extended)
                if len(extended) >= 3 and residual > 38:
                    continue
                score = seed.score + evidence - residual / 100.0
                proposals.append(_Seed(extended, score))
        proposals.sort(key=lambda item: (-len(item.candidates), -item.score,
                                         float(item.candidates[-1].observation.pixel_xy[1]),
                                         float(item.candidates[-1].observation.pixel_xy[0])))
        self.seeds = proposals[:100]
        for seed in self.seeds:
            if len(seed.candidates) < required_observations:
                continue
            xy = np.array([c.observation.pixel_xy for c in seed.candidates])
            span = seed.candidates[-1].observation.timestamp_s - seed.candidates[0].observation.timestamp_s
            mean_speed = float(np.linalg.norm(xy[-1] - xy[0]) / max(span, 1e-6))
            mean_motion = float(np.mean([c.reason_scores.get("motion", 0) for c in seed.candidates]))
            if (span <= 0.2 and mean_speed >= 500 and mean_motion >= .15
                    and self._seed_residual(seed.candidates) <= 25):
                return seed
        return None

    def _initialize(self, seed: _Seed) -> None:
        t = np.array([c.observation.timestamp_s for c in seed.candidates])
        world = np.array([self.calibration.image_to_plane(c.observation.pixel_xy) for c in seed.candidates])
        velocity = np.array([np.polyfit(t - t[-1], world[:, d], 1)[0] for d in range(3)])
        self.state = BallState(world[-1], velocity, t[-1])
        self.last_observed_world = world[-1]
        self.last_observed_pixel = seed.candidates[-1].observation.pixel_xy.copy()
        self.last_observed_time = float(t[-1])
        self.slow_observation_count = 0
        self.missed = 0
        self.status = TrackStatus.OBSERVED
        self.ever_initialized = True
        self.seeds.clear()

    def step(self, timestamp_s: float, candidates: Sequence[BallCandidate], *,
             gate_multiplier: float = 1.0, required_seed_observations: int = 3,
             predicted_velocity_weight: float = .35,
             max_prediction_frames: int | None = None) -> TrackFrame:
        timestamp_s = float(timestamp_s)
        if gate_multiplier <= 0 or not 0 <= predicted_velocity_weight <= 1:
            raise ValueError("invalid event-aware tracking parameters")
        if self.state is None:
            seed = self._advance_seeds(candidates, required_seed_observations)
            if seed is None:
                self.status = TrackStatus.LOST if self.ever_initialized else TrackStatus.UNINITIALIZED
                return TrackFrame(timestamp_s, self.status, None, None, None, 0.0, self.missed)
            self._initialize(seed)
            selected = seed.candidates[-1]
            return TrackFrame(timestamp_s, self.status, selected.observation.pixel_xy, None,
                              self.state, selected.observation.confidence, 0, selected)

        predicted_state = self._predict_state(timestamp_s)
        if predicted_state is None:
            # A discontinuity or an externally anchored observation can leave
            # the state marginally ahead of the next timestamp.  Treat it as
            # a new flight segment instead of crashing while propagating a
            # non-existent prediction.
            self.reset_for_discontinuity()
            return self.step(
                timestamp_s, candidates,
                gate_multiplier=gate_multiplier,
                required_seed_observations=required_seed_observations,
                predicted_velocity_weight=predicted_velocity_weight,
                max_prediction_frames=max_prediction_frames,
            )
        predicted_xy = self.calibration.project(predicted_state.position_m)
        candidates = [candidate for candidate in candidates
                      if candidate.reason_scores.get("motion", 0) >= 0.05
                      and (candidate.reason_scores.get("bright_unsaturated", 0) >= 0.18
                           or candidate.reason_scores.get("contrast", 0) >= 0.30)]
        observations = [candidate.observation for candidate in candidates]
        gate = gate_multiplier * (self.base_gate_px + 28 * self.missed)
        match = associate_observations(predicted_state, observations, self.calibration.project,
                                       gate_px=gate, sigma_px=0.55 * gate,
                                       timestamp_tolerance_s=0.002)
        selected = None
        if match is not None and match.score >= 0.035:
            selected = next(c for c in candidates if c.observation is match.observation)
            measured = self.calibration.image_to_plane(match.observation.pixel_xy)
            if self.last_observed_world is not None and self.last_observed_time is not None:
                dt = timestamp_s - self.last_observed_time
                measured_velocity = (measured - self.last_observed_world) / max(dt, 1e-6)
                velocity = ((1 - predicted_velocity_weight) * measured_velocity
                            + predicted_velocity_weight * predicted_state.velocity_m_s)
                apparent_speed = float(np.linalg.norm(match.observation.pixel_xy - self.last_observed_pixel) /
                                       max(dt, 1e-6)) if self.last_observed_pixel is not None else math.inf
                self.slow_observation_count = self.slow_observation_count + 1 if apparent_speed < 350 else 0
            else:
                velocity = predicted_state.velocity_m_s
                self.slow_observation_count = 0
            if self.slow_observation_count >= 2:
                self.state = None
                self.last_observed_world = None
                self.last_observed_pixel = None
                self.last_observed_time = None
                self.seeds.clear()
                self.status = TrackStatus.LOST
                return TrackFrame(timestamp_s, self.status, None, predicted_xy, None, 0.0, self.missed)
            position = 0.8 * measured + 0.2 * predicted_state.position_m
            position[1] = 0.0
            velocity[1] = 0.0
            self.state = BallState(position, velocity, timestamp_s)
            self.last_observed_world = measured
            self.last_observed_pixel = match.observation.pixel_xy.copy()
            self.last_observed_time = timestamp_s
            self.missed = 0
            self.status = TrackStatus.OBSERVED
            confidence = float(match.observation.confidence * math.exp(-match.residual_px / max(gate, 1)))
            return TrackFrame(timestamp_s, self.status, match.observation.pixel_xy, predicted_xy,
                              self.state, confidence, 0, selected)

        self.missed += 1
        prediction_limit = self.max_prediction_frames if max_prediction_frames is None else max_prediction_frames
        if self.missed <= prediction_limit:
            self.state = predicted_state
            self.status = TrackStatus.PREDICTED
            return TrackFrame(timestamp_s, self.status, None, predicted_xy, self.state,
                              float(0.55 ** self.missed), self.missed)
        self.state = None
        self.last_observed_world = None
        self.last_observed_pixel = None
        self.last_observed_time = None
        self.slow_observation_count = 0
        self.seeds.clear()
        self.status = TrackStatus.LOST
        return TrackFrame(timestamp_s, self.status, None, predicted_xy, None, 0.0, self.missed)
