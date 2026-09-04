"""Multimodal context for table-tennis ball and racket tracking.

The module is deliberately independent of MediaPipe and model runtimes.  It
contains the normalized interchange types, two-player racket association,
candidate fusion, interaction priors and the short offline event smoother.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import itertools
import math
from typing import Mapping, Sequence

import numpy as np

from .ball_detection import BallCandidate
from .ball_tracking import BallObservation
from .ball_video_tracking import PlanarSceneCalibration, TrackFrame, TrackStatus


Array = np.ndarray
RACKET_KEYPOINT_NAMES = ("top", "bottom", "handle", "left", "right")


class InteractionMode(str, Enum):
    FREE_FLIGHT = "FREE_FLIGHT"
    BOUNCE_PENDING = "BOUNCE_PENDING"
    CONTACT_PENDING = "CONTACT_PENDING"
    POST_EVENT = "POST_EVENT"


@dataclass(frozen=True)
class BodyPoseFrame:
    player_id: str
    right_wrist_xy: Array | None = None
    right_elbow_xy: Array | None = None
    right_shoulder_xy: Array | None = None
    bbox_xyxy: Array | None = None
    confidence: float = 0.0
    source: str = "unavailable"
    landmarks_xy: Array | None = None
    landmark_confidence: Array | None = None

    def __post_init__(self) -> None:
        for name in ("right_wrist_xy", "right_elbow_xy", "right_shoulder_xy"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.shape != (2,) or not np.all(np.isfinite(value)):
                    raise ValueError(f"{name} must be a finite 2-vector")
                object.__setattr__(self, name, value)
        if self.bbox_xyxy is not None:
            bbox = np.asarray(self.bbox_xyxy, dtype=float)
            if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
                raise ValueError("bbox_xyxy must contain four finite values")
            object.__setattr__(self, "bbox_xyxy", bbox)
        if not 0 <= self.confidence <= 1:
            raise ValueError("body confidence must be in [0,1]")
        if self.landmarks_xy is not None:
            landmarks = np.asarray(self.landmarks_xy, dtype=float)
            if landmarks.shape != (33, 2):
                raise ValueError("landmarks_xy must have shape (33,2)")
            object.__setattr__(self, "landmarks_xy", landmarks)
        if self.landmark_confidence is not None:
            quality = np.asarray(self.landmark_confidence, dtype=float)
            if quality.shape != (33,):
                raise ValueError("landmark_confidence must have shape (33,)")
            object.__setattr__(self, "landmark_confidence", quality)


@dataclass(frozen=True)
class RacketPoseObservation:
    timestamp_s: float
    bbox_xyxy: Array
    keypoints_xy: Array
    keypoint_scores: Array
    confidence: float
    source: str = "racketvision_racketpose"

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox_xyxy, dtype=float)
        points = np.asarray(self.keypoints_xy, dtype=float)
        scores = np.asarray(self.keypoint_scores, dtype=float)
        if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
            raise ValueError("bbox_xyxy must contain four finite values")
        if points.shape != (5, 2) or not np.all(np.isfinite(points)):
            raise ValueError("keypoints_xy must have shape (5,2)")
        if scores.shape != (5,) or not np.all(np.isfinite(scores)):
            raise ValueError("keypoint_scores must have shape (5,)")
        if not 0 <= self.confidence <= 1 or np.any((scores < 0) | (scores > 1)):
            raise ValueError("racket confidences must be in [0,1]")
        object.__setattr__(self, "timestamp_s", float(self.timestamp_s))
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "keypoints_xy", points)
        object.__setattr__(self, "keypoint_scores", scores)

    def point(self, name: str) -> Array:
        return self.keypoints_xy[RACKET_KEYPOINT_NAMES.index(name)]

    @property
    def handle_xy(self) -> Array:
        return self.point("handle")

    @property
    def head_center_xy(self) -> Array:
        return self.keypoints_xy[[0, 1, 3, 4]].mean(axis=0)

    @property
    def head_polygon_xy(self) -> Array:
        return self.keypoints_xy[[0, 4, 1, 3]]


@dataclass(frozen=True)
class RacketTrackFrame:
    player_id: str
    timestamp_s: float
    status: TrackStatus
    pose: RacketPoseObservation | None
    velocity_px_s: Array = field(default_factory=lambda: np.zeros(2))
    confidence: float = 0.0
    association_score: float = 0.0

    def __post_init__(self) -> None:
        velocity = np.asarray(self.velocity_px_s, dtype=float)
        if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity_px_s must be a finite 2-vector")
        object.__setattr__(self, "velocity_px_s", velocity)

    @property
    def center_xy(self) -> Array | None:
        return None if self.pose is None else self.pose.head_center_xy


@dataclass(frozen=True)
class InteractionContext:
    timestamp_s: float
    bodies: Mapping[str, BodyPoseFrame] = field(default_factory=dict)
    rackets: Mapping[str, RacketTrackFrame] = field(default_factory=dict)
    mode: InteractionMode = InteractionMode.FREE_FLIGHT


@dataclass(frozen=True)
class InteractionMetric:
    player_id: str
    distance_px: float
    closing_speed_px_s: float
    time_to_closest_s: float | None
    source: str
    swept_intersection: bool = False


@dataclass(frozen=True)
class MultimodalTrackFrame:
    track: TrackFrame
    mode: InteractionMode
    interaction_metrics: tuple[InteractionMetric, ...]
    alternatives: tuple[tuple[float, float, float, str], ...]
    candidate_count_by_source: Mapping[str, int]


class InteractionAwareBallTracker:
    """Causal tracker with local high-recall gates around physical events."""

    def __init__(self, tracker, scene: PlanarSceneCalibration, *, event_prediction_frames: int = 6,
                 enable_bounce: bool = True, enable_contact: bool = True):
        self.tracker = tracker
        self.scene = scene
        self.event_prediction_frames = event_prediction_frames
        self.enable_bounce = enable_bounce
        self.enable_contact = enable_contact
        self.mode = InteractionMode.FREE_FLIGHT
        self.post_event_remaining = 0
        self.observed_history: list[tuple[float, Array]] = []

    def _pixel_velocity(self) -> Array | None:
        if len(self.observed_history) < 2:
            return None
        (t0, p0), (t1, p1) = self.observed_history[-2:]
        return None if t1 <= t0 else (p1 - p0) / (t1 - t0)

    def step(self, timestamp_s: float, classical: Sequence[BallCandidate],
             model: Sequence[BallObservation], context: InteractionContext,
             anchor: TrackFrame | None = None) -> MultimodalTrackFrame:
        predicted = self.tracker.predict_pixel(timestamp_s)
        reference = predicted
        if reference is None and self.observed_history:
            reference = self.observed_history[-1][1]
        velocity = self._pixel_velocity()
        proposed_mode, metrics = infer_interaction_mode(reference, velocity, context, self.scene, self.mode)
        if proposed_mode == InteractionMode.BOUNCE_PENDING and not self.enable_bounce:
            proposed_mode = InteractionMode.FREE_FLIGHT
        if proposed_mode == InteractionMode.CONTACT_PENDING and not self.enable_contact:
            proposed_mode = InteractionMode.FREE_FLIGHT
        if self.post_event_remaining > 0:
            proposed_mode = InteractionMode.POST_EVENT
            self.post_event_remaining -= 1
        event_active = proposed_mode != InteractionMode.FREE_FLIGHT
        contextual = replace(context, mode=proposed_mode)
        candidates = fuse_ball_candidates(classical, model, contextual, predicted, velocity)
        by_source: dict[str, int] = {}
        for candidate in candidates:
            for source in candidate.observation.source.split("+"):
                by_source[source] = by_source.get(source, 0) + 1
        prior_missed = self.tracker.missed
        if anchor is not None and anchor.status == TrackStatus.OBSERVED and anchor.observed_xy is not None:
            # The proven classical association remains the conservative anchor.
            # Learned heatmaps are used to bridge failures, not to displace a
            # valid observation in ordinary flight.
            selected = min(candidates, key=lambda candidate: np.linalg.norm(
                candidate.observation.pixel_xy - anchor.observed_xy), default=anchor.selected_candidate)
            result = replace(anchor, selected_candidate=selected)
            self.tracker.state = anchor.state
            self.tracker.status = TrackStatus.OBSERVED
            self.tracker.missed = 0
            self.tracker.ever_initialized = True
            self.tracker.last_observed_world = self.scene.image_to_plane(anchor.observed_xy)
            self.tracker.last_observed_pixel = anchor.observed_xy.copy()
            self.tracker.last_observed_time = timestamp_s
            self.tracker.slow_observation_count = 0
            self.tracker.seeds.clear()
        else:
            result = self.tracker.step(
                timestamp_s, candidates,
                gate_multiplier=2.2 if event_active else 1.0,
                required_seed_observations=2 if event_active or self.mode != InteractionMode.FREE_FLIGHT else 3,
                predicted_velocity_weight=0.0 if event_active else .35,
                max_prediction_frames=self.event_prediction_frames if event_active else None,
            )
        prior_mode = self.mode
        if result.observed_xy is not None:
            if self.mode in (InteractionMode.BOUNCE_PENDING, InteractionMode.CONTACT_PENDING) and prior_missed:
                proposed_mode = InteractionMode.POST_EVENT
                self.post_event_remaining = 2
            self.observed_history.append((timestamp_s, result.observed_xy.copy()))
            self.observed_history[:] = self.observed_history[-4:]
        if (proposed_mode == InteractionMode.POST_EVENT
                and prior_mode in (InteractionMode.CONTACT_PENDING, InteractionMode.BOUNCE_PENDING)
                and self.post_event_remaining == 0):
            self.post_event_remaining = 2
        self.mode = proposed_mode
        alternatives = tuple((float(c.observation.pixel_xy[0]), float(c.observation.pixel_xy[1]),
                              float(c.observation.confidence), c.observation.source)
                             for c in candidates[:3])
        return MultimodalTrackFrame(result, proposed_mode, tuple(metrics), alternatives, by_source)


def wrist_proxy_racket(body: BodyPoseFrame, timestamp_s: float) -> RacketPoseObservation | None:
    """Build a low-confidence geometric racket proxy from wrist and elbow."""
    if body.right_wrist_xy is None or body.right_elbow_xy is None or body.confidence <= 0:
        return None
    direction = body.right_wrist_xy - body.right_elbow_xy
    norm = float(np.linalg.norm(direction))
    if norm < 2:
        return None
    direction /= norm
    perpendicular = np.array([-direction[1], direction[0]])
    bottom = body.right_wrist_xy + 8 * direction
    top = body.right_wrist_xy + 62 * direction
    center = body.right_wrist_xy + 38 * direction
    points = np.stack([top, bottom, body.right_wrist_xy,
                       center - 24 * perpendicular, center + 24 * perpendicular])
    bbox = np.r_[points.min(axis=0) - 5, points.max(axis=0) + 5]
    confidence = min(0.35, 0.45 * body.confidence)
    return RacketPoseObservation(timestamp_s, bbox, points, np.full(5, confidence), confidence,
                                 source="right_wrist_forearm_proxy")


def _bbox_contains(bbox: Array | None, point: Array, margin: float = 0.0) -> bool:
    return bool(bbox is not None and bbox[0] - margin <= point[0] <= bbox[2] + margin
                and bbox[1] - margin <= point[1] <= bbox[3] + margin)


def _association_score(observation: RacketPoseObservation, body: BodyPoseFrame,
                       previous: RacketTrackFrame | None, frame_width: float) -> float:
    if body.right_wrist_xy is None:
        wrist_score = 0.05
    else:
        wrist_score = math.exp(-0.5 * (np.linalg.norm(observation.handle_xy - body.right_wrist_xy) / 100) ** 2)
    side_expected = 0.25 * frame_width if body.player_id == "left" else 0.75 * frame_width
    side_score = math.exp(-0.5 * ((observation.handle_xy[0] - side_expected) / (0.45 * frame_width)) ** 2)
    overlap_score = 1.0 if _bbox_contains(body.bbox_xyxy, observation.handle_xy, 100) else 0.25
    continuity = 0.5
    if previous is not None and previous.center_xy is not None:
        dt = observation.timestamp_s - previous.timestamp_s
        expected = previous.center_xy + max(dt, 0) * previous.velocity_px_s
        continuity = math.exp(-0.5 * (np.linalg.norm(observation.head_center_xy - expected) / 120) ** 2)
    return float(observation.confidence * (0.52 * wrist_score + 0.13 * side_score
                                            + 0.12 * overlap_score + 0.23 * continuity))


def associate_rackets_to_players(observations: Sequence[RacketPoseObservation],
                                 bodies: Mapping[str, BodyPoseFrame],
                                 previous: Mapping[str, RacketTrackFrame] | None,
                                 frame_width: int) -> dict[str, tuple[RacketPoseObservation, float]]:
    """Globally associate up to one racket per player without SciPy."""
    players = tuple(player for player in ("left", "right") if player in bodies)
    if not players or not observations:
        return {}
    previous = previous or {}
    best_score = -math.inf
    best: dict[str, tuple[RacketPoseObservation, float]] = {}
    # Include an explicit unmatched option for each player.
    choices = list(range(len(observations))) + [None]
    for assignment in itertools.product(choices, repeat=len(players)):
        assigned = [idx for idx in assignment if idx is not None]
        if len(assigned) != len(set(assigned)):
            continue
        total = 0.0
        proposal = {}
        for player, index in zip(players, assignment):
            if index is None:
                continue
            score = _association_score(observations[index], bodies[player], previous.get(player), frame_width)
            if score < 0.08:
                continue
            total += score
            proposal[player] = (observations[index], score)
        if total > best_score:
            best_score, best = total, proposal
    return best


class TwoRacketTracker:
    def __init__(self, max_prediction_frames: int = 4):
        self.max_prediction_frames = max_prediction_frames
        self.previous: dict[str, RacketTrackFrame] = {}
        self.missed = {"left": 0, "right": 0}

    def step(self, timestamp_s: float, observations: Sequence[RacketPoseObservation],
             bodies: Mapping[str, BodyPoseFrame], frame_width: int) -> dict[str, RacketTrackFrame]:
        matches = associate_rackets_to_players(observations, bodies, self.previous, frame_width)
        result: dict[str, RacketTrackFrame] = {}
        for player in ("left", "right"):
            if player in matches:
                pose, score = matches[player]
                old = self.previous.get(player)
                velocity = np.zeros(2)
                if old is not None and old.center_xy is not None:
                    dt = timestamp_s - old.timestamp_s
                    if dt > 0:
                        velocity = (pose.head_center_xy - old.center_xy) / dt
                frame = RacketTrackFrame(player, timestamp_s, TrackStatus.OBSERVED, pose, velocity,
                                         pose.confidence, score)
                self.missed[player] = 0
            else:
                old = self.previous.get(player)
                self.missed[player] += 1
                proxy = wrist_proxy_racket(bodies[player], timestamp_s) if player in bodies else None
                if proxy is not None:
                    frame = RacketTrackFrame(player, timestamp_s, TrackStatus.PREDICTED, proxy,
                                             np.zeros(2), proxy.confidence, proxy.confidence)
                elif old is not None and old.pose is not None and self.missed[player] <= self.max_prediction_frames:
                    dt = timestamp_s - old.timestamp_s
                    shifted = old.pose.keypoints_xy + dt * old.velocity_px_s
                    bbox = np.r_[shifted.min(axis=0), shifted.max(axis=0)]
                    pose = RacketPoseObservation(timestamp_s, bbox, shifted, old.pose.keypoint_scores * .7,
                                                 old.confidence * .7, source="temporal_racket_prediction")
                    frame = RacketTrackFrame(player, timestamp_s, TrackStatus.PREDICTED, pose,
                                             old.velocity_px_s, pose.confidence, 0.0)
                else:
                    frame = RacketTrackFrame(player, timestamp_s, TrackStatus.LOST, None)
            result[player] = frame
        self.previous = result
        return result


def _distance_to_polygon(point: Array, polygon: Array) -> float:
    best = math.inf
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        segment = end - start
        fraction = float(np.clip(np.dot(point - start, segment) / max(np.dot(segment, segment), 1e-9), 0, 1))
        best = min(best, float(np.linalg.norm(point - (start + fraction * segment))))
    return best


def interaction_metrics(ball_xy: Array | None, ball_velocity_px_s: Array | None,
                        context: InteractionContext) -> list[InteractionMetric]:
    if ball_xy is None:
        return []
    ball_xy = np.asarray(ball_xy, dtype=float)
    ball_velocity = np.zeros(2) if ball_velocity_px_s is None else np.asarray(ball_velocity_px_s, dtype=float)
    metrics = []
    for player, racket in context.rackets.items():
        if racket.pose is None:
            continue
        center = racket.center_xy
        relative_position = ball_xy - center
        relative_velocity = ball_velocity - racket.velocity_px_s
        distance = _distance_to_polygon(ball_xy, racket.pose.head_polygon_xy)
        radial = float(np.dot(relative_position, relative_velocity) / max(np.linalg.norm(relative_position), 1e-9))
        closing = max(0.0, -radial)
        speed2 = float(np.dot(relative_velocity, relative_velocity))
        ttc = None if speed2 < 1e-9 else float(np.clip(-np.dot(relative_position, relative_velocity) / speed2, 0, .5))
        swept_intersection = False
        for horizon in (1 / 30, 2 / 30, 3 / 30, 4 / 30):
            future_ball = ball_xy + horizon * ball_velocity
            future_polygon = racket.pose.head_polygon_xy + horizon * racket.velocity_px_s
            if _distance_to_polygon(future_ball, future_polygon) <= 35:
                swept_intersection = True
                break
        metrics.append(InteractionMetric(player, distance, closing, ttc, racket.pose.source,
                                         swept_intersection))
    return metrics


def _candidate_from_model(observation: BallObservation) -> BallCandidate:
    size = max(3, int(round(observation.size_px or 7)))
    center = observation.pixel_xy
    bbox = (int(round(center[0] - size / 2)), int(round(center[1] - size / 2)), size, size)
    reasons = {"model_heatmap": observation.confidence, "motion": .35,
               "bright_unsaturated": .35, "contrast": .35}
    return BallCandidate(observation, bbox, float(size * size), 1.0, 1.0, 0, 0, 0, 0, 0, reasons)


def fuse_ball_candidates(classical: Sequence[BallCandidate], model: Sequence[BallObservation],
                         context: InteractionContext, predicted_xy: Array | None = None,
                         predicted_velocity_px_s: Array | None = None,
                         merge_radius_px: float = 18.0) -> list[BallCandidate]:
    """Fuse independent evidence while preserving per-factor interpretability."""
    pool = list(classical) + [_candidate_from_model(item) for item in model]
    scored: list[BallCandidate] = []
    for candidate in pool:
        point = candidate.observation.pixel_xy
        is_model = candidate.observation.source.startswith("racketvision")
        event_window = context.mode in (InteractionMode.BOUNCE_PENDING, InteractionMode.CONTACT_PENDING,
                                        InteractionMode.POST_EVENT)
        visual_scale = .85 if is_model and event_window else .35 if is_model else 1.0
        visual = candidate.observation.confidence * visual_scale
        if predicted_xy is None:
            dynamic = .5
        else:
            residual = point - np.asarray(predicted_xy, dtype=float)
            direction = None
            if predicted_velocity_px_s is not None and np.linalg.norm(predicted_velocity_px_s) > 1e-6:
                direction = np.asarray(predicted_velocity_px_s, dtype=float)
            elif candidate.observation.blur_vector_px is not None:
                direction = candidate.observation.blur_vector_px
            if direction is None or np.linalg.norm(direction) <= 1e-6:
                dynamic = math.exp(-0.5 * (np.linalg.norm(residual) / 95) ** 2)
            else:
                direction = direction / np.linalg.norm(direction)
                perpendicular = np.array([-direction[1], direction[0]])
                parallel_error = float(np.dot(residual, direction))
                cross_error = float(np.dot(residual, perpendicular))
                dynamic = math.exp(-0.5 * ((parallel_error / 145) ** 2 + (cross_error / 65) ** 2))
        body_factor = 1.0
        inside_body = any(_bbox_contains(body.bbox_xyxy, point, 10) and body.confidence >= .25
                          for body in context.bodies.values())
        racket_distances = [_distance_to_polygon(point, racket.pose.head_polygon_xy)
                            for racket in context.rackets.values() if racket.pose is not None]
        nearest_racket = min(racket_distances, default=math.inf)
        interaction = math.exp(-0.5 * (nearest_racket / 110) ** 2)
        protected = nearest_racket <= 130 or context.mode == InteractionMode.CONTACT_PENDING
        if inside_body and not protected:
            body_factor = .35
        source_prior = .45 if is_model else .90
        confidence = float(np.clip(body_factor * (0.48 * visual + 0.25 * dynamic
                                                   + 0.17 * interaction + 0.10 * source_prior), 0, 1))
        reasons = dict(candidate.reason_scores)
        reasons.update({"visual_evidence": visual, "dynamic_evidence": dynamic,
                        "body_suppression_factor": body_factor,
                        "interaction_evidence": interaction, "source_prior": source_prior})
        observation = replace(candidate.observation, confidence=confidence)
        scored.append(replace(candidate, observation=observation, reason_scores=reasons))

    # Merge near-duplicate sources, retaining the geometry of the strongest and
    # a noisy-OR confidence.  This is evidence fusion, not feature concatenation.
    merged: list[BallCandidate] = []
    for candidate in sorted(scored, key=lambda item: -item.observation.confidence):
        duplicate_index = next((index for index, item in enumerate(merged) if np.linalg.norm(
            item.observation.pixel_xy - candidate.observation.pixel_xy) <= merge_radius_px), None)
        if duplicate_index is None:
            merged.append(candidate)
            continue
        duplicate = merged.pop(duplicate_index)
        confidence = 1 - (1 - duplicate.observation.confidence) * (1 - candidate.observation.confidence)
        sources = sorted(set(duplicate.observation.source.split("+") + candidate.observation.source.split("+")))
        reasons = {**candidate.reason_scores, **duplicate.reason_scores,
                   "source_agreement": math.exp(-0.5 * (np.linalg.norm(
                       duplicate.observation.pixel_xy - candidate.observation.pixel_xy) / merge_radius_px) ** 2)}
        observation = replace(duplicate.observation, confidence=float(np.clip(confidence, 0, 1)),
                              source="+".join(sources))
        merged.append(replace(duplicate, observation=observation, reason_scores=reasons))
    return sorted(merged, key=lambda item: -item.observation.confidence)


def infer_interaction_mode(predicted_xy: Array | None, ball_velocity_px_s: Array | None,
                           context: InteractionContext, scene: PlanarSceneCalibration,
                           previous_mode: InteractionMode = InteractionMode.FREE_FLIGHT) -> tuple[InteractionMode, list[InteractionMetric]]:
    metrics = interaction_metrics(predicted_xy, ball_velocity_px_s, context)
    contact = False
    for metric in metrics:
        racket = context.rackets.get(metric.player_id)
        if racket is None or racket.pose is None:
            continue
        model_observation = racket.status == TrackStatus.OBSERVED and racket.confidence >= .35
        strict_proxy = (metric.distance_px <= 95 and metric.closing_speed_px_s >= 300
                        and metric.time_to_closest_s is not None and metric.time_to_closest_s <= .10)
        distance_limit = 145 if model_observation else 95
        model_approach = model_observation and (
            metric.closing_speed_px_s >= 80 or
            metric.time_to_closest_s is not None and metric.time_to_closest_s <= .14 or
            metric.swept_intersection)
        if metric.distance_px <= distance_limit and (model_approach or strict_proxy):
            contact = True
            break
    if contact:
        return InteractionMode.CONTACT_PENDING, metrics
    if predicted_xy is not None:
        x, y = np.asarray(predicted_xy, dtype=float)
        table_x = scene.table_polygon_xy[:, 0]
        near_surface = (table_x.min() - 30 <= x <= table_x.max() + 30
                        and scene.close_to_table_surface(np.array([x, y]), margin_px=35))
        velocity = np.zeros(2) if ball_velocity_px_s is None else np.asarray(ball_velocity_px_s)
        if near_surface and velocity[1] >= 60:
            return InteractionMode.BOUNCE_PENDING, metrics
    if previous_mode in (InteractionMode.CONTACT_PENDING, InteractionMode.BOUNCE_PENDING):
        return InteractionMode.POST_EVENT, metrics
    return InteractionMode.FREE_FLIGHT, metrics


def smooth_event_gaps(frames: Sequence[TrackFrame], modes: Sequence[InteractionMode],
                      scene: PlanarSceneCalibration, max_gap: int = 2,
                      window_frames: int = 9) -> tuple[list[TrackFrame], list[dict]]:
    """Backward-smooth short event gaps without relabelling them as observed."""
    if len(frames) != len(modes):
        raise ValueError("frames and modes must have equal length")
    output = list(frames)
    corrections: list[dict] = []
    i = 0
    while i < len(frames):
        if frames[i].status == TrackStatus.OBSERVED:
            i += 1
            continue
        start = i
        while i < len(frames) and frames[i].status != TrackStatus.OBSERVED:
            i += 1
        end = i
        gap = end - start
        if start == 0 or end >= len(frames) or gap > max_gap:
            continue
        local_modes = modes[max(0, start - window_frames // 2):min(len(modes), end + window_frames // 2 + 1)]
        if not any(mode != InteractionMode.FREE_FLIGHT for mode in local_modes):
            continue
        left = frames[start - 1].observed_xy
        right = frames[end].observed_xy
        if left is None or right is None:
            continue
        bounce = InteractionMode.BOUNCE_PENDING in local_modes
        for offset, index in enumerate(range(start, end), 1):
            alpha = offset / (gap + 1)
            point = (1 - alpha) * left + alpha * right
            method = "event_linear"
            if bounce:
                # Quadratic Bezier through a table-surface impact anchor.
                top_y, near_y = scene.surface_bounds_y(point[0])
                impact = np.array([point[0], np.clip(point[1], top_y, near_y)])
                point = (1 - alpha) ** 2 * left + 2 * (1 - alpha) * alpha * impact + alpha**2 * right
                method = "piecewise_quadratic_bounce"
            output[index] = replace(frames[index], status=TrackStatus.PREDICTED,
                                    observed_xy=None, predicted_xy=point,
                                    confidence=max(frames[index].confidence, .30))
            corrections.append({"frame": index, "x_px": float(point[0]), "y_px": float(point[1]),
                                "method": method, "original_status": frames[index].status.value})
    return output, corrections
