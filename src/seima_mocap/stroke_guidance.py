"""Experimental causal pose prior. Not enabled in the professionals launcher.

The phase labels are hypotheses, not contact annotations. This first ablation
changes only candidate ranking: no hard crop, contact reset, new detections,
physical parameter changes, or signed blur endpoint selection.
"""
from dataclasses import dataclass, asdict, replace

import numpy as np

from .ball_video_tracking import TrackStatus


@dataclass(frozen=True)
class StrokePriorConfig:
    min_body_confidence: float = .25
    min_wrist_confidence: float = .20
    min_racket_confidence: float = .30
    max_motion_gap_s: float = .12
    max_normalized_speed: float = 8.
    forward_speed: float = .50
    backward_speed: float = -.15
    outgoing_seconds: float = .16
    outside_floor: float = .25
    front_offset: float = .12
    contact_radius_x: float = .28
    radius_y: float = .18


@dataclass(frozen=True)
class StrokeRegion:
    player: str
    phase: str
    center: tuple
    radii: tuple
    reliability: float
    source: str

    def distance(self, point):
        return float(np.linalg.norm((np.asarray(point)-self.center)/self.radii))


@dataclass(frozen=True)
class StrokePriorFrame:
    timestamp_s: float
    regions: tuple
    fallback: str
    outside_floor: float

    def contains(self, point):
        return any(r.distance(point) <= 1 for r in self.regions)

    def weight(self, point):
        if not self.regions:
            return 1.  # Unavailable pose leaves the baseline untouched.
        strength = max(r.reliability for r in self.regions)
        proximity = max(np.exp(-.5*max(0., r.distance(point)-1.)**2) for r in self.regions)
        return float(1-strength*(1-self.outside_floor)*(1-proximity))

    def apply(self, candidates):
        result = []
        for candidate in candidates:
            weight = self.weight(candidate.observation.pixel_xy)
            observation = replace(candidate.observation, confidence=candidate.observation.confidence*weight)
            reasons = {**candidate.reason_scores, "stroke_prior_weight": weight,
                       "confidence_before_stroke_prior": candidate.observation.confidence}
            result.append(replace(candidate, observation=observation, reason_scores=reasons))
        # Stable ties preserve baseline ordering when all weights equal one.
        return sorted(result, key=lambda c: -c.observation.confidence)

    def record(self):
        return {"timestamp_s": self.timestamp_s, "regions": [asdict(r) for r in self.regions],
                "fallback": self.fallback, "outside_floor": self.outside_floor}


class StrokeGuide:
    def __init__(self, table_polygon, config=StrokePriorConfig()):
        self.table = np.asarray(table_polygon, float)
        self.span = float(np.ptp(self.table[:, 0]))
        if self.span <= 0:
            raise ValueError("Table must have positive image width")
        self.center_x = float(self.table[:, 0].mean())
        self.config = config
        self.previous = {}
        self.swing_times = {}
        self.last_time = None

    def step(self, context):
        t, cfg = context.timestamp_s, self.config
        if self.last_time is not None and t <= self.last_time:
            raise ValueError("Stroke guide requires strictly increasing real timestamps")
        self.last_time = t
        regions, active = [], []
        for player in ("left", "right"):
            body = context.bodies.get(player)
            if body is None or body.right_wrist_xy is None or body.confidence < cfg.min_body_confidence:
                self.previous.pop(player, None)
                continue
            quality = body.confidence
            if body.landmark_confidence is not None:
                quality = min(quality, float(body.landmark_confidence[16]))
            if not np.isfinite(quality) or quality < cfg.min_wrist_confidence:
                self.previous.pop(player, None)
                continue
            point = body.right_wrist_xy.copy()
            source = "wrist"
            racket = context.rackets.get(player)
            if (racket is not None and racket.status == TrackStatus.OBSERVED and racket.center_xy is not None
                    and racket.confidence >= cfg.min_racket_confidence
                    and np.linalg.norm(racket.center_xy-point) < .35*self.span):
                point = racket.center_xy.copy()
                source = "observed_racket_near_wrist"
                quality = min(quality, racket.confidence)
            sign = 1. if self.center_x > point[0] else -1.
            phase, along = "uncertain", 0.
            old = self.previous.get(player)
            # Never infer a gesture from switching wrist/racket reference systems.
            if old is not None and 0 < t-old[0] <= cfg.max_motion_gap_s and old[2] == source:
                velocity = (point-old[1])/(t-old[0])
                if np.linalg.norm(velocity)/self.span <= cfg.max_normalized_speed:
                    along = float(sign*velocity[0]/self.span)
                    if along > cfg.forward_speed:
                        phase = "contact_hypothesis"
                        self.swing_times[player] = t
                    elif t-self.swing_times.get(player, -100) <= cfg.outgoing_seconds and along >= cfg.backward_speed:
                        phase = "outgoing_hypothesis"
                    elif along < cfg.backward_speed:
                        phase = "preparation_hypothesis"
            self.previous[player] = (t, point.copy(), source)
            center = point + [sign*cfg.front_offset*self.span, 0.]
            rx, ry = cfg.contact_radius_x*self.span, cfg.radius_y*self.span
            if phase == "outgoing_hypothesis":
                # Gesture-supported exit corridor, not incoming ball extrapolation.
                center[0] = (point[0]+self.center_x)/2
                rx = max(rx, abs(point[0]-self.center_x)/2+.15*self.span)
            reliability = min(.8, quality) * (1. if phase != "uncertain" else .35)
            region = StrokeRegion(player, phase, tuple(center), (rx, ry), reliability, source)
            regions.append(region)
            if phase in ("contact_hypothesis", "outgoing_hypothesis"):
                active.append(region)
        if active:
            selected, fallback = tuple(active), "soft_full_candidate_pool"
        elif regions:
            # Both player fronts plus the intervening corridor when phase is unclear.
            y = float(np.mean([r.center[1] for r in regions]))
            corridor = StrokeRegion("both", "uncertain_corridor", (self.center_x, y),
                                    (.65*self.span, .25*self.span), .20, "table_and_wrists")
            selected, fallback = (*regions, corridor), "expanded_uncertain_phase"
        else:
            selected, fallback = (), "no_pose_baseline_unchanged"
        return StrokePriorFrame(t, selected, fallback, cfg.outside_floor)
