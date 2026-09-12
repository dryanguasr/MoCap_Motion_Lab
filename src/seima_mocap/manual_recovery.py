"""GUI-independent recovery, intervention journal and causal loss episodes."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import uuid

import numpy as np

from .ball_video_tracking import TrackFrame, TrackStatus
from .interaction_tracking import InteractionMode
from .video_io import atomic_json


@dataclass(frozen=True)
class ManualSeed:
    frame: int
    timestamp_s: float
    xy: tuple[float, float]
    kind: str = "visible"

    def __post_init__(self):
        if self.kind not in ("visible", "estimated_occluded") or not np.isfinite(self.xy).all() or len(self.xy) != 2:
            raise ValueError("Invalid manual seed")


class ManualRecovery:
    """A position-only prior. Only subsequent visual detections set velocity."""

    def __init__(self, interaction_tracker, timeout_s=.25):
        self.interaction_tracker = interaction_tracker
        self.timeout_s = timeout_s
        self.seed = None
        self.previous_candidates = []
        self.state = "idle"

    def reset(self):
        interaction = self.interaction_tracker
        interaction.tracker.reset_for_discontinuity()
        interaction.observed_history.clear()
        interaction.mode = InteractionMode.FREE_FLIGHT
        interaction.post_event_remaining = 0
        self.seed = None
        self.previous_candidates = []
        self.state = "idle"

    def begin(self, seed: ManualSeed):
        self.reset()
        self.seed = seed
        self.state = "pending"

    def step(self, timestamp_s, candidates):
        if self.state != "pending":
            raise ValueError("No pending manual seed")
        tracker = self.interaction_tracker.tracker
        dt = timestamp_s - self.seed.timestamp_s
        empty = TrackFrame(timestamp_s, TrackStatus.LOST, None, None, None, 0., 0)
        if dt <= 0:
            return empty
        if dt > self.timeout_s:
            self.state = "failed"
            return empty
        # Expand from positional uncertainty; never borrow pre-loss velocity.
        radius = (90. if self.seed.kind == "estimated_occluded" else 35.) + 6000*dt
        eligible = [c for c in candidates if c.observation.confidence >= .15
                    and abs(c.observation.timestamp_s - timestamp_s) < .002
                    and np.linalg.norm(c.observation.pixel_xy - self.seed.xy) <= radius
                    and c.reason_scores.get("motion", 0) >= .08]
        pairs = []
        uncertainty = 90. if self.seed.kind == "estimated_occluded" else 35.
        for candidate in eligible:
            for prior in self.previous_candidates:
                delta = timestamp_s - prior.observation.timestamp_s
                if not 0 < delta <= .14:
                    continue
                velocity = (candidate.observation.pixel_xy - prior.observation.pixel_xy) / delta
                speed = float(np.linalg.norm(velocity))
                projected_seed = prior.observation.pixel_xy - velocity*(prior.observation.timestamp_s-self.seed.timestamp_s)
                residual = float(np.linalg.norm(projected_seed-self.seed.xy))
                if 70 <= speed <= 6000 and residual <= uncertainty:
                    score = prior.observation.confidence + candidate.observation.confidence - residual/uncertainty
                    pairs.append((score, prior, candidate))
        if pairs:
            _, prior, selected = max(pairs, key=lambda item: item[0])
            result = tracker.initialize_from_visual_pair(prior, selected)
            self.interaction_tracker.observed_history = [(c.observation.timestamp_s, c.observation.pixel_xy.copy())
                                                         for c in (prior, selected)]
            self.state = "confirmed"
            return result
        self.previous_candidates = [c for c in self.previous_candidates if timestamp_s-c.observation.timestamp_s <= .14]
        self.previous_candidates.extend(eligible[:20])
        return empty


class LossEpisodes:
    def __init__(self, times, initialization_timeout=.5, start_frame=0):
        self.times = np.asarray(times)
        self.initialization_timeout = initialization_timeout
        self.first_missing = start_frame
        self.start_frame = start_frame
        self.last_observed_frame = None
        self.reported = False

    def step(self, index, track):
        if track.status == TrackStatus.OBSERVED:
            self.first_missing = None
            self.last_observed_frame = index
            self.reported = False
            return None
        if self.first_missing is None:
            self.first_missing = index
        confirmed_loss = track.status == TrackStatus.LOST and self.last_observed_frame is not None
        uninitialized = self.last_observed_frame is None and self.times[index]-self.times[self.start_frame] >= self.initialization_timeout
        if not self.reported and (confirmed_loss or uninitialized):
            self.reported = True
            return {"episode_start": self.first_missing, "trigger_frame": index,
                    "last_observed_frame": self.last_observed_frame,
                    "reason": "lost" if confirmed_loss else "initialization_timeout"}
        return None


class InterventionJournal:
    SCHEMA = "seima.ball-interventions.v1"

    def __init__(self, path: Path, identity: dict, times, source_frames, source_pts, width, height):
        self.path = Path(path)
        self.times = list(times)
        self.source_frames = list(source_frames)
        self.source_pts = list(source_pts)
        self.width, self.height = width, height
        self.payload = {"schema_version": self.SCHEMA, "identity": identity,
                        "actions": [], "history": [], "cursor": 0}
        if self.path.exists():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            if self.payload.get("schema_version") != self.SCHEMA or self.payload.get("identity") != identity:
                raise ValueError("Interventions belong to a different source, clip, geometry or cache version")
            for action in self.actions:
                self.validate(action)

    @property
    def actions(self):
        return self.payload["actions"]

    def validate(self, action):
        index = action["frame"]
        if not isinstance(index, int) or not 0 <= index < len(self.times):
            raise ValueError("Intervention frame outside clip")
        if action["action"] not in ("seed", "skip", "end"):
            raise ValueError("Unknown intervention action")
        if not 0 <= action["episode_start"] <= action["trigger_frame"] < len(self.times):
            raise ValueError("Invalid loss episode")
        if action["action"] != "seed" and index < action["episode_start"]:
            raise ValueError("El final del tramo debe estar despues de la perdida")
        if action["action"] == "seed":
            seed = ManualSeed(index, self.times[index], tuple(action["xy"]), action["kind"])
            x, y = seed.xy
            if not 0 <= x < self.width or not 0 <= y < self.height:
                raise ValueError("Seed outside original image")
        if (action.get("source_frame") != self.source_frames[index]
                or action.get("timestamp_s") != self.times[index]
                or action.get("source_pts_s") != self.source_pts[index]):
            raise ValueError("Intervention source-frame mapping mismatch")

    def append(self, action):
        if not isinstance(action.get("frame"), int) or not 0 <= action["frame"] < len(self.times):
            raise ValueError("Intervention frame outside clip")
        action = {**action, "id": uuid.uuid4().hex,
                  "source_frame": self.source_frames[action["frame"]],
                  "timestamp_s": self.times[action["frame"]], "source_pts_s": self.source_pts[action["frame"]]}
        self.validate(action)
        # Editing the past invalidates dependent later decisions; retain audit history.
        cutoff = min(action["frame"], action["episode_start"])
        retained = [a for a in self.actions if max(a["frame"], a["trigger_frame"]) < cutoff]
        self.payload["history"].append({"superseded": [a for a in self.actions if a not in retained], "replacement": action["id"]})
        self.payload["actions"] = retained + [action]
        self.payload["cursor"] = action["frame"]
        self.payload.pop("pending_request", None)
        self.save()
        return action

    def save(self, cursor=None):
        if cursor is not None:
            self.payload["cursor"] = cursor
        atomic_json(self.path, self.payload)


def display_to_source(x, y, scale, image_width, image_height, top=0):
    if scale <= 0 or not 0 <= x < image_width*scale or not top <= y < top+image_height*scale:
        return None
    return (float(x/scale), float((y-top)/scale))
