"""Basic kinematics derived from 3D pose landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import FilteredDerivative
from .landmarks import JOINT_TRIPLETS


def angle_3d(a, b, c) -> float:
    """Return the smaller angle ABC in degrees using 3D Cartesian points."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    u = a - b
    v = c - b
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return float("nan")

    cosine = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


@dataclass
class JointEstimate:
    angle_deg: float
    angular_velocity_deg_s: float


class JointKinematicsEstimator:
    """Estimate filtered scalar joint angles and angular velocities."""

    def __init__(self, alpha: float = 0.25):
        self.filters = {
            name: FilteredDerivative(alpha_position=alpha, alpha_velocity=alpha)
            for name in JOINT_TRIPLETS
        }

    def update(self, positions: dict[int, np.ndarray], timestamp_s: float):
        estimates: dict[str, JointEstimate] = {}

        for name, (a_idx, b_idx, c_idx) in JOINT_TRIPLETS.items():
            if not all(idx in positions for idx in (a_idx, b_idx, c_idx)):
                continue

            angle = angle_3d(positions[a_idx], positions[b_idx], positions[c_idx])
            if not np.isfinite(angle):
                continue

            filtered_angle, angular_velocity = self.filters[name].update(angle, timestamp_s)
            estimates[name] = JointEstimate(
                angle_deg=float(np.asarray(filtered_angle)),
                angular_velocity_deg_s=float(np.asarray(angular_velocity)),
            )

        return estimates
