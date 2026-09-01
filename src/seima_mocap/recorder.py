"""CSV recording helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .landmarks import LANDMARK_NAMES


class SessionRecorder:
    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._landmark_file = (self.session_dir / "landmarks.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._joint_file = (self.session_dir / "joints.csv").open(
            "w", newline="", encoding="utf-8"
        )

        self.landmark_writer = csv.writer(self._landmark_file)
        self.joint_writer = csv.writer(self._joint_file)

        self.landmark_writer.writerow(
            [
                "timestamp_s",
                "landmark_index",
                "landmark_name",
                "visibility",
                "x_m",
                "y_m",
                "z_m",
                "vx_m_s",
                "vy_m_s",
                "vz_m_s",
                "speed_m_s",
            ]
        )
        self.joint_writer.writerow(
            ["timestamp_s", "joint", "angle_deg", "angular_velocity_deg_s"]
        )

    def write_landmark(
        self,
        timestamp_s: float,
        index: int,
        visibility: float,
        position: np.ndarray,
        velocity: np.ndarray,
    ):
        position = np.asarray(position, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        self.landmark_writer.writerow(
            [
                f"{timestamp_s:.6f}",
                index,
                LANDMARK_NAMES[index],
                f"{visibility:.5f}",
                *[f"{x:.8f}" for x in position],
                *[f"{x:.8f}" for x in velocity],
                f"{np.linalg.norm(velocity):.8f}",
            ]
        )

    def write_joint(self, timestamp_s: float, name: str, estimate):
        self.joint_writer.writerow(
            [
                f"{timestamp_s:.6f}",
                name,
                f"{estimate.angle_deg:.6f}",
                f"{estimate.angular_velocity_deg_s:.6f}",
            ]
        )

    def close(self):
        self._landmark_file.close()
        self._joint_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
