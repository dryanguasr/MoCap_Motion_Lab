"""Simple causal filters and derivative estimators for the first prototype."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EMAState:
    """Exponential moving average for scalar or vector signals."""

    alpha: float = 0.25
    value: np.ndarray | float | None = None

    def update(self, measurement: np.ndarray | float):
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")

        current = np.asarray(measurement, dtype=float)
        if self.value is None:
            filtered = current
        else:
            filtered = self.alpha * current + (1.0 - self.alpha) * np.asarray(
                self.value, dtype=float
            )
        self.value = filtered
        return filtered.copy()


class FilteredDerivative:
    """Filter a signal, differentiate it, and optionally filter the derivative.

    The implementation is deliberately simple and causal so students can
    understand every step before replacing it with more sophisticated filters.
    """

    def __init__(self, alpha_position: float = 0.25, alpha_velocity: float = 0.25):
        self.position_filter = EMAState(alpha_position)
        self.velocity_filter = EMAState(alpha_velocity)
        self.previous_position: np.ndarray | None = None
        self.previous_time: float | None = None

    def update(self, measurement, timestamp_s: float):
        position = np.asarray(self.position_filter.update(measurement), dtype=float)

        if self.previous_position is None or self.previous_time is None:
            velocity = np.zeros_like(position)
        else:
            dt = timestamp_s - self.previous_time
            if dt <= 1e-6:
                velocity = np.zeros_like(position)
            else:
                raw_velocity = (position - self.previous_position) / dt
                velocity = np.asarray(self.velocity_filter.update(raw_velocity), dtype=float)

        self.previous_position = position.copy()
        self.previous_time = float(timestamp_s)
        return position, velocity
