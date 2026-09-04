"""Conditional 2-D inverse dynamics; NOT validated forces or anatomical moments.

Coordinates are image x (right), z (up). Positive scalar moment is x*Fz-z*Fx.
All moments act on the distal segment. Inputs/outputs use SI units.
"""
from __future__ import annotations

import numpy as np

G = 9.81


def cross2(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def local_polynomial(time, values, valid, window_s=0.4, degree=3):
    """Centered fits using actual timestamps; never bridge rejected observations.

    Returns position, velocity, acceleration and full-window acceptance. Endpoints
    without a complete window are NaN, not extrapolated. Window is rounded to an
    odd frame count using median dt; each fit uses the original, possibly VFR, dt.
    """
    time, values = np.asarray(time), np.asarray(values, dtype=float)
    if len(time) != len(values) or np.any(np.diff(time) <= 0):
        raise ValueError("Timestamps must be strictly increasing and match values")
    half = max(2, int(round(window_s / np.median(np.diff(time)) / 2)))
    outputs = [np.full_like(values, np.nan) for _ in range(3)]
    accepted = np.zeros(len(time), dtype=bool)
    for i in range(half, len(time) - half):
        sl = slice(i - half, i + half + 1)
        if not np.all(valid[sl]) or not np.all(np.isfinite(values[sl])):
            continue
        dt = time[sl] - time[i]
        # Missing/abnormally spaced timestamps must not be treated as continuous.
        if np.max(np.diff(time[sl])) > 1.75 * np.median(np.diff(time)):
            continue
        design = np.vander(dt, degree + 1, increasing=True)
        coeff = np.linalg.lstsq(design, values[sl].reshape(len(dt), -1), rcond=None)[0]
        for order, factor in enumerate((1, 1, 2)):
            outputs[order][i] = (coeff[order] * factor).reshape(values.shape[1:])
        accepted[i] = True
    return (*outputs, accepted)


def whole_body_wrench(masses, coms, accelerations, inertias, angular_accelerations):
    """Required external floor force and moment rate about system COM.

    Other external forces are assumed zero. This is a necessary balance, not a
    measurement. masses/inertias: S; coms/accelerations: (..., S, 2).
    """
    masses = np.asarray(masses)
    center = np.sum(coms * masses[..., None], axis=-2) / masses.sum()
    center_acc = np.sum(accelerations * masses[..., None], axis=-2) / masses.sum()
    force = masses.sum() * (center_acc + np.array([0.0, G]))
    hdot = np.sum(
        np.asarray(inertias) * angular_accelerations
        + cross2(coms - center[..., None, :], accelerations * masses[..., None]),
        axis=-1,
    )
    return center, center_acc, force, hdot


def double_support(center, force, hdot, cop_left, cop_right, min_lever_m=0.06):
    """Assume parallel GRFs and prescribed COPs, with zero free foot moments.

    Solve one load fraction from angular balance. Negative loads, near-coincident
    projected supports, or non-positive total vertical force are inadmissible.
    Fractions are NEVER clipped to manufacture a plausible solution.
    """
    denominator = cross2(cop_left - cop_right, force)
    lever = np.abs(denominator) / np.maximum(np.linalg.norm(force, axis=-1), 1e-12)
    fraction = np.divide(
        hdot - cross2(cop_right - center, force), denominator,
        out=np.full_like(np.asarray(hdot, dtype=float), np.nan),
        where=np.abs(denominator) > 1e-9,
    )
    admissible = (lever >= min_lever_m) & (fraction >= 0) & (fraction <= 1) & (force[..., 1] > 0)
    left = force * fraction[..., None]
    right = force * (1 - fraction[..., None])
    residual = cross2(cop_left - center, left) + cross2(cop_right - center, right) - hdot
    return left, right, fraction, admissible, residual, lever


def segment_proximal_wrench(mass, com, acceleration, inertia, alpha,
                            distal_point, distal_force, distal_moment, proximal_point):
    """Newton-Euler recursion for one rigid planar segment."""
    proximal_force = mass * (acceleration + np.array([0.0, G])) - distal_force
    proximal_moment = (
        inertia * alpha - distal_moment
        - cross2(distal_point - com, distal_force)
        - cross2(proximal_point - com, proximal_force)
    )
    return proximal_force, proximal_moment
