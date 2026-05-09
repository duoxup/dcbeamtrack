# tests/_helpers.py
"""Shared test utilities for circular-pipe space-charge tests."""

from __future__ import annotations

import numpy as np

from dcbeamtrack.constants import EPS0


def direct_mirror_charge_field(
    x: np.ndarray, y: np.ndarray, q: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reference O(N**2) mirror-charge field for a grounded circular pipe.

    Direct line-charge contribution plus a single image charge of opposite sign at
    the inverse point ``(radius**2 / r_j**2) * (x_j, y_j)``. Pure-Python loop, used
    as ground truth for both CircularGreenSpaceChargeSolver (on grid nodes only)
    and AnalyticMirrorChargeSpaceChargeSolver (everywhere).
    """
    coef = 1.0 / (2.0 * np.pi * EPS0)
    n = x.size
    ex = np.zeros(n)
    ey = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r2 = dx * dx + dy * dy
            ex[i] += coef * q[j] * dx / r2
            ey[i] += coef * q[j] * dy / r2

            r_src2 = x[j] * x[j] + y[j] * y[j]
            scale = (radius * radius) / r_src2
            xi_img = scale * x[j]
            yi_img = scale * y[j]
            dxi = x[i] - xi_img
            dyi = y[i] - yi_img
            r2_img = dxi * dxi + dyi * dyi
            ex[i] -= coef * q[j] * dxi / r2_img
            ey[i] -= coef * q[j] * dyi / r2_img
    return ex, ey
