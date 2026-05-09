from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from dcbeamtrack.constants import EPS0


_WALL_HIT_KEY = "wall_hit"
_R_EPS_SQ_DEFAULT = 1.0e-36  # (1e-18)**2 — guard against on-axis source divergence


@njit(parallel=True, fastmath=True, cache=True)
def _analytic_mirror_charge_numba(
    coef: float,
    radius: float,
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    active_idx: np.ndarray,
    r_eps_sq: float,
):
    """Direct mirror-charge field summation for a grounded circular pipe.

    For each target i in active_idx, sum over each source j in active_idx (j != i):
    direct line-charge contribution at (x_j, y_j) plus opposite-sign image at the
    inverse point (radius**2 / |r_j|**2) * (x_j, y_j).

    Sources whose squared radius is below ``r_eps_sq`` are skipped (would put the
    image at infinity, which is a numerical singularity even though the analytic
    contribution vanishes).

    Returns (ex, ey) sized like x; entries outside active_idx remain zero.
    """
    n = x.size
    ex_out = np.zeros(n, np.float64)
    ey_out = np.zeros(n, np.float64)
    n_active = active_idx.size

    radius_sq = radius * radius

    for ia in prange(n_active):
        i = active_idx[ia]
        xi = x[i]
        yi = y[i]
        s_ex = 0.0
        s_ey = 0.0
        for ja in range(n_active):
            j = active_idx[ja]
            if j == i:
                continue
            xj = x[j]
            yj = y[j]
            qj = q[j]

            r_src_sq = xj * xj + yj * yj
            if r_src_sq < r_eps_sq:
                continue

            dx = xi - xj
            dy = yi - yj
            r2_src = dx * dx + dy * dy

            scale = radius_sq / r_src_sq
            xj_img = scale * xj
            yj_img = scale * yj
            dxi = xi - xj_img
            dyi = yi - yj_img
            r2_img = dxi * dxi + dyi * dyi

            s_ex += qj * (dx / r2_src - dxi / r2_img)
            s_ey += qj * (dy / r2_src - dyi / r2_img)

        ex_out[i] = coef * s_ex
        ey_out[i] = coef * s_ey

    return ex_out, ey_out


@dataclass(frozen=True)
class AnalyticMirrorChargeFieldMap:
    """Field cache for the analytic mirror-charge solver, valid only at source positions."""

    ex: np.ndarray
    ey: np.ndarray

    def field_at(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if x_arr.size != self.ex.size or y_arr.size != self.ey.size:
            raise NotImplementedError(
                "AnalyticMirrorChargeSpaceChargeSolver only supports field queries at "
                f"the source particle positions (n={self.ex.size}); off-source queries "
                "are not implemented."
            )
        return self.ex.copy(), self.ey.copy()


class AnalyticMirrorChargeSpaceChargeSolver:
    """Direct mirror-charge solver for a grounded circular pipe.

    No discretization grid: the per-pair mirror-charge formula is summed exactly
    via a numba-parallel O(N**2) kernel. Used as a long-term ground truth for the
    polar-grid CircularGreenSpaceChargeSolver.
    """

    def __init__(self, radius: float) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        self.radius = float(radius)
        self._coef = 1.0 / (2.0 * np.pi * EPS0)
        self._r_eps_sq = _R_EPS_SQ_DEFAULT

    def solve(self, distribution) -> AnalyticMirrorChargeFieldMap:
        x = np.asarray(distribution.x, dtype=np.float64)
        y = np.asarray(distribution.y, dtype=np.float64)
        lam = np.asarray(distribution.lam, dtype=np.float64)

        if _WALL_HIT_KEY in distribution.extra_quantity_keys:
            wall_hit = np.asarray(distribution.get_data(_WALL_HIT_KEY), dtype=bool)
        else:
            wall_hit = np.zeros(x.shape, dtype=bool)

        r_sq = x * x + y * y
        radius_sq = self.radius * self.radius
        active = (~wall_hit) & (r_sq < radius_sq)
        active_idx = np.nonzero(active)[0].astype(np.int64)

        if active_idx.size == 0:
            ex = np.zeros_like(x)
            ey = np.zeros_like(x)
        else:
            ex, ey = _analytic_mirror_charge_numba(
                self._coef,
                self.radius,
                x,
                y,
                lam,
                active_idx,
                self._r_eps_sq,
            )

        return AnalyticMirrorChargeFieldMap(ex=ex, ey=ey)
