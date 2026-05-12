from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from dcbeamtrack.constants import EPS0


_TWOPI = 2.0 * np.pi
_WALL_HIT_KEY = "wall_hit"


def _precompute_green_kernels(
    radius: float, gr: np.ndarray, gu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror-charge Green kernels for a grounded circular pipe.

    Returns ``(G_Er, G_Eu)`` of shape ``(n_r, n_r, n_theta)``, scaled by
    ``1/(2*pi*eps0)``. The self-term at ``j == jo, theta_k == 0`` is zeroed.
    """
    rj = gr[:, None, None]
    ro = gr[None, :, None]
    dth = gu[None, None, :]
    cosd = np.cos(dth)
    sind = np.sin(dth)

    c1 = radius**2 + (rj**2 * ro**2) / radius**2 - 2.0 * rj * ro * cosd
    c2 = rj**2 - 2.0 * rj * ro * cosd + ro**2
    eps = 1e-18
    c1 = np.where(np.abs(c1) < eps, eps, c1)
    c2 = np.where(np.abs(c2) < eps, eps, c2)

    # NOTE: legacy t2dcore.py had a bug in Eu (extra radius**2 factor on the c1
    # term, making the two terms dimensionally inconsistent). The correct
    # mirror-charge derivation gives Eu = ro*sind*(1/c1 - 1/c2).
    er = (rj - ro * cosd) / c2 - (rj * ro**2 - radius**2 * ro * cosd) / (c1 * radius**2)
    eu = (ro * sind) / c1 - (ro * sind) / c2

    coef = 1.0 / (2.0 * np.pi * EPS0)
    g_er = (coef * er).astype(np.float64, copy=False)
    g_eu = (coef * eu).astype(np.float64, copy=False)

    k_zero = int(np.argmin(np.abs(gu)))
    idx = np.arange(gr.size, dtype=np.int64)
    g_er[idx, idx, k_zero] = 0.0
    g_eu[idx, idx, k_zero] = 0.0

    return np.ascontiguousarray(g_er), np.ascontiguousarray(g_eu)


@njit(parallel=True, fastmath=True, cache=True)
def _pp_fast_numba(
    gr0: float,
    dr: float,
    n_r: int,
    gu0: float,
    dgu: float,
    n_theta: int,
    k_zero: int,
    g_er: np.ndarray,
    g_eu: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    q: np.ndarray,
    active_idx: np.ndarray,
):
    n = r.size
    er_out = np.zeros(n, np.float64)
    eu_out = np.zeros(n, np.float64)
    n_active = active_idx.size

    ring_src = np.empty(n_active, np.int64)
    for ia in prange(n_active):
        j = active_idx[ia]
        ir = int(round((r[j] - gr0) / dr))
        if ir < 0:
            ir = 0
        elif ir >= n_r:
            ir = n_r - 1
        ring_src[ia] = ir

    two_pi = 2.0 * np.pi
    for ia in prange(n_active):
        i = active_idx[ia]
        ir_t = int(round((r[i] - gr0) / dr))
        if ir_t < 0:
            ir_t = 0
        elif ir_t >= n_r:
            ir_t = n_r - 1

        ui = theta[i]
        s_er = 0.0
        s_eu = 0.0

        for ja in range(n_active):
            j = active_idx[ja]
            if j == i:
                continue
            ir_s = ring_src[ja]

            du = (theta[j] - ui) % two_pi
            f = (du - gu0) / dgu
            iu0 = int(np.floor(f))
            iu1 = iu0 + 1
            w1 = f - iu0
            w0 = 1.0 - w1
            iu0 %= n_theta
            iu1 %= n_theta

            if (ir_t == ir_s) and (iu0 == k_zero) and (iu1 == k_zero):
                iu1 = (k_zero + 1) % n_theta
                w0 = 0.0
                w1 = 1.0

            val_er = w0 * g_er[ir_t, ir_s, iu0] + w1 * g_er[ir_t, ir_s, iu1]
            val_eu = w0 * g_eu[ir_t, ir_s, iu0] + w1 * g_eu[ir_t, ir_s, iu1]
            s_er += val_er * q[j]
            s_eu += val_eu * q[j]

        er_out[i] = s_er
        eu_out[i] = s_eu

    return er_out, eu_out


@dataclass(frozen=True)
class CircularGreenFieldMap:
    """Field cache for the circular Green solver, valid only at source positions."""

    ex: np.ndarray
    ey: np.ndarray

    def field_at(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if x_arr.size != self.ex.size or y_arr.size != self.ey.size:
            raise NotImplementedError(
                "CircularGreenSpaceChargeSolver only supports field queries at the "
                f"source particle positions (n={self.ex.size}); off-source queries are "
                "not implemented."
            )
        return self.ex.copy(), self.ey.copy()


class CircularGreenSpaceChargeSolver:
    """Mirror-charge Green's-function solver for a grounded circular pipe."""

    def __init__(
        self,
        radius: float,
        *,
        n_r: int = 96,
        n_theta: int = 720,
        r_eps: float = 1e-9,
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if n_r < 4:
            raise ValueError("n_r must be at least 4.")
        if n_theta < 16:
            raise ValueError("n_theta must be at least 16.")
        if r_eps <= 0.0 or r_eps >= radius:
            raise ValueError("r_eps must satisfy 0 < r_eps < radius.")

        self.radius = float(radius)
        self.n_r = int(n_r)
        self.n_theta = int(n_theta)
        self.r_eps = float(r_eps)

        self.gr = np.linspace(self.r_eps, self.radius - self.r_eps, self.n_r, dtype=np.float64)
        self.gu = np.linspace(0.0, _TWOPI, self.n_theta, endpoint=False, dtype=np.float64)
        self._g_er, self._g_eu = _precompute_green_kernels(self.radius, self.gr, self.gu)
        self._gr0 = float(self.gr[0])
        self._dr = float(self.gr[1] - self.gr[0])
        self._gu0 = float(self.gu[0])
        self._dgu = float(self.gu[1] - self.gu[0])
        self._k_zero = int(np.argmin(np.abs(self.gu)))

    def solve(self, distribution) -> CircularGreenFieldMap:
        x = np.asarray(distribution.x, dtype=np.float64)
        y = np.asarray(distribution.y, dtype=np.float64)
        lam = np.asarray(distribution.lam, dtype=np.float64)

        r = np.hypot(x, y)
        theta = np.mod(np.arctan2(y, x), _TWOPI)

        if _WALL_HIT_KEY in distribution.extra_quantity_keys:
            wall_hit = np.asarray(distribution.get_data(_WALL_HIT_KEY), dtype=bool)
        else:
            wall_hit = np.zeros(x.shape, dtype=bool)

        active = (~wall_hit) & (r < self.radius)
        active_idx = np.nonzero(active)[0].astype(np.int64)

        if active_idx.size == 0:
            er = np.zeros_like(x)
            eu = np.zeros_like(x)
        else:
            er, eu = _pp_fast_numba(
                self._gr0,
                self._dr,
                self.n_r,
                self._gu0,
                self._dgu,
                self.n_theta,
                self._k_zero,
                self._g_er,
                self._g_eu,
                r,
                theta,
                lam,
                active_idx,
            )

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        ex = er * cos_t - eu * sin_t
        ey = er * sin_t + eu * cos_t
        return CircularGreenFieldMap(ex=ex, ey=ey)

    def summary(self) -> dict:
        return {
            "type": "CircularGreenSpaceChargeSolver",
            "radius": float(self.radius),
            "n_r": int(self.n_r),
            "n_theta": int(self.n_theta),
        }
