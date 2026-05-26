"""External electric fields evaluated by the tracker at particle positions.

Mirrors :mod:`dcbeamtrack.fields.magnetic`: a small :class:`ElectricField`
protocol plus :class:`ZeroElectricField`, :class:`UniformElectricField`,
and :class:`RadialElectricField` for the common modeling cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ElectricField(Protocol):
    """External electric field sampled by the tracker at particle positions.

    Frame: laboratory, time-independent. Implementations must not mutate
    inputs and must broadcast naturally over any input shape.

    See :class:`ZeroElectricField`, :class:`UniformElectricField`, and
    :class:`RadialElectricField`.
    """

    def at(self, x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(Ex, Ey, Ez)`` in V/m at the requested positions."""


@dataclass(frozen=True)
class ZeroElectricField:
    """Identically zero external electric field."""

    def at(self, x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        zeros = np.zeros_like(x_arr)
        return zeros, zeros.copy(), zeros.copy()

    def summary(self) -> dict:
        return {"type": "ZeroElectricField"}


@dataclass(frozen=True)
class UniformElectricField:
    """Spatially uniform external electric field."""

    ex: float = 0.0
    ey: float = 0.0
    ez: float = 0.0

    def at(self, x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        return (
            np.full_like(x_arr, self.ex, dtype=float),
            np.full_like(x_arr, self.ey, dtype=float),
            np.full_like(x_arr, self.ez, dtype=float),
        )

    def summary(self) -> dict:
        return {
            "type": "UniformElectricField",
            "ex": float(self.ex),
            "ey": float(self.ey),
            "ez": float(self.ez),
        }


@dataclass(frozen=True)
class RadialElectricField:
    """Axisymmetric, z-independent radial electric field ``Er(r)``.

    The Cartesian projection is

    .. math::

        E_x(x, y) = E_r(r) \\cdot x / r, \\quad
        E_y(x, y) = E_r(r) \\cdot y / r, \\quad
        E_z = 0

    with ``r = sqrt(x^2 + y^2)``. Linear interpolation over the stored
    ``(r, Er)`` table; outside ``[0, r_max]`` the field uses
    ``fill_value``.

    Regularity on the axis requires ``Er(r=0) = 0`` (otherwise the
    Cartesian decomposition is ambiguous). If the table does not include
    ``r = 0``, a leading ``(0, 0)`` sample is prepended automatically.
    If the table includes ``r = 0`` with non-zero ``Er``, a ``ValueError``
    is raised.

    .. note::

       A strictly z-independent, axisymmetric, purely-radial vacuum field
       satisfies ``div(E) = (1/r) d(r * Er)/dr = 0`` only when
       ``Er ∝ 1/r``. Arbitrary ``Er(r)`` profiles supplied here are an
       engineering idealization, not a Maxwell-consistent free-space
       field — useful for parametric studies, not as a faithful model of
       a physical electrode geometry in vacuum.
    """

    r_m: np.ndarray
    er_v_per_m: np.ndarray
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        r = np.asarray(self.r_m, dtype=float).reshape(-1)
        er = np.asarray(self.er_v_per_m, dtype=float).reshape(-1)
        if r.size != er.size:
            raise ValueError("r_m and er_v_per_m must have the same length.")
        if r.size < 1:
            raise ValueError("At least one (r, Er) sample is required.")
        if np.any(r < 0.0):
            raise ValueError("r_m must be non-negative.")
        if r.size >= 2 and np.any(np.diff(r) <= 0.0):
            raise ValueError("r_m must be strictly increasing.")
        if r[0] == 0.0:
            if er[0] != 0.0:
                raise ValueError(
                    "Er(r=0) must be 0 by axisymmetry; "
                    f"got Er[0]={float(er[0])}."
                )
        else:
            r = np.concatenate([[0.0], r])
            er = np.concatenate([[0.0], er])
        object.__setattr__(self, "r_m", r)
        object.__setattr__(self, "er_v_per_m", er)

    @classmethod
    def from_arrays(
        cls,
        r,
        er,
        *,
        r_unit: str = "m",
        fill_value: float = 0.0,
    ) -> "RadialElectricField":
        r_arr = np.asarray(r, dtype=float)
        unit = r_unit.lower()
        if unit == "m":
            r_m = r_arr
        elif unit == "mm":
            r_m = r_arr * 1e-3
        else:
            raise ValueError("r_unit must be 'm' or 'mm'.")
        return cls(
            r_m=r_m,
            er_v_per_m=np.asarray(er, dtype=float),
            fill_value=fill_value,
        )

    def at(self, x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.broadcast_to(np.asarray(y, dtype=float), x_arr.shape)
        r = np.hypot(x_arr, y_arr)
        er = np.interp(
            r,
            self.r_m,
            self.er_v_per_m,
            left=float(self.fill_value),
            right=float(self.fill_value),
        )
        ex = np.zeros_like(x_arr)
        ey = np.zeros_like(x_arr)
        mask = r > 0.0
        if mask.any():
            inv_r = 1.0 / r[mask]
            ex[mask] = er[mask] * x_arr[mask] * inv_r
            ey[mask] = er[mask] * y_arr[mask] * inv_r
        ez = np.zeros_like(x_arr)
        return ex, ey, ez

    def summary(self) -> dict:
        r = np.asarray(self.r_m)
        er = np.asarray(self.er_v_per_m)
        return {
            "type": "RadialElectricField",
            "n_samples": int(r.size),
            "r_min_m": float(r.min()),
            "r_max_m": float(r.max()),
            "er_min_v_per_m": float(er.min()),
            "er_max_v_per_m": float(er.max()),
            "fill_value": float(self.fill_value),
        }
