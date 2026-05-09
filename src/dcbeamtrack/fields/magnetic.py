from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class MagneticField(Protocol):
    def at(self, x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Bx, By, Bz [T] at particle positions."""


def _local_polynomial_derivative(
    z: np.ndarray,
    values: np.ndarray,
    *,
    window: int,
    poly_degree: int,
) -> np.ndarray:
    """Derivative from the legacy stable local-polynomial fit."""
    z = np.asarray(z, dtype=float)
    values = np.asarray(values, dtype=float)
    if z.size < 3:
        return np.gradient(values, z)

    idx_sort = np.argsort(z)
    z_sorted = z[idx_sort]
    values_sorted = values[idx_sort]

    n_samples = z_sorted.size
    degree = min(int(poly_degree), n_samples - 1)
    if degree < 1:
        return np.gradient(values, z)

    fit_window = max(int(window), degree + 1)
    if fit_window % 2 == 0:
        fit_window += 1
    fit_window = min(fit_window, n_samples if n_samples % 2 == 1 else n_samples - 1)
    if fit_window < degree + 1:
        degree = fit_window - 1
    half_window = fit_window // 2

    deriv_sorted = np.empty_like(values_sorted)
    for i in range(n_samples):
        j0 = max(0, i - half_window)
        j1 = min(n_samples, j0 + fit_window)
        j0 = max(0, j1 - fit_window)

        x_local = z_sorted[j0:j1] - z_sorted[i]
        y_local = values_sorted[j0:j1]
        vandermonde = np.empty((x_local.size, degree + 1), dtype=float)
        vandermonde[:, 0] = 1.0
        for power in range(1, degree + 1):
            vandermonde[:, power] = vandermonde[:, power - 1] * x_local
        coeffs, *_ = np.linalg.lstsq(vandermonde, y_local, rcond=None)
        deriv_sorted[i] = coeffs[1]

    inv_idx = np.empty_like(idx_sort)
    inv_idx[idx_sort] = np.arange(n_samples)
    return deriv_sorted[inv_idx]


@dataclass(frozen=True)
class UniformMagneticField:
    bx: float = 0.0
    by: float = 0.0
    bz: float = 0.0

    def at(self, x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        return (
            np.full_like(x_arr, self.bx, dtype=float),
            np.full_like(x_arr, self.by, dtype=float),
            np.full_like(x_arr, self.bz, dtype=float),
        )


@dataclass(frozen=True)
class AxialMagneticField:
    z_m: np.ndarray
    bz_t: np.ndarray
    include_radial: bool = True
    br_window: int = 9
    br_poly_degree: int = 3

    def __post_init__(self) -> None:
        z_m = np.asarray(self.z_m, dtype=float).reshape(-1)
        bz_t = np.asarray(self.bz_t, dtype=float).reshape(-1)
        if z_m.size != bz_t.size:
            raise ValueError("z_m and bz_t must have the same length.")
        if z_m.size < 2:
            raise ValueError("At least two z/Bz samples are required.")
        if np.any(np.diff(z_m) <= 0.0):
            raise ValueError("z_m must be strictly increasing.")
        object.__setattr__(self, "z_m", z_m)
        object.__setattr__(self, "bz_t", bz_t)
        dbz_dz = _local_polynomial_derivative(
            z_m,
            bz_t,
            window=self.br_window,
            poly_degree=self.br_poly_degree,
        )
        object.__setattr__(self, "_dbz_dz", dbz_dz)
        object.__setattr__(self, "_br_coef", -0.5 * dbz_dz)

    @classmethod
    def from_z_bz(
        cls,
        z,
        bz,
        *,
        z_unit: str = "m",
        include_radial: bool = True,
        br_window: int = 9,
        br_poly_degree: int = 3,
    ) -> "AxialMagneticField":
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        unit = z_unit.lower()
        if unit == "m":
            z_m = z_arr
        elif unit == "mm":
            z_m = z_arr * 1e-3
        else:
            raise ValueError("z_unit must be 'm' or 'mm'.")
        return cls(
            z_m=z_m,
            bz_t=np.asarray(bz, dtype=float),
            include_radial=include_radial,
            br_window=br_window,
            br_poly_degree=br_poly_degree,
        )

    @classmethod
    def from_array(
        cls,
        data,
        *,
        z_unit: str = "mm",
        include_radial: bool = True,
        br_window: int = 9,
        br_poly_degree: int = 3,
    ) -> "AxialMagneticField":
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("Magnetic field data must have shape (N, 2): [z, Bz].")
        return cls.from_z_bz(
            arr[:, 0],
            arr[:, 1],
            z_unit=z_unit,
            include_radial=include_radial,
            br_window=br_window,
            br_poly_degree=br_poly_degree,
        )

    def bz(self, z) -> np.ndarray:
        return np.interp(z, self.z_m, self.bz_t)

    def br_coef(self, z) -> np.ndarray:
        """Return Br/r = -0.5 dBz/dz [T/m] from the paraxial solenoidal approximation."""
        return np.interp(z, self.z_m, self._br_coef)

    def at(self, x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        bz = np.broadcast_to(self.bz(z_arr), x_arr.shape).astype(float, copy=True)

        if not self.include_radial:
            return np.zeros_like(x_arr), np.zeros_like(x_arr), bz

        coef = np.broadcast_to(self.br_coef(z_arr), x_arr.shape)
        return coef * x_arr, coef * y_arr, bz
