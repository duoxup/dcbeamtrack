from __future__ import annotations

import numpy as np
from partdist import SliceDistribution

from dcbeamtrack.mesh import RectGrid
from dcbeamtrack.tracking import velocity_to_momentum_evc

from .transverse import TransverseDensity


def _infer_vz0(density: TransverseDensity, vz0: float | None) -> float:
    if vz0 is not None:
        return float(vz0)
    value = getattr(density, "vz", None)
    if value is None:
        raise ValueError("vz0 must be provided when the density object has no 'vz' attribute.")
    return float(value)


def _broadcast_particle_value(value, n: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    arr = arr.reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must be scalar or have length {n}.")
    return arr


def _integrate_density_on_rect_grid(
    density: TransverseDensity,
    grid: RectGrid,
    *,
    quad_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_centers_1d = grid.cell_x_centers
    y_centers_1d = grid.cell_y_centers
    xc, yc = np.meshgrid(x_centers_1d, y_centers_1d, indexing="xy")

    if quad_order < 1:
        raise ValueError("quad_order must be >= 1.")
    xi, wi = np.polynomial.legendre.leggauss(int(quad_order))
    jac = 0.25 * grid.dx * grid.dy
    qcells = np.zeros_like(xc, dtype=float)
    for i, xq in enumerate(xi):
        for j, yq in enumerate(xi):
            qcells += wi[i] * wi[j] * density.rho(
                xc + 0.5 * grid.dx * xq,
                yc + 0.5 * grid.dy * yq,
            )
    qcells *= jac
    return xc.ravel(), yc.ravel(), qcells.ravel()


def _integrate_density_on_cell_grid(
    density: TransverseDensity,
    grid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        x, y = grid.cell_centers()
        areas = np.asarray(grid.cell_areas, dtype=float).reshape(-1)
    except AttributeError as exc:
        raise TypeError("grid must provide cell_centers() and cell_areas for sampling.") from exc
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if not (x.size == y.size == areas.size):
        raise ValueError("grid cell centers and areas must have the same length.")
    return x, y, density.rho(x, y) * areas


def sample_slice_from_density_on_grid(
    density: TransverseDensity,
    *,
    grid,
    z0: float = 0.0,
    vz0: float | None = None,
    vx0=0.0,
    vy0=0.0,
    quad_order: int = 3,
    remove_small_charge: bool = True,
    min_lam_fraction: float = 1e-3,
    min_lam_absolute: float | None = None,
    renormalize: bool = True,
) -> SliceDistribution:
    """Discretize a transverse line-charge density into a SliceDistribution."""
    vz0 = _infer_vz0(density, vz0)
    if isinstance(grid, RectGrid):
        x_all, y_all, lam_all = _integrate_density_on_rect_grid(
            density, grid, quad_order=quad_order
        )
    else:
        x_all, y_all, lam_all = _integrate_density_on_cell_grid(density, grid)

    if x_all.size == 0:
        raise ValueError("grid has no cells.")

    abs_lam = np.abs(lam_all)
    max_abs = float(abs_lam.max()) if abs_lam.size else 0.0
    if remove_small_charge and max_abs > 0.0:
        keep = abs_lam >= max_abs * float(min_lam_fraction)
        if min_lam_absolute is not None:
            keep &= abs_lam >= float(min_lam_absolute)
    else:
        keep = np.ones_like(lam_all, dtype=bool)

    if not np.any(keep):
        keep[int(np.argmax(abs_lam))] = True

    x = x_all[keep]
    y = y_all[keep]
    lam = lam_all[keep]

    if renormalize:
        target = density.total_linear_charge
        if target is not None:
            current = float(np.sum(lam))
            if current == 0.0:
                raise RuntimeError("Cannot renormalize a zero-charge sampled distribution.")
            lam = lam * (float(target) / current)

    n = x.size
    vx = _broadcast_particle_value(vx0, n, "vx0")
    vy = _broadcast_particle_value(vy0, n, "vy0")
    vz = _broadcast_particle_value(vz0, n, "vz0")
    px, py, pz = velocity_to_momentum_evc(vx, vy, vz)

    return SliceDistribution(
        z=float(z0),
        x=x,
        y=y,
        px=px,
        py=py,
        pz=pz,
        t=np.zeros(n, dtype=float),
        lam=lam,
    )
