import numpy as np
import pytest
from partdist import SliceDistribution

from dcbeamtrack import RectGrid
from dcbeamtrack.spacecharge.rect_fdm_circular import (
    RectFdmCircularBoundarySolver,
)
from dcbeamtrack.spacecharge.solver import FieldMap2D
from dcbeamtrack.tracking import velocity_to_momentum_evc


def _slice(x, y, lam=None) -> SliceDistribution:
    n = x.size
    vx = vy = vz = np.zeros(n)
    px, py, pz = velocity_to_momentum_evc(vx, vy, vz)
    if lam is None:
        lam = np.full(n, 1.0e-12)
    return SliceDistribution(
        z=0.0,
        x=x,
        y=y,
        px=px,
        py=py,
        pz=pz,
        t=np.zeros(n),
        lam=lam,
    )


def test_rect_fdm_circular_solve_smoke():
    radius = 1.0e-3
    grid = RectGrid.from_size(width=2.2e-3, height=2.2e-3, spacing=1.0e-4)
    solver = RectFdmCircularBoundarySolver(grid, radius=radius)

    # Single particle near origin
    x = np.array([0.0])
    y = np.array([0.0])
    lam = np.array([1.0e-12])
    fm = solver.solve(_slice(x, y, lam=lam))

    # Returned object shape and type
    assert isinstance(fm, FieldMap2D)
    assert fm.potential.shape == (grid.ny, grid.nx)
    assert fm.ex_nodes.shape == (grid.ny, grid.nx)
    assert fm.ey_nodes.shape == (grid.ny, grid.nx)

    # Strict zero outside the circle
    yy, xx = np.meshgrid(grid.y_nodes, grid.x_nodes, indexing="ij")
    r_node = np.hypot(xx, yy)
    outside = r_node >= radius
    assert np.all(fm.potential[outside] == 0.0)

    # Non-trivial potential inside (at the geometric center)
    center_iy = grid.ny // 2
    center_ix = grid.nx // 2
    assert fm.potential[center_iy, center_ix] != 0.0
