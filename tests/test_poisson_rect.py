import numpy as np
from partdist import SliceDistribution

from dcbeamtrack.mesh import RectGrid
from dcbeamtrack.spacecharge import RectFdmSpaceChargeSolver


def test_rect_fdm_spacecharge_returns_finite_field_map():
    grid = RectGrid.from_size(width=2e-3, height=2e-3, spacing=0.5e-3)
    dist = SliceDistribution(
        z=0.0,
        x=np.array([0.0]),
        y=np.array([0.0]),
        px=np.array([0.0]),
        py=np.array([0.0]),
        pz=np.array([1.0e5]),
        t=np.array([0.0]),
        lam=np.array([1.0e-12]),
    )

    field = RectFdmSpaceChargeSolver(grid, deposition="NGP").solve(dist)
    ex, ey = field.field_at(np.array([0.0]), np.array([0.0]))

    assert field.potential.shape == (grid.ny, grid.nx)
    assert field.ex_nodes.shape == (grid.ny, grid.nx)
    assert field.ey_nodes.shape == (grid.ny, grid.nx)
    assert np.all(np.isfinite(field.potential))
    assert np.all(np.isfinite(field.ex_nodes))
    assert np.all(np.isfinite(field.ey_nodes))
    assert abs(ex[0]) < 1e-6
    assert abs(ey[0]) < 1e-6
