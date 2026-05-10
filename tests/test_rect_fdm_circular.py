import numpy as np
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


def test_rect_fdm_circular_x_axis_symmetry():
    radius = 1.0e-3
    grid = RectGrid.from_size(width=2.2e-3, height=2.2e-3, spacing=1.0e-4)
    solver = RectFdmCircularBoundarySolver(grid, radius=radius)

    x = np.array([0.3e-3, 0.3e-3])
    y = np.array([+0.2e-3, -0.2e-3])
    lam = np.array([1.0e-12, 1.0e-12])
    fm = solver.solve(_slice(x, y, lam=lam))

    ex_at, ey_at = fm.field_at(x, y)

    np.testing.assert_allclose(ex_at[0], ex_at[1], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(ey_at[0], -ey_at[1], rtol=1e-8, atol=1e-8)


def test_rect_fdm_circular_low_r_matches_analytic_loose():
    """Canary: detect global regression by comparing to the analytic ground
    truth in the well-resolved interior (r < 0.5*radius). 10% threshold is
    engineering headroom for staircase boundary + bilinear interp + grid
    discretization noise — only fires on global formula regressions, not on
    routine numerical noise.
    """
    from dcbeamtrack.spacecharge.analytic_mirror_charge import (
        AnalyticMirrorChargeSpaceChargeSolver,
    )

    radius = 1.0e-3
    grid = RectGrid.from_size(width=2.2e-3, height=2.2e-3, spacing=2.0e-5)
    rng = np.random.default_rng(7)
    n = 8
    r = rng.uniform(0.05 * radius, 0.5 * radius, size=n)
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    lam = rng.uniform(0.5e-12, 1.5e-12, size=n)
    dist = _slice(x, y, lam=lam)

    fdm_solver = RectFdmCircularBoundarySolver(grid, radius=radius)
    fm_fdm = fdm_solver.solve(dist)
    ex_fdm, ey_fdm = fm_fdm.field_at(x, y)

    fm_ref = AnalyticMirrorChargeSpaceChargeSolver(radius=radius).solve(dist)
    ex_ref, ey_ref = fm_ref.ex, fm_ref.ey

    scale = max(np.max(np.abs(ex_ref)), np.max(np.abs(ey_ref)))
    rel_ex = np.max(np.abs(ex_fdm - ex_ref)) / scale
    rel_ey = np.max(np.abs(ey_fdm - ey_ref)) / scale
    assert rel_ex <= 0.10, f"Ex max rel error {rel_ex:.3e} > 10%"
    assert rel_ey <= 0.10, f"Ey max rel error {rel_ey:.3e} > 10%"
