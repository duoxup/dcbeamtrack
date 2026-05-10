import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

from dcbeamtrack.spacecharge.poisson import make_sparse_solve


_HAS_PYPARDISO = importlib.util.find_spec("pypardiso") is not None
_skip_if_no_pypardiso = pytest.mark.skipif(
    not _HAS_PYPARDISO, reason="pypardiso not installed"
)


def test_make_sparse_solve_rejects_invalid_backend():
    matrix = sp.eye(4, format="csc")
    with pytest.raises(ValueError):
        make_sparse_solve(matrix, backend="bogus")


@_skip_if_no_pypardiso
def test_sparse_solve_scipy_pypardiso_match_small():
    from dcbeamtrack.spacecharge.poisson import build_laplacian_5pt_masked

    mask = np.ones((10, 10), dtype=bool)
    mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
    matrix, _ = build_laplacian_5pt_masked((10, 10), mask)
    rhs = np.arange(int(mask.sum()), dtype=float)

    solve_scipy = make_sparse_solve(matrix, backend="scipy")
    solve_pard = make_sparse_solve(matrix, backend="pypardiso")
    np.testing.assert_allclose(solve_scipy(rhs), solve_pard(rhs), atol=1e-10)


@_skip_if_no_pypardiso
def test_rect_fdm_solvers_match_across_backends():
    from partdist import SliceDistribution

    from dcbeamtrack import (
        RectFdmCircularBoundarySolver,
        RectFdmSpaceChargeSolver,
        RectGrid,
    )
    from dcbeamtrack.tracking import velocity_to_momentum_evc

    grid = RectGrid.from_size(width=2.2e-3, height=2.2e-3, spacing=1.0e-4)
    n = 5
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.4e-3, 0.4e-3, size=n)
    y = rng.uniform(-0.4e-3, 0.4e-3, size=n)
    lam = np.full(n, 1.0e-12)
    px, py, pz = velocity_to_momentum_evc(np.zeros(n), np.zeros(n), np.zeros(n))
    dist = SliceDistribution(
        z=0.0, x=x, y=y, px=px, py=py, pz=pz, t=np.zeros(n), lam=lam
    )

    for solver_cls, kwargs in [
        (RectFdmSpaceChargeSolver, {}),
        (RectFdmCircularBoundarySolver, {"radius": 1.0e-3}),
    ]:
        fm_scipy = solver_cls(grid, backend="scipy", **kwargs).solve(dist)
        fm_pard = solver_cls(grid, backend="pypardiso", **kwargs).solve(dist)
        np.testing.assert_allclose(fm_scipy.potential, fm_pard.potential, atol=1e-10)
        np.testing.assert_allclose(fm_scipy.ex_nodes, fm_pard.ex_nodes, atol=1e-10)
        np.testing.assert_allclose(fm_scipy.ey_nodes, fm_pard.ey_nodes, atol=1e-10)
