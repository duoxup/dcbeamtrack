import numpy as np

from dcbeamtrack.mesh import RectGrid
from dcbeamtrack.spacecharge import deposit_line_charge


def test_deposition_conserves_internal_line_charge_for_center_particles():
    grid = RectGrid.from_size(width=2.0, height=2.0, spacing=0.5)
    x = np.array([0.0, 0.0])
    y = np.array([0.0, 0.0])
    lam = np.array([1.0, -0.25])

    for method in ("NGP", "CIC", "TSC"):
        deposited, _ = deposit_line_charge(x, y, lam, grid, method=method)
        assert np.isclose(np.sum(deposited), np.sum(lam))


def test_deposition_ignores_particles_outside_open_domain():
    grid = RectGrid.from_size(width=2.0, height=2.0, spacing=0.5)
    deposited, _ = deposit_line_charge(
        np.array([0.0, 1.2]),
        np.array([0.0, 0.0]),
        np.array([1.0, 5.0]),
        grid,
        method="CIC",
    )

    assert np.isclose(np.sum(deposited), 1.0)
