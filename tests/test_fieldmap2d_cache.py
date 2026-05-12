import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dcbeamtrack.spacecharge.solver import FieldMap2D


def _trivial_fieldmap() -> FieldMap2D:
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 5)
    nx, ny = x.size, y.size
    return FieldMap2D(
        x_nodes=x,
        y_nodes=y,
        potential=np.zeros((ny, nx)),
        ex_nodes=np.ones((ny, nx)),
        ey_nodes=2.0 * np.ones((ny, nx)),
        line_charge_internal=np.zeros((ny - 2, nx - 2)),
    )


def test_field_at_returns_expected_values():
    fm = _trivial_fieldmap()
    ex, ey = fm.field_at([0.0, 0.3], [0.0, -0.2])
    np.testing.assert_allclose(ex, [1.0, 1.0])
    np.testing.assert_allclose(ey, [2.0, 2.0])


def test_field_at_reuses_cached_interpolators():
    fm = _trivial_fieldmap()
    # The cached attrs exist and are RegularGridInterpolator instances.
    assert isinstance(fm._interp_ex, RegularGridInterpolator)
    assert isinstance(fm._interp_ey, RegularGridInterpolator)

    # Identity is preserved across calls — interpolators are not rebuilt.
    ex0 = fm._interp_ex
    ey0 = fm._interp_ey
    for _ in range(3):
        fm.field_at(np.array([0.1]), np.array([0.1]))
    assert fm._interp_ex is ex0
    assert fm._interp_ey is ey0
