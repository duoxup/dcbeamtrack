import math

import numpy as np
import pytest

from dcbeamtrack.mesh import RectGrid


def test_rect_grid_summary():
    grid = RectGrid.from_size(width=3.0e-4, height=3.0e-4, spacing=3.0e-6)
    summary = grid.summary()
    assert summary["type"] == "RectGrid"
    assert summary["nx"] == grid.nx
    assert summary["ny"] == grid.ny
    assert math.isclose(summary["width"], 3.0e-4)
    assert math.isclose(summary["height"], 3.0e-4)
    assert math.isclose(summary["spacing"], 3.0e-6)
    # No numpy arrays or lists in the dict.
    for v in summary.values():
        assert not isinstance(v, (np.ndarray, list))


def test_unbounded_boundary_summary():
    from dcbeamtrack import UnboundedBoundary
    summary = UnboundedBoundary().summary()
    assert summary == {"type": "UnboundedBoundary"}


def test_circular_boundary_summary():
    from dcbeamtrack import CircularBoundary
    summary = CircularBoundary(radius=1.5e-4).summary()
    assert summary["type"] == "CircularBoundary"
    assert math.isclose(summary["radius"], 1.5e-4)


def test_rect_boundary_summary():
    from dcbeamtrack import RectBoundary
    summary = RectBoundary(xmin=-1.0, xmax=2.0, ymin=-0.5, ymax=1.5).summary()
    assert summary["type"] == "RectBoundary"
    assert math.isclose(summary["xmin"], -1.0)
    assert math.isclose(summary["xmax"], 2.0)
    assert math.isclose(summary["ymin"], -0.5)
    assert math.isclose(summary["ymax"], 1.5)


def test_uniform_magnetic_field_summary():
    from dcbeamtrack import UniformMagneticField
    summary = UniformMagneticField(bx=0.0, by=0.0, bz=0.6).summary()
    assert summary == {"type": "UniformMagneticField", "bx": 0.0, "by": 0.0, "bz": 0.6}


def test_axial_magnetic_field_summary():
    from dcbeamtrack import AxialMagneticField
    z_mm = np.linspace(-3.0, 33.0, 1001)
    bz_t = 0.35 * np.sin(2.0 * np.pi * z_mm / 4.93)
    field = AxialMagneticField.from_z_bz(z_mm, bz_t, z_unit="mm", include_radial=True)
    summary = field.summary()
    assert summary["type"] == "AxialMagneticField"
    assert summary["n_samples"] == 1001
    assert math.isclose(summary["z_min_m"], -3.0e-3)
    assert math.isclose(summary["z_max_m"], 33.0e-3)
    assert math.isclose(summary["bz_min_t"], float(bz_t.min()))
    assert math.isclose(summary["bz_max_t"], float(bz_t.max()))
    assert summary["include_radial"] is True
    assert summary["br_window"] == 9
    assert summary["br_poly_degree"] == 3
    # No raw arrays anywhere.
    for v in summary.values():
        assert not isinstance(v, (np.ndarray, list))


def test_rect_fdm_solver_summary():
    from dcbeamtrack import RectFdmSpaceChargeSolver
    grid = RectGrid.from_size(width=3.0e-4, height=3.0e-4, spacing=3.0e-6)
    solver = RectFdmSpaceChargeSolver(grid, deposition="CIC")
    summary = solver.summary()
    assert summary["type"] == "RectFdmSpaceChargeSolver"
    assert summary["deposition"] == "CIC"
    assert summary["backend"] == "auto"
    assert summary["grid"]["type"] == "RectGrid"
    assert summary["grid"]["nx"] == grid.nx
    assert summary["grid"]["ny"] == grid.ny


def test_rect_fdm_circular_solver_summary():
    from dcbeamtrack import RectFdmCircularBoundarySolver
    grid = RectGrid.from_size(width=3.0e-4, height=3.0e-4, spacing=3.0e-6)
    solver = RectFdmCircularBoundarySolver(grid, radius=1.0e-4, deposition="CIC")
    summary = solver.summary()
    assert summary["type"] == "RectFdmCircularBoundarySolver"
    assert math.isclose(summary["radius"], 1.0e-4)
    assert summary["deposition"] == "CIC"
    assert summary["backend"] == "auto"
    assert summary["grid"]["type"] == "RectGrid"
    assert summary["n_interior_nodes"] > 0


def test_circular_green_solver_summary():
    from dcbeamtrack import CircularGreenSpaceChargeSolver
    solver = CircularGreenSpaceChargeSolver(radius=1.5e-4, n_r=64, n_theta=180)
    summary = solver.summary()
    assert summary["type"] == "CircularGreenSpaceChargeSolver"
    assert math.isclose(summary["radius"], 1.5e-4)
    assert summary["n_r"] == 64
    assert summary["n_theta"] == 180


def test_analytic_mirror_solver_summary():
    from dcbeamtrack import AnalyticMirrorChargeSpaceChargeSolver
    solver = AnalyticMirrorChargeSpaceChargeSolver(radius=2.0e-4)
    summary = solver.summary()
    assert summary == {"type": "AnalyticMirrorChargeSpaceChargeSolver", "radius": 2.0e-4}
