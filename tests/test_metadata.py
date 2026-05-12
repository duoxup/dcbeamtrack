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
