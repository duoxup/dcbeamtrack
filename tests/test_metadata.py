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
