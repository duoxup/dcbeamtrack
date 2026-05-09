from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class FieldMap2DProtocol(Protocol):
    """Interpolatable transverse electric field map."""

    def field_at(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return Ex, Ey [V/m] at the requested transverse coordinates."""


class SpaceChargeSolver(Protocol):
    """Space-charge solver interface consumed by the tracker."""

    def solve(self, distribution: object) -> FieldMap2DProtocol:
        """Solve the transverse space-charge field for a fixed-z slice."""
