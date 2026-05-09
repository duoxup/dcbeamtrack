from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dcbeamtrack.mesh import RectGrid


class TransverseBoundary(Protocol):
    """Transverse wall used by the tracker to absorb particles."""

    def absorb(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray
    ) -> np.ndarray:
        """Clip wall-hit particles in place and return a bool mask of which were hit.

        The mask covers every particle currently outside the wall (idempotent: a
        particle already pinned to the wall will report True again). The tracker
        OR-accumulates these masks into a persistent ``wall_hit`` flag.
        """


@dataclass(frozen=True)
class UnboundedBoundary:
    """No transverse wall: absorption is a no-op."""

    def absorb(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray
    ) -> np.ndarray:
        return np.zeros(x.shape, dtype=bool)


@dataclass(frozen=True)
class CircularBoundary:
    """Grounded circular pipe of given radius centered at the origin."""

    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("CircularBoundary radius must be positive.")

    def absorb(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray
    ) -> np.ndarray:
        r2 = x * x + y * y
        hit = r2 >= self.radius * self.radius
        if np.any(hit):
            r = np.sqrt(r2[hit])
            scale = self.radius / r
            x[hit] = x[hit] * scale
            y[hit] = y[hit] * scale
            vx[hit] = 0.0
            vy[hit] = 0.0
        return hit


@dataclass(frozen=True)
class RectBoundary:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if not (self.xmax > self.xmin and self.ymax > self.ymin):
            raise ValueError("RectBoundary requires xmax > xmin and ymax > ymin.")

    @classmethod
    def from_grid(cls, grid: RectGrid) -> "RectBoundary":
        xmin, xmax, ymin, ymax = grid.bounds
        return cls(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    def absorb(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray
    ) -> np.ndarray:
        hit = (x <= self.xmin) | (x >= self.xmax) | (y <= self.ymin) | (y >= self.ymax)
        if np.any(hit):
            vx[hit] = 0.0
            vy[hit] = 0.0
            x[hit] = np.clip(x[hit], self.xmin, self.xmax)
            y[hit] = np.clip(y[hit], self.ymin, self.ymax)
        return hit
