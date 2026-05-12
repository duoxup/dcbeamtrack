from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dcbeamtrack.mesh import RectGrid


class TransverseBoundary(Protocol):
    """Transverse wall used by the tracker to absorb particles.

    See ``dcbeamtrack.boundary`` for the built-in :class:`UnboundedBoundary`,
    :class:`CircularBoundary`, and :class:`RectBoundary` implementations.
    """

    def absorb(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray
    ) -> np.ndarray:
        """Clip wall-hit particles in place and return a bool hit mask.

        Parameters
        ----------
        x, y, vx, vy : numpy.ndarray
            All four with identical shape ``(n_particles,)``, float dtype.

        Returns
        -------
        numpy.ndarray
            Bool array of shape ``(n_particles,)``. ``True`` indicates the
            particle was outside (or on, depending on the wall convention)
            and was clipped on this call.

        Notes
        -----
        Implementations may mutate ``x``, ``y``, ``vx``, ``vy`` in place to
        clip particles to the wall and zero out transverse velocities — the
        built-in :class:`CircularBoundary` and :class:`RectBoundary` both
        do. The call is idempotent: a particle previously clipped will
        report ``True`` again if it still sits exactly on the wall. The
        tracker OR-accumulates these masks into a persistent ``wall_hit``
        flag, so repeated ``True`` s are harmless.
        """


@dataclass(frozen=True)
class UnboundedBoundary:
    """No transverse wall: absorption is a no-op."""

    def absorb(
        self, x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray
    ) -> np.ndarray:
        return np.zeros(x.shape, dtype=bool)

    def summary(self) -> dict:
        return {"type": "UnboundedBoundary"}


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

    def summary(self) -> dict:
        return {"type": "CircularBoundary", "radius": float(self.radius)}


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

    def summary(self) -> dict:
        return {
            "type": "RectBoundary",
            "xmin": float(self.xmin),
            "xmax": float(self.xmax),
            "ymin": float(self.ymin),
            "ymax": float(self.ymax),
        }
