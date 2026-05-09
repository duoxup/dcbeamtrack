from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class RectGrid:
    x_nodes: np.ndarray
    y_nodes: np.ndarray

    def __post_init__(self) -> None:
        x_nodes = np.asarray(self.x_nodes, dtype=float).reshape(-1)
        y_nodes = np.asarray(self.y_nodes, dtype=float).reshape(-1)
        if x_nodes.size < 3 or y_nodes.size < 3:
            raise ValueError("RectGrid requires at least 3 nodes along each axis.")

        dxs = np.diff(x_nodes)
        dys = np.diff(y_nodes)
        if not np.all(dxs > 0.0) or not np.all(dys > 0.0):
            raise ValueError("Grid nodes must be strictly increasing.")
        if not np.allclose(dxs, dxs[0]) or not np.allclose(dys, dys[0]):
            raise ValueError("RectGrid requires uniform spacing.")
        if not np.isclose(dxs[0], dys[0]):
            raise ValueError("First-version FDM requires square cells: dx == dy.")

        object.__setattr__(self, "x_nodes", x_nodes)
        object.__setattr__(self, "y_nodes", y_nodes)

    @classmethod
    def from_size(
        cls,
        *,
        width: float,
        height: float,
        spacing: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> "RectGrid":
        if width <= 0.0 or height <= 0.0 or spacing <= 0.0:
            raise ValueError("width, height, and spacing must be positive.")
        nx = int(round(width / spacing))
        ny = int(round(height / spacing))
        if not np.isclose(nx * spacing, width) or not np.isclose(ny * spacing, height):
            raise ValueError("width and height must be integer multiples of spacing.")
        cx, cy = center
        x_nodes = np.linspace(cx - 0.5 * width, cx + 0.5 * width, nx + 1)
        y_nodes = np.linspace(cy - 0.5 * height, cy + 0.5 * height, ny + 1)
        return cls(x_nodes=x_nodes, y_nodes=y_nodes)

    @classmethod
    def from_bounds(
        cls,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        spacing: float,
    ) -> "RectGrid":
        return cls.from_size(
            width=float(xmax) - float(xmin),
            height=float(ymax) - float(ymin),
            spacing=spacing,
            center=(0.5 * (float(xmin) + float(xmax)), 0.5 * (float(ymin) + float(ymax))),
        )

    @property
    def dx(self) -> float:
        return float(self.x_nodes[1] - self.x_nodes[0])

    @property
    def dy(self) -> float:
        return float(self.y_nodes[1] - self.y_nodes[0])

    @property
    def spacing(self) -> float:
        return self.dx

    @property
    def nx(self) -> int:
        return int(self.x_nodes.size)

    @property
    def ny(self) -> int:
        return int(self.y_nodes.size)

    @property
    def width(self) -> float:
        return float(self.x_nodes[-1] - self.x_nodes[0])

    @property
    def height(self) -> float:
        return float(self.y_nodes[-1] - self.y_nodes[0])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            float(self.x_nodes[0]),
            float(self.x_nodes[-1]),
            float(self.y_nodes[0]),
            float(self.y_nodes[-1]),
        )

    @property
    def internal_shape(self) -> tuple[int, int]:
        return self.ny - 2, self.nx - 2

    @property
    def internal_size(self) -> int:
        return (self.nx - 2) * (self.ny - 2)

    @property
    def cell_x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_nodes[:-1] + self.x_nodes[1:])

    @property
    def cell_y_centers(self) -> np.ndarray:
        return 0.5 * (self.y_nodes[:-1] + self.y_nodes[1:])

    @property
    def cell_area(self) -> float:
        return self.dx * self.dy

    @property
    def cell_areas(self) -> np.ndarray:
        return np.full(self.num_cells, self.cell_area, dtype=float)

    @property
    def num_cells(self) -> int:
        return (self.nx - 1) * (self.ny - 1)

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        x_centers, y_centers = np.meshgrid(
            self.cell_x_centers, self.cell_y_centers, indexing="xy"
        )
        return x_centers.ravel(), y_centers.ravel()

    def inside_open(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return (
            (x > self.x_nodes[0])
            & (x < self.x_nodes[-1])
            & (y > self.y_nodes[0])
            & (y < self.y_nodes[-1])
        )
