from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def polygon_area_and_centroid(vertices: np.ndarray) -> tuple[float, float, float]:
    poly = np.asarray(vertices, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("vertices must have shape (N, 2) with N >= 3.")
    x = poly[:, 0]
    y = poly[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = 0.5 * float(np.sum(cross))
    if area == 0.0:
        raise ValueError("Degenerate polygon with zero signed area.")
    cx = float(np.sum((x + x_next) * cross) / (6.0 * area))
    cy = float(np.sum((y + y_next) * cross) / (6.0 * area))
    return area, cx, cy


@dataclass(frozen=True)
class CellGrid:
    grid_type: str
    xc: np.ndarray
    yc: np.ndarray
    areas: np.ndarray
    polygons: tuple[np.ndarray, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        xc = np.asarray(self.xc, dtype=float).reshape(-1)
        yc = np.asarray(self.yc, dtype=float).reshape(-1)
        areas = np.asarray(self.areas, dtype=float).reshape(-1)
        if not (xc.size == yc.size == areas.size):
            raise ValueError("xc, yc, and areas must have the same length.")
        if np.any(areas <= 0.0):
            raise ValueError("All cell areas must be positive.")
        polygons = tuple(np.asarray(poly, dtype=float) for poly in self.polygons)
        if polygons and len(polygons) != xc.size:
            raise ValueError("polygons length must match number of cells.")
        object.__setattr__(self, "xc", xc)
        object.__setattr__(self, "yc", yc)
        object.__setattr__(self, "areas", areas)
        object.__setattr__(self, "polygons", polygons)

    @classmethod
    def from_polygons(
        cls,
        *,
        grid_type: str,
        polygons: Iterable[np.ndarray],
    ) -> "CellGrid":
        polys = tuple(np.asarray(poly, dtype=float) for poly in polygons)
        centers = []
        areas = []
        for poly in polys:
            area, cx, cy = polygon_area_and_centroid(poly)
            centers.append((cx, cy))
            areas.append(abs(area))
        if not polys:
            raise ValueError("At least one polygon is required.")
        centers_arr = np.asarray(centers, dtype=float)
        return cls(
            grid_type=grid_type,
            xc=centers_arr[:, 0],
            yc=centers_arr[:, 1],
            areas=np.asarray(areas, dtype=float),
            polygons=polys,
        )

    @property
    def num_cells(self) -> int:
        return int(self.xc.size)

    @property
    def cell_areas(self) -> np.ndarray:
        return self.areas

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        return self.xc.copy(), self.yc.copy()
