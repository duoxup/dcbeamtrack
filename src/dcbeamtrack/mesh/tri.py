from __future__ import annotations

from typing import Tuple

import numpy as np

from .cell import CellGrid
from .rect import RectGrid


class TriGrid(CellGrid):
    """Triangular cell grid built by splitting a uniform rectangular grid."""

    def __init__(
        self,
        *,
        x_nodes,
        y_nodes,
        diagonal: str = "alternating",
    ) -> None:
        rect = RectGrid(x_nodes=np.asarray(x_nodes, dtype=float), y_nodes=np.asarray(y_nodes, dtype=float))
        diagonal = diagonal.lower()
        if diagonal not in {"slash", "backslash", "alternating"}:
            raise ValueError("diagonal must be 'slash', 'backslash', or 'alternating'.")

        polygons = []
        for j in range(rect.ny - 1):
            y0 = rect.y_nodes[j]
            y1 = rect.y_nodes[j + 1]
            for i in range(rect.nx - 1):
                x0 = rect.x_nodes[i]
                x1 = rect.x_nodes[i + 1]
                p00 = np.array([x0, y0])
                p10 = np.array([x1, y0])
                p11 = np.array([x1, y1])
                p01 = np.array([x0, y1])
                mode = diagonal
                if mode == "alternating":
                    mode = "slash" if (i + j) % 2 == 0 else "backslash"
                if mode == "slash":
                    polygons.append(np.array([p00, p10, p01], dtype=float))
                    polygons.append(np.array([p10, p11, p01], dtype=float))
                else:
                    polygons.append(np.array([p00, p10, p11], dtype=float))
                    polygons.append(np.array([p00, p11, p01], dtype=float))

        base = CellGrid.from_polygons(grid_type="tri", polygons=polygons)
        super().__init__(
            grid_type=base.grid_type,
            xc=base.xc,
            yc=base.yc,
            areas=base.areas,
            polygons=base.polygons,
        )
        object.__setattr__(self, "rect", rect)

    @classmethod
    def from_size(
        cls,
        *,
        width: float,
        height: float,
        spacing: float,
        center: Tuple[float, float] = (0.0, 0.0),
        diagonal: str = "alternating",
    ) -> "TriGrid":
        rect = RectGrid.from_size(width=width, height=height, spacing=spacing, center=center)
        return cls(x_nodes=rect.x_nodes, y_nodes=rect.y_nodes, diagonal=diagonal)

    @classmethod
    def from_bounds(
        cls,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        spacing: float,
        diagonal: str = "alternating",
    ) -> "TriGrid":
        rect = RectGrid.from_bounds(
            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, spacing=spacing
        )
        return cls(x_nodes=rect.x_nodes, y_nodes=rect.y_nodes, diagonal=diagonal)
