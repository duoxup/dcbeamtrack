from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import factorized

from dcbeamtrack.constants import EPS0
from dcbeamtrack.mesh import RectGrid

from .deposition import deposit_line_charge
from .poisson import build_laplacian_5pt, field_from_potential


@dataclass(frozen=True)
class FieldMap2D:
    x_nodes: np.ndarray
    y_nodes: np.ndarray
    potential: np.ndarray
    ex_nodes: np.ndarray
    ey_nodes: np.ndarray
    line_charge_internal: np.ndarray

    def field_at(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        interp_ex = RegularGridInterpolator(
            (self.y_nodes, self.x_nodes), self.ex_nodes, bounds_error=False, fill_value=0.0
        )
        interp_ey = RegularGridInterpolator(
            (self.y_nodes, self.x_nodes), self.ey_nodes, bounds_error=False, fill_value=0.0
        )
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        points = np.column_stack([y_arr, x_arr])
        return interp_ex(points), interp_ey(points)


class RectFdmSpaceChargeSolver:
    def __init__(self, grid: RectGrid, *, deposition: str = "CIC") -> None:
        self.grid = grid
        self.deposition = deposition
        matrix = build_laplacian_5pt(grid.nx, grid.ny)
        self._solve = factorized(matrix)

    def solve(self, distribution) -> FieldMap2D:
        if not hasattr(distribution, "z"):
            raise ValueError("RectFdmSpaceChargeSolver expects a z-slice distribution.")

        x = np.asarray(distribution.x, dtype=float)
        y = np.asarray(distribution.y, dtype=float)
        lam = np.asarray(distribution.lam, dtype=float)
        line_charge, _ = deposit_line_charge(x, y, lam, self.grid, method=self.deposition)
        phi_internal = self._solve(line_charge / EPS0)

        potential = np.zeros((self.grid.ny, self.grid.nx), dtype=float)
        potential[1:-1, 1:-1] = phi_internal.reshape(self.grid.internal_shape)
        ex_nodes, ey_nodes = field_from_potential(potential, self.grid.spacing)
        return FieldMap2D(
            x_nodes=self.grid.x_nodes,
            y_nodes=self.grid.y_nodes,
            potential=potential,
            ex_nodes=ex_nodes,
            ey_nodes=ey_nodes,
            line_charge_internal=line_charge.reshape(self.grid.internal_shape),
        )
