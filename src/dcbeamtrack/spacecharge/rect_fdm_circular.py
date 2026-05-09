from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import factorized

from dcbeamtrack.constants import EPS0
from dcbeamtrack.mesh import RectGrid

from .deposition import deposit_line_charge
from .poisson import build_laplacian_5pt_masked, field_from_potential
from .solver import FieldMap2D


class RectFdmCircularBoundarySolver:
    """Rect FDM grid + staircase circular Dirichlet boundary solver.

    First-version mask: a node (i, j) is interior iff ``r_node < radius`` AND
    the node is not on the rect edge. The reduced sparse system is built over
    interior nodes only; the LU factorization is computed once at __init__.

    Particle wall behavior is the caller's responsibility (use
    CircularBoundary(radius) for the matched physical wall).
    """

    def __init__(
        self,
        grid: RectGrid,
        radius: float,
        *,
        deposition: str = "CIC",
    ) -> None:
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        margin = min(grid.width, grid.height) / 2.0 - grid.spacing
        if radius > margin:
            raise ValueError(
                "radius must leave at least one grid spacing margin to the rect edge: "
                f"radius={radius} > min(w,h)/2 - spacing = {margin}."
            )

        self.grid = grid
        self.radius = float(radius)
        self.deposition = deposition

        yy, xx = np.meshgrid(grid.y_nodes, grid.x_nodes, indexing="ij")
        r_node = np.hypot(xx, yy)
        edge_mask = np.zeros_like(r_node, dtype=bool)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
        interior_mask = (r_node < self.radius) & (~edge_mask)

        matrix, _ = build_laplacian_5pt_masked(
            (grid.ny, grid.nx), interior_mask
        )
        self._interior_mask = interior_mask
        self._solve = factorized(matrix)

    def solve(self, distribution) -> FieldMap2D:
        if not hasattr(distribution, "z"):
            raise ValueError(
                "RectFdmCircularBoundarySolver expects a z-slice distribution."
            )

        x = np.asarray(distribution.x, dtype=float)
        y = np.asarray(distribution.y, dtype=float)
        lam = np.asarray(distribution.lam, dtype=float)
        line_charge_internal, _ = deposit_line_charge(
            x, y, lam, self.grid, method=self.deposition
        )

        # deposit_line_charge returns the rect-internal flat array
        # ((nx-2)*(ny-2),). Reshape and zero-pad to the full (ny, nx) grid,
        # then take the masked-interior subset for the new linear system.
        line_charge_full = np.zeros((self.grid.ny, self.grid.nx), dtype=float)
        line_charge_full[1:-1, 1:-1] = line_charge_internal.reshape(
            self.grid.internal_shape
        )
        rhs = line_charge_full[self._interior_mask] / EPS0

        phi_interior = self._solve(rhs)

        potential = np.zeros((self.grid.ny, self.grid.nx), dtype=float)
        potential[self._interior_mask] = phi_interior
        ex_nodes, ey_nodes = field_from_potential(potential, self.grid.spacing)

        return FieldMap2D(
            x_nodes=self.grid.x_nodes,
            y_nodes=self.grid.y_nodes,
            potential=potential,
            ex_nodes=ex_nodes,
            ey_nodes=ey_nodes,
            # 1-D (n_interior,) — differs from RectFdmSpaceChargeSolver's (ny-2, nx-2)
            line_charge_internal=line_charge_full[self._interior_mask],
        )
