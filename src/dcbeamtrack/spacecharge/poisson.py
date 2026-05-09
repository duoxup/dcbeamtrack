from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_laplacian_5pt(nx_nodes: int, ny_nodes: int) -> sp.csc_matrix:
    """Build the positive 5-point Dirichlet Laplacian for internal nodes."""
    nx = int(nx_nodes) - 2
    ny = int(ny_nodes) - 2
    if nx < 1 or ny < 1:
        raise ValueError("At least one internal node is required along each axis.")
    n = nx * ny
    main_diag = np.full(n, 4.0)
    side_diag = np.full(n - 1, -1.0)
    if side_diag.size:
        side_diag[nx - 1 :: nx] = 0.0
    top_down_diag = np.full(n - nx, -1.0)
    return sp.diags(
        [top_down_diag, side_diag, main_diag, side_diag, top_down_diag],
        [-nx, -1, 0, 1, nx],
        shape=(n, n),
        format="csc",
    )


def field_from_potential(potential: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    dphi_dy, dphi_dx = np.gradient(potential, spacing)
    return -dphi_dx, -dphi_dy
