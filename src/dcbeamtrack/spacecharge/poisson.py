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


def build_laplacian_5pt_masked(
    grid_shape: tuple[int, int],
    interior_mask: np.ndarray,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Build the positive 5-point Dirichlet Laplacian over masked interior nodes.

    The mask defines which (i, j) nodes are unknowns; all other nodes are
    treated as Dirichlet 0 (rect edge OR outside circle OR any other 0-bc).
    Off-diagonal contributions to masked-out neighbors drop to zero (Dirichlet
    0 absorbed into the linear system implicitly).

    Args:
        grid_shape: (ny, nx) full-grid node count.
        interior_mask: bool array of shape (ny, nx).

    Returns:
        matrix: csc_matrix of shape (n_interior, n_interior).
        flat_indices: int64 array of length n_interior, giving the i*nx+j flat
            index in the (ny, nx) grid for each interior unknown. Used to
            scatter the solution back to the full grid.
    """
    ny, nx = grid_shape
    if interior_mask.shape != grid_shape:
        raise ValueError("interior_mask shape must match grid_shape.")
    if interior_mask.dtype != np.bool_:
        raise ValueError("interior_mask must be a boolean array.")

    flat_indices = np.flatnonzero(interior_mask.ravel()).astype(np.int64)
    n = flat_indices.size
    if n < 1:
        raise ValueError("at least one interior node required.")

    local_of_flat = np.full(ny * nx, -1, dtype=np.int64)
    local_of_flat[flat_indices] = np.arange(n, dtype=np.int64)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for m in range(n):
        flat = int(flat_indices[m])
        i, j = divmod(flat, nx)
        rows.append(m)
        cols.append(m)
        data.append(4.0)
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx:
                neighbor_flat = ni * nx + nj
                local = local_of_flat[neighbor_flat]
                if local >= 0:
                    rows.append(m)
                    cols.append(int(local))
                    data.append(-1.0)

    return (
        sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc(),
        flat_indices,
    )
