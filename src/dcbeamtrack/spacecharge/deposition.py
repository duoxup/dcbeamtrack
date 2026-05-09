from __future__ import annotations

import numpy as np

from dcbeamtrack.mesh import RectGrid


def _as_1d_float(a, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return arr


def deposit_line_charge(
    x,
    y,
    lam,
    grid: RectGrid,
    *,
    method: str = "CIC",
) -> tuple[np.ndarray, np.ndarray]:
    """Deposit macroparticle line charge [C/m] onto internal FDM nodes."""
    x = _as_1d_float(x, "x")
    y = _as_1d_float(y, "y")
    lam = _as_1d_float(lam, "lam")
    if not (x.size == y.size == lam.size):
        raise ValueError("x, y, and lam must have the same length.")

    b = np.zeros(grid.internal_size, dtype=float)
    node_index = np.zeros((x.size, 2), dtype=int)
    inside = grid.inside_open(x, y)
    if not np.any(inside):
        return b, node_index

    px = x[inside]
    py = y[inside]
    q = lam[inside]
    original = np.nonzero(inside)[0]
    fx = (px - grid.x_nodes[0]) / grid.spacing
    fy = (py - grid.y_nodes[0]) / grid.spacing
    nx_internal = grid.nx - 2

    method_key = method.upper()
    if method_key == "NGP":
        ix = np.round(fx).astype(int)
        iy = np.round(fy).astype(int)
        valid = (ix > 0) & (ix < grid.nx - 1) & (iy > 0) & (iy < grid.ny - 1)
        if np.any(valid):
            ixv = ix[valid]
            iyv = iy[valid]
            idx = original[valid]
            node_index[idx, 0] = ixv
            node_index[idx, 1] = iyv
            flat = (iyv - 1) * nx_internal + (ixv - 1)
            np.add.at(b, flat, q[valid])
        return b, node_index

    if method_key == "CIC":
        ix0 = np.floor(fx).astype(int)
        iy0 = np.floor(fy).astype(int)
        wx = fx - ix0
        wy = fy - iy0
        offsets = (
            (0, 0, (1.0 - wx) * (1.0 - wy)),
            (1, 0, wx * (1.0 - wy)),
            (0, 1, (1.0 - wx) * wy),
            (1, 1, wx * wy),
        )
        for dx, dy, weight in offsets:
            ixn = ix0 + dx
            iyn = iy0 + dy
            valid = (ixn > 0) & (ixn < grid.nx - 1) & (iyn > 0) & (iyn < grid.ny - 1)
            if not np.any(valid):
                continue
            ixv = ixn[valid]
            iyv = iyn[valid]
            idx = original[valid]
            node_index[idx, 0] = ixv
            node_index[idx, 1] = iyv
            flat = (iyv - 1) * nx_internal + (ixv - 1)
            np.add.at(b, flat, q[valid] * weight[valid])
        return b, node_index

    if method_key == "TSC":
        i0 = np.round(fx).astype(int)
        j0 = np.round(fy).astype(int)
        ix_candidates = np.stack([i0 - 1, i0, i0 + 1], axis=1)
        iy_candidates = np.stack([j0 - 1, j0, j0 + 1], axis=1)
        rx = np.abs(fx[:, None] - ix_candidates)
        ry = np.abs(fy[:, None] - iy_candidates)

        def tsc_w(r):
            w = np.zeros_like(r)
            mask1 = r <= 0.5
            mask2 = (r > 0.5) & (r < 1.5)
            w[mask1] = 0.75 - r[mask1] ** 2
            w[mask2] = 0.5 * (1.5 - r[mask2]) ** 2
            return w

        wx = tsc_w(rx)
        wy = tsc_w(ry)
        representative = (i0 >= 0) & (i0 < grid.nx) & (j0 >= 0) & (j0 < grid.ny)
        node_index[original[representative], 0] = i0[representative]
        node_index[original[representative], 1] = j0[representative]

        for a in range(3):
            for b_idx in range(3):
                ixn = ix_candidates[:, a]
                iyn = iy_candidates[:, b_idx]
                weight = wx[:, a] * wy[:, b_idx]
                valid = (
                    (ixn > 0)
                    & (ixn < grid.nx - 1)
                    & (iyn > 0)
                    & (iyn < grid.ny - 1)
                    & (weight != 0.0)
                )
                if not np.any(valid):
                    continue
                flat = (iyn[valid] - 1) * nx_internal + (ixn[valid] - 1)
                np.add.at(b, flat, q[valid] * weight[valid])
        return b, node_index

    raise ValueError("Unknown deposition method. Supported methods: NGP, CIC, TSC.")
