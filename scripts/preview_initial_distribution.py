from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from dcbeamtrack import (
    AnnulusGrid,
    GaussianMixture,
    RectGrid,
    TriGrid,
    UniformDisk,
    sample_slice_from_density_on_grid,
)


def _plot_distribution(ax, dist, title: str) -> None:
    scatter = ax.scatter(
        dist.x * 1e3,
        dist.y * 1e3,
        c=dist.lam,
        s=28.0,
        cmap="jet",
        edgecolors="none",
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.88", linewidth=0.8)
    return scatter


def _plot_grid(ax, grid) -> None:
    if hasattr(grid, "polygons") and grid.polygons:
        segments = []
        for poly in grid.polygons:
            poly_mm = np.asarray(poly, dtype=float) * 1e3
            closed = np.vstack([poly_mm, poly_mm[0]])
            segments.extend(np.stack([closed[:-1], closed[1:]], axis=1))
        ax.add_collection(
            LineCollection(segments, colors="0.15", linewidths=0.45, alpha=0.55, zorder=1)
        )
        return

    if hasattr(grid, "x_nodes") and hasattr(grid, "y_nodes"):
        xmin = grid.x_nodes[0] * 1e3
        xmax = grid.x_nodes[-1] * 1e3
        ymin = grid.y_nodes[0] * 1e3
        ymax = grid.y_nodes[-1] * 1e3
        for x in grid.x_nodes * 1e3:
            ax.plot([x, x], [ymin, ymax], color="0.15", linewidth=0.45, alpha=0.55, zorder=1)
        for y in grid.y_nodes * 1e3:
            ax.plot([xmin, xmax], [y, y], color="0.15", linewidth=0.45, alpha=0.55, zorder=1)


def main() -> None:
    out = Path("artifacts/initial_distribution_grid_preview.png")
    rect_grid = RectGrid.from_size(width=1.2e-3, height=1.2e-3, spacing=0.08e-3)
    tri_grid = TriGrid.from_size(width=1.2e-3, height=1.2e-3, spacing=0.12e-3)
    ann_grid = AnnulusGrid.from_equal_area(radius=0.6e-3, n_rings=8, n_angular=24)
    grids = [
        ("RectGrid", rect_grid, {"quad_order": 5}),
        ("TriGrid", tri_grid, {}),
        ("AnnulusGrid", ann_grid, {}),
    ]
    vz0 = 1.0e7

    uniform = UniformDisk(radius=0.42e-3, current=0.015, vz=vz0)
    mixture = GaussianMixture(
        [
            {
                "current": 0.009,
                "x0": -0.22e-3,
                "y0": 0.10e-3,
                "sigma_x": 0.13e-3,
                "sigma_y": 0.18e-3,
            },
            {
                "current": 0.006,
                "x0": 0.20e-3,
                "y0": -0.08e-3,
                "sigma_x": 0.16e-3,
                "sigma_y": 0.12e-3,
            },
        ],
        vz=vz0,
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), constrained_layout=True)
    for col, (name, grid, kwargs) in enumerate(grids):
        uniform_dist = sample_slice_from_density_on_grid(
            uniform,
            grid=grid,
            z0=0.0,
            remove_small_charge=True,
            min_lam_fraction=2e-3,
            **kwargs,
        )
        mixture_dist = sample_slice_from_density_on_grid(
            mixture,
            grid=grid,
            z0=0.0,
            remove_small_charge=True,
            min_lam_fraction=2e-3,
            **kwargs,
        )
        sc0 = _plot_distribution(
            axes[0, col],
            uniform_dist,
            f"{name} / UniformDisk\n{len(uniform_dist)} macroparticles",
        )
        _plot_grid(axes[0, col], grid)
        sc1 = _plot_distribution(
            axes[1, col],
            mixture_dist,
            f"{name} / GaussianMixture\n{len(mixture_dist)} macroparticles",
        )
        _plot_grid(axes[1, col], grid)
        cbar0 = fig.colorbar(sc0, ax=axes[0, col], fraction=0.046, pad=0.04)
        cbar1 = fig.colorbar(sc1, ax=axes[1, col], fraction=0.046, pad=0.04)
        cbar0.set_label("lambda [C/m]")
        cbar1.set_label("lambda [C/m]")

    fig.suptitle("dcbeamtrack initial SliceDistribution preview across sampling grids")
    fig.savefig(out, dpi=180)
    print(out.resolve())


if __name__ == "__main__":
    main()
