"""Grid layout visualization.

``plot_grid(grid, ax=None, ...)`` draws cell boundaries for any supported grid
type. :class:`~dcbeamtrack.RectGrid` is rendered as lattice lines; cell-based
grids (:class:`~dcbeamtrack.TriGrid`, :class:`~dcbeamtrack.AnnulusGrid`) use
``PolyCollection`` over each cell's polygon. ``AnnulusGrid`` cells are
tessellated along arcs so the curved boundaries render smoothly (the stored
polygons use 4 corner vertices; the central full-disk cell is degenerate as a
quad and is rebuilt from ``radial_scales`` / ``n_angular`` / ``theta_offset``).

Matplotlib is only required when this module is used. It is imported lazily
inside :func:`plot_grid` so ``import dcbeamtrack`` does not need matplotlib.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..mesh import AnnulusGrid, CellGrid, RectGrid


_DEFAULT_ARC_DEG = 5.0


def plot_grid(
    grid,
    ax: Optional[Any] = None,
    *,
    edgecolor: str = "black",
    linewidth: float = 0.5,
    facecolor: str = "none",
    alpha: float = 1.0,
    show_centers: bool = False,
    center_color: str = "C0",
    center_size: float = 4.0,
    center_marker: str = ".",
    arc_step_deg: float = _DEFAULT_ARC_DEG,
):
    """Plot grid cell boundaries on a 2D axes.

    Parameters
    ----------
    grid
        :class:`RectGrid`, :class:`TriGrid`, or :class:`AnnulusGrid` instance.
    ax
        Existing matplotlib axes. If ``None``, a new ``(6, 6)`` figure is
        created.
    edgecolor, linewidth
        Cell boundary style.
    facecolor, alpha
        Cell fill (default ``"none"`` = transparent). Ignored for
        :class:`RectGrid` (rendered as lines, no fill).
    show_centers
        Overlay cell-center markers — useful to preview where
        ``sample_slice_from_density_on_grid`` will place macroparticles.
    center_color, center_size, center_marker
        Marker style for cell centers.
    arc_step_deg
        Target arc segment length (in degrees) used to tessellate
        :class:`AnnulusGrid` cells. Smaller = smoother curves, more vertices.

    Returns
    -------
    matplotlib.axes.Axes
        The axes the grid was drawn on.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if isinstance(grid, RectGrid):
        xmin = float(grid.x_nodes[0])
        xmax = float(grid.x_nodes[-1])
        ymin = float(grid.y_nodes[0])
        ymax = float(grid.y_nodes[-1])
        segments = [
            [(float(x), ymin), (float(x), ymax)] for x in grid.x_nodes
        ] + [
            [(xmin, float(y)), (xmax, float(y))] for y in grid.y_nodes
        ]
        ax.add_collection(
            LineCollection(segments, colors=edgecolor, linewidths=linewidth)
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    elif isinstance(grid, AnnulusGrid):
        polys = _annulus_polygons(grid, arc_step_deg=arc_step_deg)
        ax.add_collection(
            PolyCollection(
                polys,
                edgecolors=edgecolor,
                linewidths=linewidth,
                facecolors=facecolor,
                alpha=alpha,
            )
        )
        _autoscale_to_polygons(ax, polys)
    elif isinstance(grid, CellGrid):
        polys = list(grid.polygons)
        if not polys:
            raise ValueError("CellGrid has no polygons to plot.")
        ax.add_collection(
            PolyCollection(
                polys,
                edgecolors=edgecolor,
                linewidths=linewidth,
                facecolors=facecolor,
                alpha=alpha,
            )
        )
        _autoscale_to_polygons(ax, polys)
    else:
        raise TypeError(
            f"plot_grid does not support type {type(grid).__name__!r}; "
            "expected RectGrid, TriGrid, or AnnulusGrid."
        )

    if show_centers:
        xc, yc = grid.cell_centers()
        ax.scatter(xc, yc, c=center_color, s=center_size, marker=center_marker)

    ax.set_aspect("equal")
    return ax


def _autoscale_to_polygons(ax, polys) -> None:
    pts = np.vstack(polys)
    ax.set_xlim(float(pts[:, 0].min()), float(pts[:, 0].max()))
    ax.set_ylim(float(pts[:, 1].min()), float(pts[:, 1].max()))


def _annulus_polygons(grid: AnnulusGrid, *, arc_step_deg: float) -> list:
    """Rebuild each annular cell with arc-tessellated polygons.

    Uses ``radial_scales`` / ``n_angular`` / ``theta_offset`` rather than the
    stored 4-vertex polygons so curved edges render smoothly and the central
    full-disk cell (degenerate as a quad) becomes a proper disk.
    """
    (xmin, xmax), (ymin, ymax) = grid.domain_bounds
    a = 0.5 * (xmax - xmin)
    b = 0.5 * (ymax - ymin)
    xc0 = 0.5 * (xmin + xmax)
    yc0 = 0.5 * (ymin + ymax)
    arc_step = np.deg2rad(max(0.1, float(arc_step_deg)))
    radial_scales = np.asarray(grid.radial_scales, dtype=float)
    n_angular = np.asarray(grid.n_angular, dtype=int)
    theta_offset = float(grid.theta_offset)
    polys = []
    for ring in range(n_angular.size):
        r_in = float(radial_scales[ring])
        r_out = float(radial_scales[ring + 1])
        n_ang = int(n_angular[ring])
        theta_edges = (
            np.linspace(0.0, 2.0 * np.pi, n_ang + 1) + theta_offset
        )
        for k in range(n_ang):
            th0 = float(theta_edges[k])
            th1 = float(theta_edges[k + 1])
            n_seg = max(2, int(np.ceil((th1 - th0) / arc_step)))
            thetas = np.linspace(th0, th1, n_seg + 1)
            outer = np.column_stack(
                [xc0 + a * r_out * np.cos(thetas), yc0 + b * r_out * np.sin(thetas)]
            )
            if r_in <= 0.0:
                inner = np.array([[xc0, yc0]])
            else:
                rev = thetas[::-1]
                inner = np.column_stack(
                    [xc0 + a * r_in * np.cos(rev), yc0 + b * r_in * np.sin(rev)]
                )
            polys.append(np.vstack([outer, inner]))
    return polys
