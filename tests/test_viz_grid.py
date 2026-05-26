import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection, PolyCollection

from dcbeamtrack import AnnulusGrid, RectGrid, TriGrid
from dcbeamtrack.viz import plot_grid


def _collections_of(ax, cls):
    return [c for c in ax.collections if isinstance(c, cls)]


def test_plot_grid_rect_draws_lattice_lines():
    grid = RectGrid.from_size(width=4.0, height=4.0, spacing=1.0)
    fig, ax = plt.subplots()
    out = plot_grid(grid, ax=ax)
    assert out is ax
    lcs = _collections_of(ax, LineCollection)
    assert len(lcs) == 1
    segs = lcs[0].get_segments()
    assert len(segs) == grid.nx + grid.ny
    assert ax.get_xlim() == (grid.x_nodes[0], grid.x_nodes[-1])
    assert ax.get_aspect() == 1.0
    plt.close(fig)


def test_plot_grid_tri_uses_polycollection_with_one_poly_per_cell():
    grid = TriGrid.from_size(width=20e-6, height=20e-6, hmax=5e-6)
    fig, ax = plt.subplots()
    plot_grid(grid, ax=ax)
    pcs = _collections_of(ax, PolyCollection)
    assert len(pcs) == 1
    assert len(pcs[0].get_paths()) == grid.num_cells
    plt.close(fig)


def test_plot_grid_annulus_rebuilds_central_disk_with_nonzero_area():
    grid = AnnulusGrid.from_uniform_cells(radius=1.0, n_rings=4)
    fig, ax = plt.subplots()
    plot_grid(grid, ax=ax)
    pcs = _collections_of(ax, PolyCollection)
    assert len(pcs) == 1
    paths = pcs[0].get_paths()
    assert len(paths) == grid.num_cells
    central_vertices = paths[0].vertices
    xs = central_vertices[:, 0]
    ys = central_vertices[:, 1]
    bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    assert bbox_area > 0.0
    assert len(central_vertices) > 4
    plt.close(fig)


def test_plot_grid_show_centers_overlays_one_point_per_cell():
    grid = TriGrid.from_size(width=20e-6, height=20e-6, hmax=5e-6)
    fig, ax = plt.subplots()
    plot_grid(grid, ax=ax, show_centers=True)
    scatters = [c for c in ax.collections if c.__class__.__name__ == "PathCollection"]
    assert len(scatters) == 1
    assert scatters[0].get_offsets().shape == (grid.num_cells, 2)
    plt.close(fig)


def test_plot_grid_unsupported_type_raises():
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="plot_grid does not support"):
        plot_grid(object(), ax=ax)
    plt.close(fig)


def test_plot_grid_creates_axes_when_none_passed():
    grid = RectGrid.from_size(width=2.0, height=2.0, spacing=1.0)
    ax = plot_grid(grid)
    assert ax is not None
    plt.close(ax.figure)


def test_annulus_theta_offset_survives_into_viz_geometry():
    grid_a = AnnulusGrid.from_uniform_cells(radius=1.0, n_rings=3, theta_offset=0.0)
    grid_b = AnnulusGrid.from_uniform_cells(
        radius=1.0, n_rings=3, theta_offset=np.pi / 4
    )
    assert grid_a.theta_offset == 0.0
    assert np.isclose(grid_b.theta_offset, np.pi / 4)
    fig, (ax_a, ax_b) = plt.subplots(1, 2)
    plot_grid(grid_a, ax=ax_a)
    plot_grid(grid_b, ax=ax_b)
    pc_a = _collections_of(ax_a, PolyCollection)[0]
    pc_b = _collections_of(ax_b, PolyCollection)[0]
    assert not np.allclose(
        pc_a.get_paths()[-1].vertices, pc_b.get_paths()[-1].vertices
    )
    plt.close(fig)
