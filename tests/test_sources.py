import numpy as np

from dcbeamtrack import (
    AnnulusGrid,
    GaussianMixture,
    RectGrid,
    TriGrid,
    UniformDisk,
    sample_slice_from_density_on_grid,
)


def test_uniform_disk_sampling_returns_slice_with_total_line_charge():
    grid = RectGrid.from_size(width=1.0e-3, height=1.0e-3, spacing=0.25e-3)
    current = 0.015
    vz0 = 1.0e7
    source = UniformDisk(radius=0.7e-3, current=current, vz=vz0)

    dist = sample_slice_from_density_on_grid(
        source,
        grid=grid,
        z0=2.0e-3,
        remove_small_charge=False,
        renormalize=True,
    )

    assert np.isclose(dist.z, 2.0e-3)
    assert len(dist) == grid.num_cells
    assert np.isclose(np.sum(dist.lam), current / vz0)
    assert np.allclose(dist.vx, 0.0)
    assert np.allclose(dist.vy, 0.0)
    assert np.allclose(dist.vz, vz0)


def test_gaussian_mixture_sampling_supports_legacy_current_key():
    grid = RectGrid.from_size(width=4.0e-3, height=4.0e-3, spacing=0.5e-3)
    vz0 = 1.2e7
    source = GaussianMixture(
        [
            {"I": 0.01, "x0": -0.25e-3, "y0": 0.0, "sigma_x": 0.3e-3, "sigma_y": 0.4e-3},
            {"I": 0.02, "x0": 0.25e-3, "y0": 0.0, "sigma_x": 0.3e-3, "sigma_y": 0.4e-3},
        ],
        vz=vz0,
    )

    dist = sample_slice_from_density_on_grid(source, grid=grid, z0=0.0, renormalize=True)

    assert len(dist) > 0
    assert np.isclose(np.sum(dist.lam), 0.03 / vz0)
    assert np.isfinite(dist.x).all()
    assert np.isfinite(dist.y).all()


def test_tri_grid_sampling_returns_slice_with_total_line_charge():
    grid = TriGrid.from_size(width=1.0e-3, height=1.0e-3, spacing=0.25e-3)
    current = 0.012
    vz0 = 1.1e7
    source = UniformDisk(radius=0.7e-3, current=current, vz=vz0)

    dist = sample_slice_from_density_on_grid(
        source,
        grid=grid,
        z0=0.0,
        remove_small_charge=False,
        renormalize=True,
    )

    assert len(dist) == grid.num_cells
    assert np.isclose(np.sum(dist.lam), current / vz0)
    assert np.isclose(np.sum(grid.cell_areas), 1.0e-6)


def test_annulus_grid_sampling_matches_uniform_disk_line_charge_without_renormalization():
    radius = 0.5e-3
    current = 0.02
    vz0 = 1.0e7
    grid = AnnulusGrid.from_equal_area(radius=radius, n_rings=4, n_angular=12)
    source = UniformDisk(radius=radius, current=current, vz=vz0)

    dist = sample_slice_from_density_on_grid(
        source,
        grid=grid,
        z0=0.0,
        remove_small_charge=False,
        renormalize=False,
    )

    assert len(dist) == grid.num_cells
    assert np.isclose(np.sum(grid.cell_areas), np.pi * radius * radius)
    assert np.isclose(np.sum(dist.lam), current / vz0)
