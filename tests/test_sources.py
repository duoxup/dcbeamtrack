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
    grid = AnnulusGrid.from_uniform_cells(radius=radius, n_rings=4)
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


def test_annulus_grid_from_uniform_cells_layout():
    radius = 35e-6
    # n_rings=17 reproduces the legacy 1089-cell layout.
    grid = AnnulusGrid.from_uniform_cells(radius=radius, n_rings=17)
    assert grid.num_cells == 1089

    # Cells are equal-area: max/min should be 1 (modulo floating point).
    areas = grid.cell_areas
    assert np.allclose(areas, areas[0])

    # The macroparticles must not collapse onto a small set of angles
    # (the "spoke" failure mode produced by scalar n_angular).
    r = np.hypot(grid.xc, grid.yc)
    theta = np.arctan2(grid.yc, grid.xc)
    outer = r > 0.5 * radius
    n_unique_outer_angles = np.unique(np.round(theta[outer], 6)).size
    assert n_unique_outer_angles >= 100  # legacy outer rings have ~64..128 sectors

    # Cell aspect ratio (azimuthal / radial extent) should stay near 1 for
    # outer rings; with scalar n_angular it would blow up.
    # Use ring 8 (cells indexed 1 + 8*(1+...+7) .. 1 + 8*(1+...+8)) as a probe.
    start = 1 + 8 * (1 + 2 + 3 + 4 + 5 + 6 + 7)
    n_ring8 = 8 * 8
    rs = r[start:start + n_ring8]
    ths = theta[start:start + n_ring8]
    radial_span = rs.max() - rs.min()
    # All cells in ring 8 share the same r (centroid radius), so radial_span ≈ 0.
    # Aspect-check instead via per-cell angular pitch:
    pitch = 2 * np.pi / n_ring8
    # Ring 8 outer radius ≈ sqrt(8*8*8/1089) * radius (with leading 1 absorbed).
    mean_r = rs.mean()
    azimuthal = mean_r * pitch
    # Approximate radial width from cumulative-area formula.
    s_in_sq = (1 + 8 * (1 + 2 + 3 + 4 + 5 + 6 + 7)) / 1089
    s_out_sq = (1 + 8 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8)) / 1089
    radial_width = (np.sqrt(s_out_sq) - np.sqrt(s_in_sq)) * radius
    aspect = azimuthal / radial_width
    assert 0.5 < aspect < 2.0  # near unit aspect
