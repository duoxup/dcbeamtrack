"""Minimal rect FDM simulation — runnable companion to the README snippet.

Purpose
-------
Configure a 2.5D DC beam slice simulation with rect FDM space charge and an
axial Bz lookup table, run it for 100 steps, save the result, and print the
final-snapshot summary.

How to run
----------
From the repo root:

    PYTHONPATH=src python examples/01_rect_fdm_minimal.py

(Drop ``PYTHONPATH=src`` if the package is pip-installed.)

Outputs
-------
- ``artifacts/examples/01_rect_fdm_minimal/result.npz`` — the saved SliceTrackResult.
- stdout: a one-line print of ``result.final_summary()``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dcbeamtrack import (
    AxialMagneticField,
    DcBeamSimulation,
    RectGrid,
    TrackConfig,
    UniformDisk,
    sample_slice_from_density_on_grid,
)


def main() -> None:
    vz0 = 5.0e7
    source = UniformDisk(radius=35e-6, current=-0.015, vz=vz0)
    particle_grid = RectGrid.from_size(width=70e-6, height=70e-6, spacing=5e-6)
    field_grid = RectGrid.from_size(width=0.3e-3, height=0.3e-3, spacing=3e-6)

    initial = sample_slice_from_density_on_grid(source, grid=particle_grid, z0=0.0)

    z_table = np.linspace(-3.0, 33.0, 1001)
    bz_table = 0.35 * np.sin(2.0 * np.pi * z_table / 4.93)
    magnetic_field = AxialMagneticField.from_z_bz(
        z_table, bz_table, z_unit="mm", include_radial=True,
    )

    config = TrackConfig(dt=1e-13, steps=100, z0=0.0, vz0=vz0, Lz=30e-3)
    simulation = DcBeamSimulation.rect_fdm(
        grid=field_grid, magnetic_field=magnetic_field, config=config,
    )
    result = simulation.run(initial)

    out_dir = Path("artifacts/examples/01_rect_fdm_minimal")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_npz(out_dir / "result.npz")
    print("final_summary:", result.final_summary())


if __name__ == "__main__":
    main()
