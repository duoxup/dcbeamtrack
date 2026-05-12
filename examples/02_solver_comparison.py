"""Solver comparison — rect FDM (unbounded) vs circular Green inside a pipe.

Purpose
-------
Build the same initial slice once, run it through two space-charge solvers,
compare the resulting envelopes, and save a side-by-side plot plus a JSON
summary.

The two paths run physically different boundaries (rect FDM is unbounded;
circular Green has a Dirichlet pipe at ``radius_pipe``), so this is not a
"prove they match" demo — it's a "look at where the two paths diverge" demo.

How to run
----------
    PYTHONPATH=src python examples/02_solver_comparison.py

Outputs
-------
- ``artifacts/examples/02_solver_comparison/envelope.png`` — r_rms and r_max vs z.
- ``artifacts/examples/02_solver_comparison/summary.json`` — both envelope_summary
  dicts and the compare(...) dict.
- stdout: both summaries and the compare dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    vz0 = 7.96e7
    radius_pipe = 150e-6
    source = UniformDisk(radius=35e-6, current=-0.015, vz=vz0)
    particle_grid = RectGrid.from_size(width=80e-6, height=80e-6, spacing=4e-6)
    field_grid = RectGrid.from_size(width=320e-6, height=320e-6, spacing=4e-6)
    initial = sample_slice_from_density_on_grid(source, grid=particle_grid, z0=0.0)

    z_table = np.linspace(-3.0, 33.0, 1001)
    bz_table = 0.35 * np.sin(2.0 * np.pi * z_table / 4.93)
    magnetic_field = AxialMagneticField.from_z_bz(
        z_table, bz_table, z_unit="mm", include_radial=True,
    )

    config = TrackConfig(dt=1e-13, steps=500, z0=0.0, vz0=vz0, Lz=30e-3)

    sim_rect = DcBeamSimulation.rect_fdm(
        grid=field_grid, magnetic_field=magnetic_field, config=config,
    )
    sim_green = DcBeamSimulation.circular_green(
        radius=radius_pipe, magnetic_field=magnetic_field, config=config,
        n_r=96, n_theta=360,
    )

    result_rect = sim_rect.run(initial)
    result_green = sim_green.run(initial)

    env_rect = result_rect.envelope_summary()
    env_green = result_green.envelope_summary()
    cmp = result_rect.compare(result_green)

    print("rect_fdm envelope_summary:", env_rect)
    print("circular_green envelope_summary:", env_green)
    print("compare:", cmp)

    out_dir = Path("artifacts/examples/02_solver_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"rect_fdm": env_rect, "circular_green": env_green, "compare": cmp}
    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, kind in zip(axes, ("rms", "max")):
        z_a, env_a = result_rect.radial_envelope(kind)
        z_b, env_b = result_green.radial_envelope(kind)
        ax.plot(z_a * 1e3, env_a * 1e6, label="rect_fdm")
        ax.plot(z_b * 1e3, env_b * 1e6, label="circular_green")
        ax.set_xlabel("z [mm]")
        ax.set_ylabel(f"r_{kind} [µm]")
        ax.set_title(f"r_{kind} envelope")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "envelope.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
