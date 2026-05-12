"""Result diagnostics — exercise Cut 1/2/3 result-side APIs.

Purpose
-------
Demonstrate the diagnostic surface on ``SliceTrackResult``:

- ``envelope_summary`` / ``final_summary`` (Cut 1)
- ``metadata`` round-trip via npz (Cut 2)
- ``random_mask`` / ``evenly_spaced_mask`` / ``trajectories`` / ``where_live`` (Cut 3)

How to run
----------
    PYTHONPATH=src python examples/03_result_analysis.py

If ``artifacts/examples/01_rect_fdm_minimal/result.npz`` exists, this example
loads it (so the printed metadata reflects the original 100-step run). Otherwise
it runs a quick inline simulation (steps=200) so the example is self-sufficient.

Outputs
-------
- ``artifacts/examples/03_result_analysis/trajectories.png`` — two subplots:
  random-sample x(z) trajectories (left), evenly-spaced x(z) trajectories
  (right). Dead positions are nan-filled, so wall-hit particles draw gaps
  instead of vertical lines.
- stdout: envelope_summary, final_summary, metadata keys, and one nested
  metadata entry.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dcbeamtrack import (
    AxialMagneticField,
    DcBeamSimulation,
    RectGrid,
    SliceTrackResult,
    TrackConfig,
    TriGrid,
    UniformDisk,
    sample_slice_from_density_on_grid,
)


CACHED_RESULT = Path("artifacts/examples/01_rect_fdm_minimal/result.npz")


def _quick_sim() -> SliceTrackResult:
    vz0 = 5.0e7
    source = UniformDisk(radius=35e-6, current=-0.015, vz=vz0)
    particle_grid = TriGrid.from_size(width=70e-6, height=70e-6, hmax=3e-6)
    field_grid = RectGrid.from_size(width=0.3e-3, height=0.3e-3, spacing=3e-6)
    initial = sample_slice_from_density_on_grid(source, grid=particle_grid, z0=0.0)

    z_table = np.linspace(-3.0, 33.0, 1001)
    bz_table = 0.35 * np.sin(2.0 * np.pi * z_table / 4.93)
    magnetic_field = AxialMagneticField.from_z_bz(
        z_table, bz_table, z_unit="mm", include_radial=True,
    )
    config = TrackConfig(dt=1e-13, steps=200, z0=0.0, vz0=vz0, Lz=30e-3)
    sim = DcBeamSimulation.rect_fdm(
        grid=field_grid, magnetic_field=magnetic_field, config=config,
    )
    return sim.run(initial)


def main() -> None:
    if CACHED_RESULT.exists():
        result = SliceTrackResult.load_npz(CACHED_RESULT)
        print(f"loaded result from {CACHED_RESULT}")
    else:
        result = _quick_sim()
        print("ran a quick inline simulation (steps=200)")

    print("envelope_summary:", result.envelope_summary())
    print("final_summary:", result.final_summary())
    print("metadata keys:", sorted(result.metadata.keys()))
    print("metadata['solver']:", result.metadata.get("solver"))

    random_mask = result.random_mask(n=30, seed=0)
    even_mask = result.evenly_spaced_mask(n=30)
    traj_random = result.trajectories(random_mask).where_live()
    traj_even = result.trajectories(even_mask).where_live()
    z_mm = traj_random.z * 1e3

    out_dir = Path("artifacts/examples/03_result_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, traj, title in zip(
        axes,
        (traj_random, traj_even),
        ("random 30", "evenly-spaced 30"),
    ):
        ax.plot(z_mm, traj.data["x"] * 1e6, color="C0", alpha=0.6, linewidth=0.8)
        ax.set_xlabel("z [mm]")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("x [µm]")
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
