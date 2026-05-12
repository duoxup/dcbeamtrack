"""Custom solver walkthrough — linear restoring "field" for protocol demo.

Purpose
-------
Demonstrate how to implement ``SpaceChargeSolver`` and ``FieldMap2DProtocol``
from scratch and plug the result into ``DcBeamSimulation.from_solver``.

The "solver" returned here is a fake linear restoring field
``Ex = -k * x, Ey = -k * y`` — NOT physical space charge, just the
smallest concrete pair of classes that satisfies both protocols. For real
custom solvers, replace the bodies of ``solve()`` / ``field_at()`` with
actual physics (Coulomb sum, multipole expansion, PIC, ...) while keeping
the same signatures.

How to run
----------
    PYTHONPATH=src python examples/04_custom_solver.py

Outputs
-------
- ``artifacts/examples/04_custom_solver/result.npz`` — the saved SliceTrackResult.
- ``artifacts/examples/04_custom_solver/envelope.png`` — r_rms vs z under the
  fake restoring field; visibly different (oscillating/focusing) from the
  monotonic space-charge defocusing of ``examples/01_rect_fdm_minimal.py``.
- stdout: one ``envelope_summary`` dict.
"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass
class FakeFieldMap2D:
    """Tiny FieldMap2DProtocol implementation: Ex = -k*x, Ey = -k*y."""

    k: float

    def field_at(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return -self.k * x, -self.k * y


@dataclass
class LinearRestoringSolver:
    """Tiny SpaceChargeSolver: ignore the distribution and return a FakeFieldMap2D."""

    k: float

    def solve(self, distribution):
        return FakeFieldMap2D(self.k)


def main() -> None:
    vz0 = 5.0e7
    source = UniformDisk(radius=35e-6, current=-0.015, vz=vz0)
    particle_grid = RectGrid.from_size(width=70e-6, height=70e-6, spacing=5e-6)
    initial = sample_slice_from_density_on_grid(source, grid=particle_grid, z0=0.0)

    z_table = np.linspace(-3.0, 33.0, 1001)
    bz_table = 0.35 * np.sin(2.0 * np.pi * z_table / 4.93)
    magnetic_field = AxialMagneticField.from_z_bz(
        z_table, bz_table, z_unit="mm", include_radial=True,
    )

    config = TrackConfig(dt=1e-13, steps=100, z0=0.0, vz0=vz0, Lz=30e-3)
    simulation = DcBeamSimulation.from_solver(
        spacecharge_solver=LinearRestoringSolver(k=1e8),
        magnetic_field=magnetic_field,
        config=config,
    )
    result = simulation.run(initial)

    out_dir = Path("artifacts/examples/04_custom_solver")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.save_npz(out_dir / "result.npz")
    print("envelope_summary:", result.envelope_summary())

    z, r_rms = result.radial_envelope("rms")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(z * 1e3, r_rms * 1e6)
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("r_rms [µm]")
    ax.set_title("Linear restoring force — r_rms envelope")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "envelope.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
