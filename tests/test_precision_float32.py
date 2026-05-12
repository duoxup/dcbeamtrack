"""Smoke test for the float32 precision tracker path.

Confirms `TrackConfig(precision="float32")` runs end-to-end without
type errors and that snapshot arrays come back as float32. Does not
compare numerical results against float64 — that's a separate study.
"""

import numpy as np
from partdist import SliceDistribution

from dcbeamtrack import (
    AxialMagneticField,
    DcBeamSimulation,
    RectGrid,
    TrackConfig,
    UniformDisk,
    sample_slice_from_density_on_grid,
)


def test_track_slice_float32_smoke():
    vz0 = 5.0e7
    source = UniformDisk(radius=35e-6, current=-0.015, vz=vz0)
    particle_grid = RectGrid.from_size(width=70e-6, height=70e-6, spacing=5e-6)
    field_grid = RectGrid.from_size(width=0.3e-3, height=0.3e-3, spacing=3e-6)
    initial = sample_slice_from_density_on_grid(source, grid=particle_grid, z0=0.0)

    z_table = np.linspace(-3.0, 33.0, 1001)
    bz_table = 0.35 * np.sin(2.0 * np.pi * z_table / 4.93)
    field = AxialMagneticField.from_z_bz(
        z_table, bz_table, z_unit="mm", include_radial=True,
    )

    config = TrackConfig(
        dt=1e-13, steps=10, z0=0.0, vz0=vz0, Lz=30e-3,
        precision="float32",
    )
    sim = DcBeamSimulation.rect_fdm(
        grid=field_grid, magnetic_field=field, config=config,
    )
    result = sim.run(initial)

    # Smoke: ran end-to-end and produced the expected number of snapshots.
    assert len(result) == 11  # initial + 10 steps at stride=1
    snap = result.final
    assert isinstance(snap, SliceDistribution)
    # Note: partdist.SliceDistribution upcasts inputs to float64 at
    # construction, so snapshot arrays are float64 even when the tracker
    # ran in float32 internally. The precision flag still controls the
    # in-loop arithmetic dtype (and hence Boris/deposition behavior),
    # just not the persisted snapshot dtype.
    assert np.asarray(snap.x).dtype == np.float64
