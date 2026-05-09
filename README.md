# dcbeamtrack

Minimal 2.5D tracking tools for DC electron beams.

## Current Public API

The validated first path is a fixed-z slice simulation with rectangular FDM
space charge and a CPU Boris pusher.

```python
import numpy as np
from dcbeamtrack import (
    AxialMagneticField,
    DcBeamSimulation,
    RectGrid,
    TrackConfig,
    UniformDisk,
    sample_slice_from_density_on_grid,
)

vz0 = 5.0e7
source = UniformDisk(radius=35e-6, current=-0.015, vz=vz0)
particle_grid = RectGrid.from_size(width=70e-6, height=70e-6, spacing=5e-6)
field_grid = RectGrid.from_size(width=0.3e-3, height=0.3e-3, spacing=3e-6)

initial = sample_slice_from_density_on_grid(source, particle_grid, z0=0.0)
field = AxialMagneticField.from_z_bz(
    np.linspace(-3.0, 33.0, 1001),
    0.35 * np.sin(2.0 * np.pi * np.linspace(-3.0, 33.0, 1001) / 4.93),
    z_unit="mm",
    include_radial=True,
)
config = TrackConfig(dt=1e-13, steps=100, z0=0.0, vz0=vz0, Lz=30e-3)

simulation = DcBeamSimulation.rect_fdm(
    grid=field_grid,
    magnetic_field=field,
    config=config,
)
result = simulation.run(initial)
result.save_npz("result.npz")
```

Saved results can be restored with:

```python
from dcbeamtrack import SliceTrackResult

result = SliceTrackResult.load_npz("result.npz")
z, rms_radius = result.radial_envelope("rms")
```

## Regression Tools

`tests/test_api_result_regression.py` contains a small reference calculation that
does not depend on the legacy source tree. For local comparison against the old
working code, run:

```bash
PYTHONPATH=src python scripts/compare_t2d_rec_v4.py --steps 5000
```
