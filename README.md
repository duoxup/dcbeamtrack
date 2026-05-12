# dcbeamtrack

Minimal 2.5D tracking tools for DC electron beams.

## Installation

Not on PyPI yet. Install directly from GitHub:

```bash
pip install git+https://github.com/duoxup/dcbeamtrack.git
pip install "dcbeamtrack[fast] @ git+https://github.com/duoxup/dcbeamtrack.git"
pip install "dcbeamtrack[examples] @ git+https://github.com/duoxup/dcbeamtrack.git"
```

Extras: `[fast]` adds `pypardiso` (multi-threaded MKL PARDISO);
`[examples]` adds `matplotlib` for the runnable examples under `examples/`.

Or clone for local development:

```bash
git clone https://github.com/duoxup/dcbeamtrack.git
cd dcbeamtrack
pip install -e .
```

The `[fast]` extra pulls in `pypardiso` (Intel MKL bundle, ~600 MB one-time
download). When installed, both FDM solvers automatically use the
multi-threaded backend via `backend="auto"` (default). Measured speedup is
1.4-2.5× on a 100²-200² grid at 8 threads. To force a specific backend,
pass `backend="scipy"` or `backend="pypardiso"` to the solver constructor.

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

initial = sample_slice_from_density_on_grid(source, grid=particle_grid, z0=0.0)
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

## Examples

Four runnable examples live in `examples/`:

- `examples/01_rect_fdm_minimal.py` — minimal `rect_fdm` simulation; runnable companion to the code block above.
- `examples/02_solver_comparison.py` — same slice through `rect_fdm` and `circular_green` with envelope comparison and a saved plot.
- `examples/03_result_analysis.py` — exercises the result-side diagnostic API (`envelope_summary`, `final_summary`, `metadata`, `trajectories`, `random_mask`, `evenly_spaced_mask`, `where_live`).
- `examples/04_custom_solver.py` — write your own `SpaceChargeSolver` + `FieldMap2DProtocol` from scratch and plug it in via `DcBeamSimulation.from_solver`.

Run any example from the repo root with `python examples/<name>.py`
(`PYTHONPATH=src python examples/<name>.py` if the package is not pip-installed).
Outputs land under `artifacts/examples/<stem>/`.

## Regression Tools

`tests/test_api_result_regression.py` contains a small reference calculation that
does not depend on the legacy source tree.

## License

MIT — see [LICENSE](LICENSE).
