import math
import json
from pathlib import Path

import numpy as np
import pytest
from partdist import SliceDistribution

from dcbeamtrack import SliceTrackResult


def _snap(z: float, x, y, *, lam=None) -> SliceDistribution:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    zero = np.zeros(n)
    if lam is None:
        lam = np.full(n, 1.0e-12)
    return SliceDistribution(
        z=z, x=x, y=y, px=zero, py=zero, pz=zero,
        t=zero, lam=np.asarray(lam, dtype=float),
    )


def test_slice_track_result_metadata_default_empty():
    s0 = _snap(0.0, [1.0], [0.0])
    s1 = _snap(1.0, [1.0], [0.0])
    result = SliceTrackResult((s0, s1))
    assert result.metadata == {}


def test_slice_track_result_accepts_metadata_kwarg():
    s0 = _snap(0.0, [1.0], [0.0])
    s1 = _snap(1.0, [1.0], [0.0])
    result = SliceTrackResult((s0, s1), metadata={"foo": "bar"})
    assert result.metadata == {"foo": "bar"}


def test_safe_summary_fallback_on_unknown_class():
    from dcbeamtrack.tracking.result import _safe_summary

    class Dummy:
        pass

    assert _safe_summary(Dummy()) == {"type": "Dummy"}


def test_safe_summary_calls_method_when_present():
    from dcbeamtrack.tracking.result import _safe_summary

    class Known:
        def summary(self):
            return {"type": "Known", "value": 42}

    assert _safe_summary(Known()) == {"type": "Known", "value": 42}


def test_config_summary_for_track_config():
    from dcbeamtrack.config import TrackConfig
    from dcbeamtrack.tracking.result import _config_summary

    config = TrackConfig(dt=1e-13, steps=100, z0=0.0, vz0=5e7, Lz=3e-2)
    summary = _config_summary(config)
    assert summary["type"] == "TrackConfig"
    assert math.isclose(summary["dt"], 1e-13)
    assert summary["steps"] == 100


def test_config_summary_for_none():
    from dcbeamtrack.tracking.result import _config_summary
    assert _config_summary(None) == {"type": None}


def test_simulation_run_populates_metadata():
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
    particle_grid = RectGrid.from_size(width=70e-6, height=70e-6, spacing=10e-6)
    field_grid = RectGrid.from_size(width=3.0e-4, height=3.0e-4, spacing=3.0e-5)
    initial = sample_slice_from_density_on_grid(source, grid=particle_grid, z0=0.0)

    z_mm = np.linspace(-3.0, 33.0, 1001)
    field = AxialMagneticField.from_z_bz(
        z_mm, 0.35 * np.sin(2.0 * np.pi * z_mm / 4.93),
        z_unit="mm", include_radial=True,
    )
    config = TrackConfig(dt=1e-13, steps=5, z0=0.0, vz0=vz0, Lz=30e-3)
    simulation = DcBeamSimulation.rect_fdm(
        grid=field_grid, magnetic_field=field, config=config,
    )
    result = simulation.run(initial)

    meta = result.metadata
    assert set(meta.keys()) == {"solver", "magnetic_field", "boundary", "config", "particle_counts"}
    assert meta["solver"]["type"] == "RectFdmSpaceChargeSolver"
    assert meta["magnetic_field"]["type"] == "AxialMagneticField"
    assert meta["boundary"]["type"] == "RectBoundary"
    assert meta["config"]["type"] == "TrackConfig"
    assert meta["particle_counts"]["n_initial"] == result.n_particles
    assert meta["particle_counts"]["n_live_final"] + meta["particle_counts"]["n_wall_hit_final"] == result.n_particles
