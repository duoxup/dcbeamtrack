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
    assert set(meta.keys()) == {
        "solver",
        "magnetic_field",
        "electric_field",
        "boundary",
        "config",
        "particle_counts",
    }
    assert meta["solver"]["type"] == "RectFdmSpaceChargeSolver"
    assert meta["magnetic_field"]["type"] == "AxialMagneticField"
    assert meta["electric_field"]["type"] == "ZeroElectricField"
    assert meta["boundary"]["type"] == "RectBoundary"
    assert meta["config"]["type"] == "TrackConfig"
    assert meta["particle_counts"]["n_initial"] == result.n_particles
    assert meta["particle_counts"]["n_live_final"] + meta["particle_counts"]["n_wall_hit_final"] == result.n_particles


def test_metadata_roundtrip_v2(tmp_path: Path):
    s0 = _snap(0.0, [1.0, 0.0], [0.0, 2.0])
    s1 = _snap(1.0, [3.0, 0.0], [0.0, 4.0])
    original = SliceTrackResult(
        (s0, s1),
        metadata={
            "solver": {"type": "AnalyticMirrorChargeSpaceChargeSolver", "radius": 1.5e-4},
            "magnetic_field": {"type": "UniformMagneticField", "bx": 0.0, "by": 0.0, "bz": 0.6},
            "boundary": {"type": "CircularBoundary", "radius": 1.5e-4},
            "config": {"type": None},
            "particle_counts": {"n_initial": 2, "n_live_final": 2, "n_wall_hit_final": 0},
        },
    )
    path = tmp_path / "roundtrip.npz"
    original.save_npz(path)
    loaded = SliceTrackResult.load_npz(path)
    assert loaded.metadata == original.metadata


def test_load_v1_npz_returns_empty_metadata(tmp_path: Path):
    # Synthesize a v1-style npz by re-encoding the envelope with version=1
    # and stripping result_metadata.
    s0 = _snap(0.0, [1.0], [0.0])
    s1 = _snap(1.0, [1.0], [0.0])
    original = SliceTrackResult((s0, s1), metadata={"will": "be_stripped"})
    path = tmp_path / "v2.npz"
    original.save_npz(path)

    # Read, rewrite envelope as v1.
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "metadata"}
        envelope = json.loads(str(data["metadata"]))
    envelope["version"] = 1
    envelope.pop("result_metadata", None)
    arrays["metadata"] = np.asarray(json.dumps(envelope))
    v1_path = tmp_path / "v1.npz"
    np.savez_compressed(v1_path, **arrays)

    loaded = SliceTrackResult.load_npz(v1_path)
    assert loaded.metadata == {}
    assert len(loaded) == 2


def test_save_rejects_non_json_metadata(tmp_path: Path):
    s0 = _snap(0.0, [1.0], [0.0])
    s1 = _snap(1.0, [1.0], [0.0])
    bad = SliceTrackResult((s0, s1), metadata={"bad": np.array([1, 2])})
    with pytest.raises(ValueError, match="JSON-serializable"):
        bad.save_npz(tmp_path / "bad.npz")
