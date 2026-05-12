import math

import numpy as np
import pytest
from partdist import SliceDistribution

from dcbeamtrack import SliceTrackResult


def _snap(z: float, x, y, *, lam=None, wall_hit=None) -> SliceDistribution:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    zero = np.zeros(n)
    if lam is None:
        lam = np.full(n, 1.0e-12)
    extras = None
    if wall_hit is not None:
        extras = {"wall_hit": np.asarray(wall_hit, dtype=float)}
    return SliceDistribution(
        z=z, x=x, y=y, px=zero, py=zero, pz=zero,
        t=zero, lam=np.asarray(lam, dtype=float), extras=extras,
    )


def test_envelope_summary_keys_and_values():
    # 2 particles, equal weights, 2 snapshots.
    # snap 0: (1, 0), (0, 2) -> r = (1, 2)
    # snap 1: (3, 0), (0, 4) -> r = (3, 4)
    s0 = _snap(0.0, [1.0, 0.0], [0.0, 2.0])
    s1 = _snap(1.0, [3.0, 0.0], [0.0, 4.0])
    result = SliceTrackResult((s0, s1))
    summary = result.envelope_summary()

    # r_max per snap: [2, 4]; peak = 4 at z=1.0; final = 4
    assert math.isclose(summary["r_max_peak"], 4.0)
    assert math.isclose(summary["r_max_peak_z"], 1.0)
    assert math.isclose(summary["r_max_final"], 4.0)

    # weighted r_mean per snap (equal weights): (1+2)/2 = 1.5, (3+4)/2 = 3.5
    assert math.isclose(summary["r_mean_peak"], 3.5)
    assert math.isclose(summary["r_mean_peak_z"], 1.0)
    assert math.isclose(summary["r_mean_final"], 3.5)

    # weighted r_rms per snap (equal weights): sqrt((1+4)/2)=sqrt(2.5), sqrt((9+16)/2)=sqrt(12.5)
    assert math.isclose(summary["r_rms_peak"], math.sqrt(12.5))
    assert math.isclose(summary["r_rms_peak_z"], 1.0)
    assert math.isclose(summary["r_rms_final"], math.sqrt(12.5))

    # waist = argmin(r_rms) -> snap 0 (sqrt(2.5) < sqrt(12.5))
    assert math.isclose(summary["waist_z"], 0.0)
    assert math.isclose(summary["waist_r_rms"], math.sqrt(2.5))


def test_envelope_summary_excludes_wall_hit():
    # snap 0: both alive; snap 1: particle 0 marked wall_hit.
    s0 = _snap(0.0, [1.0, 0.5], [0.0, 0.0], wall_hit=[0.0, 0.0])
    s1 = _snap(1.0, [100.0, 0.5], [0.0, 0.0], wall_hit=[1.0, 0.0])
    result = SliceTrackResult((s0, s1))
    summary = result.envelope_summary()

    # snap 1 r_max should be 0.5 (only particle 1 contributes), not 100.0
    assert math.isclose(summary["r_max_final"], 0.5)


def test_envelope_summary_all_dead_raises():
    s0 = _snap(0.0, [1.0, 2.0], [0.0, 0.0], wall_hit=[1.0, 1.0])
    s1 = _snap(1.0, [3.0, 4.0], [0.0, 0.0], wall_hit=[1.0, 1.0])
    result = SliceTrackResult((s0, s1))
    with pytest.raises(ValueError, match="no live particles"):
        result.envelope_summary()


def test_final_summary_keys_and_values():
    # With wall_hit extra: 3 particles, particle 0 dead at final, particles 1+2 alive.
    s0 = _snap(0.0, [1.0, 0.5, 0.0], [0.0, 0.0, 0.5], wall_hit=[0.0, 0.0, 0.0])
    s1 = _snap(1.0, [9.0, 0.5, 0.0], [0.0, 0.0, 0.5], wall_hit=[1.0, 0.0, 0.0])
    result = SliceTrackResult((s0, s1))
    summary = result.final_summary()

    assert math.isclose(summary["z_final"], 1.0)
    assert summary["n_particles"] == 3
    assert summary["n_live_final"] == 2
    assert summary["n_wall_hit_final"] == 1
    # live subset r: particle 1 -> 0.5, particle 2 -> 0.5
    assert math.isclose(summary["r_max_final"], 0.5)
    assert math.isclose(summary["r_mean_final"], 0.5)
    assert math.isclose(summary["r_rms_final"], 0.5)

    # Without wall_hit extra: all alive.
    s0n = _snap(0.0, [1.0, 0.0], [0.0, 2.0])
    s1n = _snap(1.0, [3.0, 0.0], [0.0, 4.0])
    no_wh = SliceTrackResult((s0n, s1n))
    summary_n = no_wh.final_summary()
    assert summary_n["n_wall_hit_final"] == 0
    assert summary_n["n_live_final"] == 2
    assert math.isclose(summary_n["r_max_final"], 4.0)


def test_final_summary_all_dead_returns_nan():
    s0 = _snap(0.0, [1.0, 2.0], [0.0, 0.0], wall_hit=[0.0, 0.0])
    s1 = _snap(1.0, [3.0, 4.0], [0.0, 0.0], wall_hit=[1.0, 1.0])
    result = SliceTrackResult((s0, s1))
    summary = result.final_summary()
    assert summary["n_live_final"] == 0
    assert summary["n_wall_hit_final"] == 2
    assert math.isnan(summary["r_max_final"])
    assert math.isnan(summary["r_mean_final"])
    assert math.isnan(summary["r_rms_final"])
