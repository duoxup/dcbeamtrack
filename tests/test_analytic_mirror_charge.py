import numpy as np
import pytest
from partdist import SliceDistribution

from dcbeamtrack.spacecharge.analytic_mirror_charge import (
    AnalyticMirrorChargeSpaceChargeSolver,
)
from dcbeamtrack.tracking import velocity_to_momentum_evc
from _helpers import direct_mirror_charge_field


def _slice(x, y, lam=None) -> SliceDistribution:
    n = x.size
    vx = vy = vz = np.zeros(n)
    px, py, pz = velocity_to_momentum_evc(vx, vy, vz)
    if lam is None:
        lam = np.full(n, 1.0e-12)
    return SliceDistribution(
        z=0.0,
        x=x,
        y=y,
        px=px,
        py=py,
        pz=pz,
        t=np.zeros(n),
        lam=lam,
    )


def test_analytic_matches_pure_python_reference():
    radius = 1.0e-3
    rng = np.random.default_rng(0)
    n = 8
    r = rng.uniform(0.05 * radius, 0.85 * radius, size=n)
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    lam = rng.uniform(0.5e-12, 1.5e-12, size=n)

    solver = AnalyticMirrorChargeSpaceChargeSolver(radius=radius)
    fm = solver.solve(_slice(x, y, lam=lam))
    ex_ref, ey_ref = direct_mirror_charge_field(x, y, lam, radius)

    scale = max(np.max(np.abs(ex_ref)), np.max(np.abs(ey_ref)))
    np.testing.assert_allclose(fm.ex, ex_ref, atol=1.0e-10 * scale)
    np.testing.assert_allclose(fm.ey, ey_ref, atol=1.0e-10 * scale)


def test_analytic_two_particle_symmetry_about_xaxis():
    radius = 1.0e-3
    solver = AnalyticMirrorChargeSpaceChargeSolver(radius=radius)
    x = np.array([0.3e-3, 0.3e-3])
    y = np.array([+0.2e-3, -0.2e-3])
    fm = solver.solve(_slice(x, y, lam=np.array([1.0e-12, 1.0e-12])))

    np.testing.assert_allclose(fm.ex[0], fm.ex[1], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fm.ey[0], -fm.ey[1], rtol=1e-10, atol=1e-10)


def test_analytic_field_at_size_mismatch_raises():
    solver = AnalyticMirrorChargeSpaceChargeSolver(radius=1.0e-3)
    x = np.array([0.0, 1.0e-4])
    y = np.array([0.0, 0.0])
    fm = solver.solve(_slice(x, y))

    with pytest.raises(NotImplementedError):
        fm.field_at(np.array([0.0]), np.array([0.0]))


def test_analytic_skips_wall_hit_sources():
    radius = 1.0e-3
    solver = AnalyticMirrorChargeSpaceChargeSolver(radius=radius)
    x = np.array([0.0, 0.4e-3])
    y = np.array([0.0, 0.0])
    lam = np.array([1.0e-12, 1.0e-12])
    base = _slice(x, y, lam=lam)
    fm_full = solver.solve(base)

    from partdist import ParticleArrayQuantity

    masked = SliceDistribution(
        z=0.0,
        x=x,
        y=y,
        px=base.px,
        py=base.py,
        pz=base.pz,
        t=base.t,
        lam=lam,
        extras={
            "wall_hit": ParticleArrayQuantity(
                name="wall_hit",
                data=np.array([False, True]),
                dtype_kind="bool",
            )
        },
    )
    fm_masked = solver.solve(masked)

    assert not np.allclose(fm_full.ex, fm_masked.ex)
    assert fm_masked.ex[1] == 0.0 and fm_masked.ey[1] == 0.0
