import numpy as np
import pytest
from partdist import SliceDistribution

from dcbeamtrack import (
    AxialMagneticField,
    CircularBoundary,
    CircularGreenSpaceChargeSolver,
    DcBeamSimulation,
    SliceTrackResult,
    TrackConfig,
    UniformMagneticField,
)
from dcbeamtrack.constants import EPS0
from dcbeamtrack.tracking import velocity_to_momentum_evc


def _direct_mirror_charge_field(x: np.ndarray, y: np.ndarray, q: np.ndarray, radius: float):
    """Reference O(N**2) mirror-charge field for a grounded circular pipe.

    Direct line-charge contribution plus a single image charge of opposite sign at
    the inverse point ``(radius**2 / r_j**2) * (x_j, y_j)``.
    """
    coef = 1.0 / (2.0 * np.pi * EPS0)
    n = x.size
    ex = np.zeros(n)
    ey = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r2 = dx * dx + dy * dy
            ex[i] += coef * q[j] * dx / r2
            ey[i] += coef * q[j] * dy / r2

            r_src2 = x[j] * x[j] + y[j] * y[j]
            scale = (radius * radius) / r_src2
            xi_img = scale * x[j]
            yi_img = scale * y[j]
            dxi = x[i] - xi_img
            dyi = y[i] - yi_img
            r2_img = dxi * dxi + dyi * dyi
            ex[i] -= coef * q[j] * dxi / r2_img
            ey[i] -= coef * q[j] * dyi / r2_img
    return ex, ey


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


def test_circular_boundary_absorb_clips_to_radius():
    boundary = CircularBoundary(radius=1.0e-4)
    x = np.array([0.0, 2.0e-4, -1.0e-4])
    y = np.array([0.0, 0.0, 1.0e-4])
    vx = np.array([1.0, 1.0, 1.0])
    vy = np.array([1.0, 1.0, 1.0])

    hit = boundary.absorb(x, y, vx, vy)

    assert hit.tolist() == [False, True, True]
    assert np.isclose(np.hypot(x[1], y[1]), boundary.radius)
    assert np.isclose(np.hypot(x[2], y[2]), boundary.radius)
    assert vx[0] == 1.0 and vy[0] == 1.0
    assert vx[1] == 0.0 and vy[1] == 0.0
    assert vx[2] == 0.0 and vy[2] == 0.0


def test_circular_boundary_rejects_non_positive_radius():
    with pytest.raises(ValueError):
        CircularBoundary(radius=0.0)


def test_circular_green_two_particle_symmetry_about_xaxis():
    radius = 1.0e-3
    solver = CircularGreenSpaceChargeSolver(radius=radius, n_r=96, n_theta=720)
    x = np.array([0.3e-3, 0.3e-3])
    y = np.array([+0.2e-3, -0.2e-3])
    fm = solver.solve(_slice(x, y, lam=np.array([1.0e-12, 1.0e-12])))

    np.testing.assert_allclose(fm.ex[0], fm.ex[1], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fm.ey[0], -fm.ey[1], rtol=1e-10, atol=1e-10)


def test_circular_green_matches_direct_mirror_charge_on_grid_nodes():
    """On ring + on spoke: radial snap and angular linear interpolation are exact."""
    radius = 1.0e-3
    solver = CircularGreenSpaceChargeSolver(radius=radius, n_r=96, n_theta=720)
    ring_idx = np.array([10, 25, 40, 60])
    spoke_idx = np.array([0, 90, 180, 540])
    r = solver.gr[ring_idx]
    theta = solver.gu[spoke_idx]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    lam = np.array([1.0e-12, 0.7e-12, 1.3e-12, 0.5e-12])

    fm = solver.solve(_slice(x, y, lam=lam))
    ex_ref, ey_ref = _direct_mirror_charge_field(x, y, lam, radius)

    scale = max(np.max(np.abs(ex_ref)), np.max(np.abs(ey_ref)))
    np.testing.assert_allclose(fm.ex, ex_ref, atol=1.0e-9 * scale)
    np.testing.assert_allclose(fm.ey, ey_ref, atol=1.0e-9 * scale)


def test_circular_green_field_at_size_mismatch_raises():
    solver = CircularGreenSpaceChargeSolver(radius=1.0e-3, n_r=32, n_theta=180)
    x = np.array([0.0, 1.0e-4])
    y = np.array([0.0, 0.0])
    fm = solver.solve(_slice(x, y))

    with pytest.raises(NotImplementedError):
        fm.field_at(np.array([0.0]), np.array([0.0]))


def test_circular_green_skips_wall_hit_sources():
    radius = 1.0e-3
    solver = CircularGreenSpaceChargeSolver(radius=radius, n_r=64, n_theta=360)
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


def test_dcbeamsimulation_circular_green_smoke():
    radius = 1.0e-3
    config = TrackConfig(dt=1.0e-12, steps=4, z0=0.0, vz0=1.0e7, snapshot_stride=1)
    simulation = DcBeamSimulation.circular_green(
        radius=radius,
        magnetic_field=UniformMagneticField(),
        config=config,
        n_r=32,
        n_theta=180,
    )

    rng = np.random.default_rng(7)
    n = 12
    r = rng.uniform(0.1 * radius, 0.5 * radius, size=n)
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    vz0 = config.vz0
    vx = vy = np.zeros(n)
    vz = np.full(n, vz0)
    px, py, pz = velocity_to_momentum_evc(vx, vy, vz)
    initial = SliceDistribution(
        z=0.0, x=x, y=y, px=px, py=py, pz=pz, t=np.zeros(n), lam=np.full(n, 1.0e-12)
    )

    result = simulation.run(initial)

    assert isinstance(result, SliceTrackResult)
    assert len(result) == config.steps + 1
    final_r = np.hypot(result.final.x, result.final.y)
    assert np.all(final_r <= radius + 1.0e-12)
    assert "wall_hit" in result.final.extra_quantity_keys
