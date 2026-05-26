import numpy as np
import pytest
from partdist import SliceDistribution

from dcbeamtrack import (
    CircularBoundary,
    DcBeamSimulation,
    RadialElectricField,
    SliceTrackResult,
    TrackConfig,
    UniformElectricField,
    UniformMagneticField,
    ZeroElectricField,
)
from dcbeamtrack.constants import ELECTRON_MASS, ELEMENTARY_CHARGE_ABS
from dcbeamtrack.spacecharge import AnalyticMirrorChargeSpaceChargeSolver
from dcbeamtrack.tracking import track_slice, velocity_to_momentum_evc


# ---------------------------------------------------------------- protocol shapes


def test_zero_electric_field_returns_zeros_with_matching_shape():
    f = ZeroElectricField()
    x = np.linspace(-1.0, 1.0, 5)
    y = np.linspace(-2.0, 2.0, 5)
    z = np.full_like(x, 0.1)
    ex, ey, ez = f.at(x, y, z)
    assert ex.shape == ey.shape == ez.shape == x.shape
    assert np.all(ex == 0.0) and np.all(ey == 0.0) and np.all(ez == 0.0)
    assert f.summary() == {"type": "ZeroElectricField"}


def test_uniform_electric_field_broadcasts_components():
    f = UniformElectricField(ex=1.0e3, ey=-2.0e3, ez=5.0)
    x = np.linspace(-1e-3, 1e-3, 4)
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    ex, ey, ez = f.at(x, y, z)
    assert np.all(ex == 1.0e3) and np.all(ey == -2.0e3) and np.all(ez == 5.0)
    assert f.summary()["type"] == "UniformElectricField"


# ---------------------------------------------------------------- radial: init logic


def test_radial_electric_field_prepends_origin_if_missing():
    f = RadialElectricField(
        r_m=np.array([1.0e-3, 2.0e-3, 3.0e-3]),
        er_v_per_m=np.array([10.0, 20.0, 30.0]),
    )
    assert f.r_m[0] == 0.0
    assert f.er_v_per_m[0] == 0.0
    assert f.r_m.size == 4
    assert f.er_v_per_m.size == 4


def test_radial_electric_field_rejects_nonzero_er_at_axis():
    with pytest.raises(ValueError, match="Er\\(r=0\\)"):
        RadialElectricField(
            r_m=np.array([0.0, 1.0e-3]),
            er_v_per_m=np.array([5.0, 10.0]),
        )


def test_radial_electric_field_accepts_zero_er_at_axis():
    f = RadialElectricField(
        r_m=np.array([0.0, 1.0e-3, 2.0e-3]),
        er_v_per_m=np.array([0.0, 50.0, 100.0]),
    )
    assert f.r_m.size == 3
    assert f.er_v_per_m[0] == 0.0


def test_radial_electric_field_validation_errors():
    with pytest.raises(ValueError, match="same length"):
        RadialElectricField(r_m=np.array([0.0, 1.0]), er_v_per_m=np.array([0.0]))
    with pytest.raises(ValueError, match="non-negative"):
        RadialElectricField(
            r_m=np.array([-1.0e-3, 1.0e-3]),
            er_v_per_m=np.array([0.0, 5.0]),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        RadialElectricField(
            r_m=np.array([1.0e-3, 1.0e-3, 2.0e-3]),
            er_v_per_m=np.array([10.0, 20.0, 30.0]),
        )


# ---------------------------------------------------------------- radial: at(x,y,z)


def test_radial_electric_field_at_origin_returns_zero():
    f = RadialElectricField(
        r_m=np.array([1.0e-3, 2.0e-3]),
        er_v_per_m=np.array([100.0, 200.0]),
    )
    ex, ey, ez = f.at(np.array([0.0]), np.array([0.0]), np.array([0.0]))
    assert ex[0] == 0.0 and ey[0] == 0.0 and ez[0] == 0.0


def test_radial_electric_field_cartesian_projection():
    """Er(r) projected to Cartesian must satisfy (Ex, Ey) || (x, y) and |E| == Er(r)."""
    er_at_1mm = 1.0e3
    f = RadialElectricField(
        r_m=np.array([1.0e-3]),
        er_v_per_m=np.array([er_at_1mm]),
    )
    angles = np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False)
    r0 = 1.0e-3
    x = r0 * np.cos(angles)
    y = r0 * np.sin(angles)
    z = np.zeros_like(x)
    ex, ey, ez = f.at(x, y, z)
    np.testing.assert_allclose(np.hypot(ex, ey), er_at_1mm, rtol=1e-12)
    np.testing.assert_allclose(ex, er_at_1mm * np.cos(angles), atol=1e-9)
    np.testing.assert_allclose(ey, er_at_1mm * np.sin(angles), atol=1e-9)
    assert np.all(ez == 0.0)


def test_radial_electric_field_linear_profile_interpolates():
    """Linear Er(r) = k*r should give dEx/dx = k at axis (linear focusing)."""
    k = 1.0e6
    r = np.linspace(0.0, 1.0e-3, 11)
    f = RadialElectricField(r_m=r, er_v_per_m=k * r)
    x = np.array([0.5e-3])
    y = np.array([0.0])
    z = np.array([0.0])
    ex, ey, ez = f.at(x, y, z)
    np.testing.assert_allclose(ex[0], k * x[0], rtol=1e-10)


def test_radial_electric_field_fill_value_outside_table():
    f = RadialElectricField(
        r_m=np.array([1.0e-3, 2.0e-3]),
        er_v_per_m=np.array([100.0, 200.0]),
        fill_value=0.0,
    )
    x = np.array([5.0e-3])
    y = np.array([0.0])
    z = np.array([0.0])
    ex, ey, ez = f.at(x, y, z)
    assert ex[0] == 0.0 and ey[0] == 0.0


def test_radial_electric_field_summary_keys():
    f = RadialElectricField(
        r_m=np.array([0.0, 1.0e-3, 2.0e-3]),
        er_v_per_m=np.array([0.0, 100.0, 200.0]),
    )
    s = f.summary()
    assert s["type"] == "RadialElectricField"
    assert s["n_samples"] == 3
    assert s["r_min_m"] == 0.0
    assert s["r_max_m"] == 2.0e-3
    assert s["fill_value"] == 0.0


def test_radial_electric_field_from_arrays_mm():
    f = RadialElectricField.from_arrays(
        r=np.array([1.0, 2.0]),
        er=np.array([100.0, 200.0]),
        r_unit="mm",
    )
    assert np.isclose(f.r_m[-1], 2.0e-3)


# ---------------------------------------------------------------- tracker integration


def _make_slice(n: int, vz0: float, rng_seed: int = 0) -> SliceDistribution:
    rng = np.random.default_rng(rng_seed)
    x = rng.uniform(-5.0e-5, 5.0e-5, size=n)
    y = rng.uniform(-5.0e-5, 5.0e-5, size=n)
    vx = vy = np.zeros(n)
    vz = np.full(n, vz0)
    px, py, pz = velocity_to_momentum_evc(vx, vy, vz)
    return SliceDistribution(
        z=0.0, x=x, y=y, px=px, py=py, pz=pz, t=np.zeros(n), lam=np.full(n, 1.0e-12)
    )


def _no_solver():
    class _NullField:
        def field_at(self, x, y):
            return np.zeros_like(x), np.zeros_like(x)

        def summary(self):
            return {"type": "Null"}

    class _NullSolver:
        def solve(self, distribution):
            return _NullField()

        def summary(self):
            return {"type": "NullSpaceChargeSolver"}

    return _NullSolver()


def test_track_slice_zero_electric_field_matches_no_electric_field():
    config = TrackConfig(dt=1.0e-12, steps=4, z0=0.0, vz0=1.0e7, snapshot_stride=1)
    initial = _make_slice(8, vz0=config.vz0)

    r_default = track_slice(
        initial,
        spacecharge_solver=_no_solver(),
        magnetic_field=UniformMagneticField(),
        config=config,
    )
    r_zero = track_slice(
        initial,
        spacecharge_solver=_no_solver(),
        magnetic_field=UniformMagneticField(),
        config=config,
        electric_field=ZeroElectricField(),
    )
    np.testing.assert_allclose(r_default.final.x, r_zero.final.x)
    np.testing.assert_allclose(r_default.final.y, r_zero.final.y)
    np.testing.assert_allclose(r_default.final.pz, r_zero.final.pz)


def test_track_slice_uniform_ex_deflects_beam_in_plus_x():
    """Electrons (charge -e) in +Ex feel force in -x; verify sign by relative deflection."""
    n = 16
    config = TrackConfig(dt=1.0e-12, steps=10, z0=0.0, vz0=1.0e7, snapshot_stride=1)
    initial = _make_slice(n, vz0=config.vz0, rng_seed=1)

    r_off = track_slice(
        initial,
        spacecharge_solver=_no_solver(),
        magnetic_field=UniformMagneticField(),
        config=config,
    )
    r_on = track_slice(
        initial,
        spacecharge_solver=_no_solver(),
        magnetic_field=UniformMagneticField(),
        config=config,
        electric_field=UniformElectricField(ex=1.0e6),
    )
    # Electron sees F = -e*E = -|e|*Ex*x̂ so beam shifts in -x direction.
    assert np.mean(r_on.final.x) < np.mean(r_off.final.x) - 1.0e-8
    np.testing.assert_allclose(r_on.final.y, r_off.final.y, atol=1.0e-14)


def test_track_slice_uniform_ez_accelerates_beam():
    n = 8
    vz0 = 1.0e7
    config = TrackConfig(dt=1.0e-12, steps=50, z0=0.0, vz0=vz0, snapshot_stride=1)
    initial = _make_slice(n, vz0=vz0, rng_seed=2)

    r_on = track_slice(
        initial,
        spacecharge_solver=_no_solver(),
        magnetic_field=UniformMagneticField(),
        config=config,
        electric_field=UniformElectricField(ez=1.0e6),
    )
    # F_z = -e * Ez (electron); +Ez decelerates electrons, so vz decreases.
    pz_final = np.mean(r_on.final.pz)
    pz_initial = np.mean(initial.pz)
    assert pz_final < pz_initial


def test_track_slice_radial_focusing_compresses_beam():
    """Negative Er (pointing inward toward axis) attracts electrons outward.

    For electrons with q=-e: F_r = -e * Er. If Er > 0 (outward), F_r is inward
    (focusing). The test uses Er > 0 (linear in r) and confirms RMS radius shrinks.
    """
    n = 64
    config = TrackConfig(dt=1.0e-12, steps=200, z0=0.0, vz0=1.0e7, snapshot_stride=1)
    initial = _make_slice(n, vz0=config.vz0, rng_seed=3)

    r_table = np.linspace(0.0, 1.0e-4, 21)
    er_table = 1.0e7 * r_table  # outward Er, focusing for electrons
    field = RadialElectricField(r_m=r_table, er_v_per_m=er_table)

    r_off = track_slice(
        initial,
        spacecharge_solver=_no_solver(),
        magnetic_field=UniformMagneticField(),
        config=config,
    )
    r_on = track_slice(
        initial,
        spacecharge_solver=_no_solver(),
        magnetic_field=UniformMagneticField(),
        config=config,
        electric_field=field,
    )
    rms_off = np.sqrt(np.mean(r_off.final.x ** 2 + r_off.final.y ** 2))
    rms_on = np.sqrt(np.mean(r_on.final.x ** 2 + r_on.final.y ** 2))
    assert rms_on < rms_off


# ---------------------------------------------------------------- simulation factory


def test_dcbeamsimulation_accepts_electric_field_via_factory_and_records_metadata():
    radius = 1.0e-3
    config = TrackConfig(dt=1.0e-12, steps=4, z0=0.0, vz0=1.0e7, snapshot_stride=1)
    field = UniformElectricField(ex=0.0, ey=0.0, ez=1.0e3)
    simulation = DcBeamSimulation.analytic_mirror_charge(
        radius=radius,
        magnetic_field=UniformMagneticField(),
        config=config,
        electric_field=field,
    )
    assert simulation.electric_field is field

    initial = _make_slice(6, vz0=config.vz0, rng_seed=11)
    result = simulation.run(initial)

    assert isinstance(result, SliceTrackResult)
    assert result.metadata["electric_field"]["type"] == "UniformElectricField"
    assert result.metadata["electric_field"]["ez"] == 1.0e3


def test_dcbeamsimulation_default_electric_field_is_zero_in_metadata():
    radius = 1.0e-3
    config = TrackConfig(dt=1.0e-12, steps=2, z0=0.0, vz0=1.0e7, snapshot_stride=1)
    simulation = DcBeamSimulation.analytic_mirror_charge(
        radius=radius,
        magnetic_field=UniformMagneticField(),
        config=config,
    )
    initial = _make_slice(4, vz0=config.vz0, rng_seed=12)
    result = simulation.run(initial)
    assert result.metadata["electric_field"]["type"] == "ZeroElectricField"
