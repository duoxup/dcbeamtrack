import numpy as np
from partdist import SliceDistribution

from dcbeamtrack import AxialMagneticField, DcBeamSimulation, RectGrid, SliceTrackResult, TrackConfig
from dcbeamtrack.tracking import velocity_to_momentum_evc


def _reference_case():
    grid = RectGrid.from_size(width=1.0e-3, height=1.0e-3, spacing=0.25e-3)
    vz0 = 5.0e7
    vx0 = np.array([1.0e5, -0.8e5, 0.25e5])
    vy0 = np.array([0.5e5, 0.2e5, -0.4e5])
    px, py, pz = velocity_to_momentum_evc(vx0, vy0, np.full(3, vz0))
    distribution = SliceDistribution(
        z=0.0,
        x=np.array([-1.0e-4, 1.1e-4, 0.2e-4]),
        y=np.array([0.6e-4, -0.4e-4, -0.9e-4]),
        px=px,
        py=py,
        pz=pz,
        t=np.zeros(3),
        lam=np.array([-1.0e-12, -0.7e-12, -0.4e-12]),
    )
    field = AxialMagneticField.from_z_bz(
        [0.0, 5.0e-4, 1.0e-3],
        [0.03, 0.031, 0.032],
        include_radial=True,
    )
    config = TrackConfig(
        dt=1.0e-12,
        steps=5,
        z0=0.0,
        vz0=vz0,
        q_over_m=-1.76e11,
        deposition="CIC",
        precision="float64",
    )
    simulation = DcBeamSimulation.rect_fdm(grid=grid, magnetic_field=field, config=config)
    return simulation, distribution


def test_rect_fdm_reference_regression():
    simulation, distribution = _reference_case()
    result = simulation.run(distribution)

    np.testing.assert_allclose(
        result.final.x,
        np.array([-9.9511801567082840e-05, 1.0960367344998703e-04, 2.0139844424511131e-05]),
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_allclose(
        result.final.y,
        np.array([6.024481590842354e-05, -3.989198773861912e-05, -9.019538038425732e-05]),
        rtol=0.0,
        atol=5.0e-16,
    )
    np.testing.assert_allclose(
        result.final.vx,
        np.array([96067.97438899709, -78786.7383674998, 29946.07902326476]),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        result.final.vy,
        np.array([48266.70406564107, 22668.116405292683, -38440.42239577168]),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        result.final.vz,
        np.array([50000009.36707548, 50000000.719031036, 49999998.53520379]),
        rtol=0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(result.z[-1], 2.5e-4, rtol=0.0, atol=1.0e-18)
    np.testing.assert_allclose(result.r_max()[-1], 1.166376265168e-4, rtol=0.0, atol=1.0e-16)
    np.testing.assert_allclose(result.r_mean()[-1], 1.118763062431e-4, rtol=0.0, atol=1.0e-16)
    np.testing.assert_allclose(result.r_rms()[-1], 1.122738989916e-4, rtol=0.0, atol=1.0e-16)


def test_simulation_api_result_npz_roundtrip(tmp_path):
    simulation, distribution = _reference_case()
    result = simulation.run(distribution)

    path = result.save_npz(tmp_path / "result.npz")
    loaded = SliceTrackResult.load_npz(path)
    loaded_via_alias = SliceTrackResult.load(path)

    assert isinstance(loaded.config, TrackConfig)
    assert loaded.config == simulation.config
    assert loaded_via_alias.config == simulation.config
    assert loaded.n_snapshots == result.n_snapshots
    assert loaded.n_particles == result.n_particles
    for key in ("x", "y", "px", "py", "pz", "t", "lam"):
        np.testing.assert_allclose(loaded.array(key), result.array(key), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.z, result.z, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.r_rms(), result.r_rms())
