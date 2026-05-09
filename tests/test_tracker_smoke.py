import numpy as np
from partdist import SliceDistribution

from dcbeamtrack import (
    DcBeamSimulation,
    RectGrid,
    SliceTrackResult,
    TrackConfig,
    UniformMagneticField,
)
from dcbeamtrack.tracking import velocity_to_momentum_evc


def test_track_slice_keeps_reference_z_plane_and_returns_snapshots():
    grid = RectGrid.from_size(width=1.0e-3, height=1.0e-3, spacing=0.25e-3)
    vz0 = 1.0e7
    px, py, pz = velocity_to_momentum_evc(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([vz0, vz0]),
    )
    dist = SliceDistribution(
        z=0.0,
        x=np.array([-1.0e-5, 1.0e-5]),
        y=np.array([0.0, 0.0]),
        px=px,
        py=py,
        pz=pz,
        t=np.array([0.0, 0.0]),
        lam=np.array([0.0, 0.0]),
    )
    config = TrackConfig(dt=1.0e-12, steps=3, z0=0.0, vz0=vz0)

    simulation = DcBeamSimulation.rect_fdm(
        grid=grid,
        magnetic_field=UniformMagneticField(),
        config=config,
    )
    result = simulation.run(dist)

    assert isinstance(result, SliceTrackResult)
    assert len(result) == 4
    assert np.isclose(result.final.z, vz0 * config.dt * config.steps)
    assert np.allclose(result[-1].x, dist.x)
    assert np.allclose(result[-1].y, dist.y)
    assert np.allclose(result.z, np.array([0.0, 1.0e-5, 2.0e-5, 3.0e-5]))


def test_slice_track_result_envelope_waist_and_interpolation():
    grid = RectGrid.from_size(width=1.0e-3, height=1.0e-3, spacing=0.25e-3)
    vz0 = 1.0e7
    vx0 = np.array([1.0e5, -1.0e5])
    vy0 = np.array([0.0, 0.0])
    px, py, pz = velocity_to_momentum_evc(vx0, vy0, np.array([vz0, vz0]))
    dist = SliceDistribution(
        z=0.0,
        x=np.array([-1.0e-5, 1.0e-5]),
        y=np.array([0.0, 0.0]),
        px=px,
        py=py,
        pz=pz,
        t=np.array([0.0, 0.0]),
        lam=np.array([1.0e-20, 1.0e-20]),
    )
    config = TrackConfig(
        dt=1.0e-12,
        steps=2,
        z0=0.0,
        vz0=vz0,
    )

    simulation = DcBeamSimulation.rect_fdm(
        grid=grid,
        magnetic_field=UniformMagneticField(),
        config=config,
    )
    result = simulation.run(dist)
    z, rms = result.radial_envelope("rms")
    waist_z, waist_r = result.waist("rms")
    mid = result.get_at_z(0.5 * (z[0] + z[1]))

    assert z.shape == rms.shape
    assert waist_z in z
    assert np.isclose(waist_r, np.min(rms))
    assert np.isclose(mid.z, 0.5e-5)
    assert len(mid) == result.n_particles
    assert np.all(np.isfinite(result.array("x")))
