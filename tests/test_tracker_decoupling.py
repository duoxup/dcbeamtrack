import numpy as np
from partdist import SliceDistribution

from dcbeamtrack import RectBoundary, TrackConfig, UniformMagneticField, track_slice
from dcbeamtrack.tracking import velocity_to_momentum_evc


class ZeroFieldMap:
    def field_at(self, x, y):
        x_arr = np.asarray(x, dtype=float)
        return np.zeros_like(x_arr), np.zeros_like(x_arr)


class RecordingZeroSolver:
    def __init__(self):
        self.z_seen = []

    def solve(self, distribution):
        self.z_seen.append(float(distribution.z))
        return ZeroFieldMap()


def _slice_from_velocity(x, y, vx, vy, vz):
    px, py, pz = velocity_to_momentum_evc(vx, vy, vz)
    return SliceDistribution(
        z=0.0,
        x=x,
        y=y,
        px=px,
        py=py,
        pz=pz,
        t=np.zeros_like(x, dtype=float),
        lam=np.zeros_like(x, dtype=float),
    )


def test_track_slice_accepts_solver_without_rect_grid():
    solver = RecordingZeroSolver()
    vx = np.array([1.0e5, -2.0e5])
    vy = np.array([0.5e5, 0.25e5])
    vz = np.array([1.0e7, 1.0e7])
    initial = _slice_from_velocity(
        np.array([1.0e-5, -1.0e-5]),
        np.array([0.0, 2.0e-5]),
        vx,
        vy,
        vz,
    )
    config = TrackConfig(
        dt=1.0e-12,
        steps=3,
        z0=0.0,
        vz0=1.0e7,
    )

    result = track_slice(
        initial,
        magnetic_field=UniformMagneticField(),
        config=config,
        spacecharge_solver=solver,
    )

    np.testing.assert_allclose(solver.z_seen, [1.0e-5, 2.0e-5, 3.0e-5])
    np.testing.assert_allclose(result.final.x, initial.x + vx * config.dt * config.steps)
    np.testing.assert_allclose(result.final.y, initial.y + vy * config.dt * config.steps)


def test_track_slice_uses_supplied_boundary_without_rect_grid():
    solver = RecordingZeroSolver()
    vx = np.array([1.0e5, 1.0e5])
    vy = np.array([0.0, 0.0])
    vz = np.array([1.0e7, 1.0e7])
    initial = _slice_from_velocity(
        np.array([0.0, 2.0e-4]),
        np.array([0.0, 0.0]),
        vx,
        vy,
        vz,
    )
    config = TrackConfig(dt=1.0e-12, steps=2, z0=0.0, vz0=1.0e7)
    boundary = RectBoundary(xmin=-1.0e-4, xmax=1.0e-4, ymin=-1.0e-4, ymax=1.0e-4)

    result = track_slice(
        initial,
        magnetic_field=UniformMagneticField(),
        config=config,
        spacecharge_solver=solver,
        boundary=boundary,
    )

    np.testing.assert_allclose(result.final.x[0], vx[0] * config.dt * config.steps)
    assert result.final.x[1] == boundary.xmax
    assert result.final.vx[1] == 0.0
