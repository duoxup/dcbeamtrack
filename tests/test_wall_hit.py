import numpy as np
from partdist import ParticleArrayQuantity, SliceDistribution

from dcbeamtrack import (
    RectBoundary,
    TrackConfig,
    UnboundedBoundary,
    UniformMagneticField,
    track_slice,
)
from dcbeamtrack.tracking import velocity_to_momentum_evc


class _ZeroFieldMap:
    def field_at(self, x, y):
        x_arr = np.asarray(x, dtype=float)
        return np.zeros_like(x_arr), np.zeros_like(x_arr)


class _ZeroSolver:
    def solve(self, distribution):
        return _ZeroFieldMap()


def _slice(x, y, vx, vy, vz, *, extras=None):
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
        extras=extras,
    )


def _config(steps: int = 3) -> TrackConfig:
    return TrackConfig(dt=1.0e-12, steps=steps, z0=0.0, vz0=1.0e7)


def test_wall_hit_flags_absorbed_particles():
    initial = _slice(
        np.array([0.0, 2.0e-4]),
        np.array([0.0, 0.0]),
        np.array([1.0e5, 1.0e5]),
        np.array([0.0, 0.0]),
        np.array([1.0e7, 1.0e7]),
    )
    boundary = RectBoundary(xmin=-1.0e-4, xmax=1.0e-4, ymin=-1.0e-4, ymax=1.0e-4)

    result = track_slice(
        initial,
        spacecharge_solver=_ZeroSolver(),
        magnetic_field=UniformMagneticField(),
        config=_config(),
        boundary=boundary,
    )

    final_hit = np.asarray(result.final.get_data("wall_hit"), dtype=bool)
    assert final_hit.tolist() == [False, True]


def test_wall_hit_freezes_particle_at_wall():
    """Once a particle is absorbed, subsequent steps must not move it further."""
    initial = _slice(
        np.array([0.9e-4]),
        np.array([0.0]),
        np.array([1.0e6]),
        np.array([0.0]),
        np.array([1.0e7]),
    )
    boundary = RectBoundary(xmin=-1.0e-4, xmax=1.0e-4, ymin=-1.0e-4, ymax=1.0e-4)

    result = track_slice(
        initial,
        spacecharge_solver=_ZeroSolver(),
        magnetic_field=UniformMagneticField(),
        config=_config(steps=10),
        boundary=boundary,
    )

    wall_hit_history = result.array("wall_hit")
    assert wall_hit_history[0, 0] == 0.0
    assert wall_hit_history[-1, 0] == 1.0
    hit_step = int(np.argmax(wall_hit_history[:, 0] > 0.5))
    assert hit_step >= 1
    assert np.allclose(result.array("x")[hit_step:, 0], boundary.xmax)


def test_unbounded_boundary_keeps_all_alive():
    initial = _slice(
        np.array([0.0, 2.0e-4]),
        np.array([0.0, 0.0]),
        np.array([1.0e5, 1.0e5]),
        np.array([0.0, 0.0]),
        np.array([1.0e7, 1.0e7]),
    )

    result = track_slice(
        initial,
        spacecharge_solver=_ZeroSolver(),
        magnetic_field=UniformMagneticField(),
        config=_config(),
        boundary=UnboundedBoundary(),
    )

    final_hit = np.asarray(result.final.get_data("wall_hit"), dtype=bool)
    assert not final_hit.any()
    assert result.live_mask().all()


def test_initial_wall_hit_is_respected():
    initial = _slice(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0e5, 1.0e5]),
        np.array([0.0, 0.0]),
        np.array([1.0e7, 1.0e7]),
        extras={
            "wall_hit": ParticleArrayQuantity(
                name="wall_hit",
                data=np.array([False, True]),
                dtype_kind="bool",
            )
        },
    )

    result = track_slice(
        initial,
        spacecharge_solver=_ZeroSolver(),
        magnetic_field=UniformMagneticField(),
        config=_config(steps=5),
        boundary=UnboundedBoundary(),
    )

    final_x = result.final.x
    assert final_x[0] > 0.0
    assert final_x[1] == 0.0
    assert np.asarray(result.final.get_data("wall_hit"), dtype=bool).tolist() == [False, True]


def test_live_mask_matches_wall_hit_accumulation():
    initial = _slice(
        np.array([0.0, 2.0e-4, 0.0]),
        np.array([0.0, 0.0, 2.0e-4]),
        np.array([1.0e5, 1.0e5, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0e7, 1.0e7, 1.0e7]),
    )
    boundary = RectBoundary(xmin=-1.0e-4, xmax=1.0e-4, ymin=-1.0e-4, ymax=1.0e-4)

    result = track_slice(
        initial,
        spacecharge_solver=_ZeroSolver(),
        magnetic_field=UniformMagneticField(),
        config=_config(steps=4),
        boundary=boundary,
    )

    live = result.live_mask()
    assert live[0, 0] and live[0, 1] and live[0, 2]
    assert live[-1, 0]
    assert not live[-1, 1]
    assert not live[-1, 2]
    monotone = np.all(np.diff(live.astype(int), axis=0) <= 0)
    assert monotone
