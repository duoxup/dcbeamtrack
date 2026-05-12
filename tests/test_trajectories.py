import math

import numpy as np
import pytest
from partdist import SliceDistribution

from dcbeamtrack import SliceTrackResult
from dcbeamtrack.tracking.result import ParticleTrajectories


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


def test_particle_trajectories_dataclass_fields():
    z = np.array([0.0, 1.0])
    data = {"x": np.array([[0.0, 1.0], [2.0, 3.0]])}
    live = np.array([[True, True], [True, False]])
    mask = np.array([True, False, True])
    keys = ("x",)
    traj = ParticleTrajectories(z=z, data=data, live=live, mask=mask, keys=keys)
    assert traj.z.shape == (2,)
    assert traj.data["x"].shape == (2, 2)
    assert traj.live.shape == (2, 2)
    assert traj.mask.shape == (3,)
    assert traj.keys == ("x",)


def test_particle_trajectories_is_frozen():
    z = np.zeros(1)
    traj = ParticleTrajectories(z=z, data={}, live=np.zeros((1, 0), dtype=bool),
                                mask=np.zeros(0, dtype=bool), keys=())
    with pytest.raises(Exception):
        traj.z = np.ones(1)
