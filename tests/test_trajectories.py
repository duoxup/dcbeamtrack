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


def test_trajectories_default_keys_shapes_and_values():
    # 3 particles, 2 snapshots, all live.
    s0 = _snap(0.0, [0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
    s1 = _snap(1.0, [3.0, 4.0, 5.0], [13.0, 14.0, 15.0])
    result = SliceTrackResult((s0, s1))

    mask = np.array([True, False, True])
    traj = result.trajectories(mask)

    # Default keys: ("x", "y", "px", "py", "pz").
    assert traj.keys == ("x", "y", "px", "py", "pz")
    assert set(traj.data.keys()) == {"x", "y", "px", "py", "pz"}

    # Shapes.
    assert traj.z.shape == (2,)
    for key in ("x", "y", "px", "py", "pz"):
        assert traj.data[key].shape == (2, 2)
    assert traj.live.shape == (2, 2)
    assert traj.mask.shape == (3,)
    assert traj.mask.dtype == bool

    # Values for x and y on selected particles 0 and 2.
    np.testing.assert_array_equal(traj.data["x"], np.array([[0.0, 2.0], [3.0, 5.0]]))
    np.testing.assert_array_equal(traj.data["y"], np.array([[10.0, 12.0], [13.0, 15.0]]))

    # mask field is a copy, not the input array.
    assert traj.mask is not mask
    np.testing.assert_array_equal(traj.mask, mask)
