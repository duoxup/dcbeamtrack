from __future__ import annotations

import numpy as np
from partdist import ParticleArrayQuantity, SliceDistribution

from dcbeamtrack.boundary import TransverseBoundary, UnboundedBoundary
from dcbeamtrack.config import TrackConfig
from dcbeamtrack.fields import MagneticField
from dcbeamtrack.spacecharge import SpaceChargeSolver

from .boris import boris_push_velocity, velocity_to_momentum_evc
from .result import SliceTrackResult


WALL_HIT_KEY = "wall_hit"


def _wall_hit_quantity(wall_hit: np.ndarray) -> ParticleArrayQuantity:
    return ParticleArrayQuantity(
        name=WALL_HIT_KEY,
        data=wall_hit.copy(),
        unit="",
        dtype_kind="bool",
        category="other",
    )


def _dtype_from_config(config: TrackConfig) -> type[np.float32] | type[np.float64]:
    return np.float32 if config.precision == "float32" else np.float64


def _effective_steps(config: TrackConfig) -> int:
    candidates = [int(config.steps)]
    if config.Lz is not None and config.Lz > 0.0 and abs(config.vz0) > 0.0:
        candidates.append(max(0, int(np.floor(config.Lz / (abs(config.vz0) * config.dt)))))
    return max(0, min(candidates))


def _make_slice_like(
    source: SliceDistribution,
    *,
    z: float,
    x: np.ndarray,
    y: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
    t: np.ndarray,
    wall_hit: np.ndarray,
    lam: np.ndarray | None = None,
) -> SliceDistribution:
    extras = {key: source.get_quantity(key).copy() for key in source.extra_quantity_keys}
    extras[WALL_HIT_KEY] = _wall_hit_quantity(wall_hit)
    lam_array = source.lam if lam is None else lam
    return SliceDistribution(
        z=z,
        x=x.copy(),
        y=y.copy(),
        px=px.copy(),
        py=py.copy(),
        pz=pz.copy(),
        t=t.copy(),
        lam=lam_array.copy(),
        extras=extras or None,
    )


def _validate_slice(distribution: SliceDistribution) -> None:
    if not isinstance(distribution, SliceDistribution):
        raise TypeError("track_slice expects a partdist.SliceDistribution.")


def track_slice(
    distribution: SliceDistribution,
    *,
    spacecharge_solver: SpaceChargeSolver,
    magnetic_field: MagneticField,
    config: TrackConfig,
    boundary: TransverseBoundary | None = None,
) -> SliceTrackResult:
    """Track a DC-beam slice while constraining all particles to a reference z plane."""
    _validate_slice(distribution)
    if not np.isclose(distribution.z, config.z0):
        raise ValueError(
            "distribution.z must match config.z0 so the reference z plane is unique."
        )
    solver = spacecharge_solver
    active_boundary: TransverseBoundary = (
        boundary if boundary is not None else UnboundedBoundary()
    )

    dtype = _dtype_from_config(config)
    n_steps = _effective_steps(config)

    x = np.asarray(distribution.x, dtype=dtype).copy()
    y = np.asarray(distribution.y, dtype=dtype).copy()
    t = np.asarray(distribution.t, dtype=dtype).copy()
    lam = np.asarray(distribution.lam, dtype=dtype).copy()
    vx = np.asarray(distribution.vx, dtype=dtype).copy()
    vy = np.asarray(distribution.vy, dtype=dtype).copy()
    vz = np.asarray(distribution.vz, dtype=dtype).copy()
    px = np.asarray(distribution.px, dtype=dtype).copy()
    py = np.asarray(distribution.py, dtype=dtype).copy()
    pz = np.asarray(distribution.pz, dtype=dtype).copy()

    if WALL_HIT_KEY in distribution.extra_quantity_keys:
        wall_hit = np.asarray(distribution.get_data(WALL_HIT_KEY), dtype=bool).copy()
    else:
        wall_hit = np.zeros(x.shape, dtype=bool)

    history = [
        _make_slice_like(
            distribution,
            z=distribution.z,
            x=x,
            y=y,
            px=px,
            py=py,
            pz=pz,
            t=t,
            lam=lam,
            wall_hit=wall_hit,
        )
    ]

    for step in range(1, n_steps + 1):
        current_z = config.z0 + config.vz0 * config.dt * step
        z_plane = float(dtype(current_z))
        current = _make_slice_like(
            distribution,
            z=z_plane,
            x=x,
            y=y,
            px=px,
            py=py,
            pz=pz,
            t=t,
            lam=lam,
            wall_hit=wall_hit,
        )

        field_map = solver.solve(current)
        ex, ey = field_map.field_at(x, y)
        bx, by, bz = magnetic_field.at(x, y, current_z)
        alive = ~wall_hit

        vx_new, vy_new, vz_new = boris_push_velocity(
            vx,
            vy,
            vz,
            ex=ex,
            ey=ey,
            ez=np.zeros_like(ex),
            bx=bx,
            by=by,
            bz=bz,
            q_over_m=config.q_over_m,
            dt=config.dt,
        )
        vx[alive] = vx_new[alive]
        vy[alive] = vy_new[alive]
        vz[alive] = vz_new[alive]

        x[alive] += vx[alive] * config.dt
        y[alive] += vy[alive] * config.dt
        t += config.dt

        new_hits = active_boundary.absorb(x, y, vx, vy)
        wall_hit |= np.asarray(new_hits, dtype=bool)

        px, py, pz = velocity_to_momentum_evc(vx, vy, vz)

        should_snapshot = step % config.snapshot_stride == 0 or step == n_steps
        if should_snapshot:
            history.append(
                _make_slice_like(
                    distribution,
                    z=z_plane,
                    x=x,
                    y=y,
                    px=px,
                    py=py,
                    pz=pz,
                    t=t,
                    lam=lam,
                    wall_hit=wall_hit,
                )
            )

    return SliceTrackResult(tuple(history), config=config)
