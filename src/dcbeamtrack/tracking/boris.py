from __future__ import annotations

import numpy as np

from dcbeamtrack.constants import ELEMENTARY_CHARGE_ABS, ELECTRON_MASS, LIGHT_SPEED


def _as_1d_float(a, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def velocity_to_momentum_evc(
    vx,
    vy,
    vz,
    *,
    mass: float = ELECTRON_MASS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vx = _as_1d_float(vx, "vx")
    vy = _as_1d_float(vy, "vy")
    vz = _as_1d_float(vz, "vz")
    speed2 = vx * vx + vy * vy + vz * vz
    if np.any(speed2 >= LIGHT_SPEED * LIGHT_SPEED):
        raise ValueError("Velocity magnitude must satisfy |v| < c.")
    gamma = 1.0 / np.sqrt(1.0 - speed2 / (LIGHT_SPEED * LIGHT_SPEED))
    factor = gamma * mass * LIGHT_SPEED / ELEMENTARY_CHARGE_ABS
    return factor * vx, factor * vy, factor * vz


def boris_push_velocity(
    vx,
    vy,
    vz,
    *,
    ex,
    ey,
    ez,
    bx,
    by,
    bz,
    q_over_m: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relativistic Boris velocity update matching the legacy velocity-form pusher."""
    vx = _as_1d_float(vx, "vx")
    vy = _as_1d_float(vy, "vy")
    vz = _as_1d_float(vz, "vz")
    ex = np.broadcast_to(np.asarray(ex, dtype=float), vx.shape)
    ey = np.broadcast_to(np.asarray(ey, dtype=float), vx.shape)
    ez = np.broadcast_to(np.asarray(ez, dtype=float), vx.shape)
    bx = np.broadcast_to(np.asarray(bx, dtype=float), vx.shape)
    by = np.broadcast_to(np.asarray(by, dtype=float), vx.shape)
    bz = np.broadcast_to(np.asarray(bz, dtype=float), vx.shape)

    speed2 = vx * vx + vy * vy + vz * vz
    if np.any(speed2 >= LIGHT_SPEED * LIGHT_SPEED):
        raise ValueError("Velocity magnitude must satisfy |v| < c before Boris push.")
    gamma = 1.0 / np.sqrt(1.0 - speed2 / (LIGHT_SPEED * LIGHT_SPEED))
    h = (q_over_m * dt) / (2.0 * gamma)

    vxm = vx + h * ex
    vym = vy + h * ey
    vzm = vz + h * ez

    tx = h * bx
    ty = h * by
    tz = h * bz
    t2 = tx * tx + ty * ty + tz * tz
    s = 2.0 / (1.0 + t2)

    vpx = vxm + (vym * tz - vzm * ty)
    vpy = vym + (vzm * tx - vxm * tz)
    vpz = vzm + (vxm * ty - vym * tx)

    vxp = vxm + s * (vpy * tz - vpz * ty)
    vyp = vym + s * (vpz * tx - vpx * tz)
    vzp = vzm + s * (vpx * ty - vpy * tx)

    return vxp + h * ex, vyp + h * ey, vzp + h * ez
