import numpy as np

from dcbeamtrack.constants import ELECTRON_Q_OVER_M
from dcbeamtrack.tracking import boris_push_velocity


def test_boris_no_field_leaves_velocity_unchanged():
    vx = np.array([1.0e5, -2.0e5])
    vy = np.array([3.0e5, 4.0e5])
    vz = np.array([1.0e7, 1.1e7])

    out = boris_push_velocity(
        vx,
        vy,
        vz,
        ex=np.zeros_like(vx),
        ey=np.zeros_like(vx),
        ez=np.zeros_like(vx),
        bx=np.zeros_like(vx),
        by=np.zeros_like(vx),
        bz=np.zeros_like(vx),
        q_over_m=ELECTRON_Q_OVER_M,
        dt=1e-12,
    )

    assert np.allclose(out[0], vx)
    assert np.allclose(out[1], vy)
    assert np.allclose(out[2], vz)


def test_boris_pure_bz_conserves_speed_and_vz():
    vx = np.array([1.0e5, -2.0e5])
    vy = np.array([3.0e5, 4.0e5])
    vz = np.array([1.0e7, 1.1e7])
    speed0 = np.sqrt(vx * vx + vy * vy + vz * vz)

    vx1, vy1, vz1 = boris_push_velocity(
        vx,
        vy,
        vz,
        ex=np.zeros_like(vx),
        ey=np.zeros_like(vx),
        ez=np.zeros_like(vx),
        bx=np.zeros_like(vx),
        by=np.zeros_like(vx),
        bz=np.full_like(vx, 0.05),
        q_over_m=ELECTRON_Q_OVER_M,
        dt=1e-11,
    )
    speed1 = np.sqrt(vx1 * vx1 + vy1 * vy1 + vz1 * vz1)

    assert np.allclose(vz1, vz)
    assert np.allclose(speed1, speed0)
