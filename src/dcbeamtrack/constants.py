from scipy.constants import c as LIGHT_SPEED
from scipy.constants import e as ELEMENTARY_CHARGE_ABS
from scipy.constants import epsilon_0 as EPS0
from scipy.constants import m_e as ELECTRON_MASS

ELECTRON_CHARGE = -ELEMENTARY_CHARGE_ABS
ELECTRON_Q_OVER_M = ELECTRON_CHARGE / ELECTRON_MASS

__all__ = [
    "ELECTRON_CHARGE",
    "ELECTRON_MASS",
    "ELECTRON_Q_OVER_M",
    "ELEMENTARY_CHARGE_ABS",
    "EPS0",
    "LIGHT_SPEED",
]
