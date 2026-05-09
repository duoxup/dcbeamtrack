from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .constants import ELECTRON_Q_OVER_M


@dataclass(frozen=True)
class TrackConfig:
    dt: float
    steps: int
    z0: float
    vz0: float
    Lz: float | None = None
    q_over_m: float = ELECTRON_Q_OVER_M
    deposition: Literal["NGP", "CIC", "TSC"] = "CIC"
    snapshot_stride: int = 1
    precision: Literal["float64", "float32"] = "float64"

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")
        if self.snapshot_stride < 1:
            raise ValueError("snapshot_stride must be >= 1.")
        if self.deposition.upper() not in {"NGP", "CIC", "TSC"}:
            raise ValueError("deposition must be one of {'NGP', 'CIC', 'TSC'}.")
        if self.Lz is not None and self.Lz < 0.0:
            raise ValueError("Lz must be non-negative when provided.")
        if self.precision not in {"float64", "float32"}:
            raise ValueError("precision must be either 'float64' or 'float32'.")
