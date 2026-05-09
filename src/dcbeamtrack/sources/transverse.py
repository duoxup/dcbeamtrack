from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import numpy as np


class TransverseDensity(Protocol):
    @property
    def total_linear_charge(self) -> float | None:
        """Total line charge [C/m], if known."""

    def rho(self, x, y) -> np.ndarray:
        """Surface line-charge density [C/m^2] in the transverse plane."""


@dataclass(frozen=True)
class CallableDensity:
    func: Callable
    total_linear_charge_value: float | None = None

    @property
    def total_linear_charge(self) -> float | None:
        return self.total_linear_charge_value

    def rho(self, x, y) -> np.ndarray:
        return np.asarray(self.func(x, y), dtype=float)


@dataclass(frozen=True)
class UniformDisk:
    radius: float
    current: float
    vz: float
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be positive.")
        if self.vz == 0.0:
            raise ValueError("vz must be non-zero.")

    @property
    def total_linear_charge(self) -> float:
        return float(self.current) / float(self.vz)

    @property
    def rho0(self) -> float:
        return self.total_linear_charge / (np.pi * self.radius * self.radius)

    def rho(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cx, cy = self.center
        r2 = (x - cx) ** 2 + (y - cy) ** 2
        out = np.zeros_like(x, dtype=float)
        out[r2 <= self.radius * self.radius] = self.rho0
        return out


class GaussianMixture:
    def __init__(self, beamlets: list[Mapping[str, float]], vz: float) -> None:
        if vz == 0.0:
            raise ValueError("vz must be non-zero.")
        self.vz = float(vz)
        self.beamlets = [self._normalize_beamlet(beamlet) for beamlet in beamlets]
        if not self.beamlets:
            raise ValueError("At least one beamlet is required.")

    @staticmethod
    def _normalize_beamlet(beamlet: Mapping[str, float]) -> dict[str, float]:
        current = beamlet.get("current", beamlet.get("I"))
        if current is None:
            raise KeyError("Each beamlet must contain 'current' or legacy key 'I'.")
        out = {
            "current": float(current),
            "x0": float(beamlet.get("x0", 0.0)),
            "y0": float(beamlet.get("y0", 0.0)),
            "sigma_x": float(beamlet["sigma_x"]),
            "sigma_y": float(beamlet["sigma_y"]),
        }
        if out["sigma_x"] <= 0.0 or out["sigma_y"] <= 0.0:
            raise ValueError("sigma_x and sigma_y must be positive.")
        return out

    @property
    def total_linear_charge(self) -> float:
        return sum(beamlet["current"] for beamlet in self.beamlets) / self.vz

    def rho(self, x, y) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        out = np.zeros_like(x, dtype=float)
        for beamlet in self.beamlets:
            sx = beamlet["sigma_x"]
            sy = beamlet["sigma_y"]
            line_charge = beamlet["current"] / self.vz
            amp = line_charge / (2.0 * np.pi * sx * sy)
            out += amp * np.exp(
                -((x - beamlet["x0"]) ** 2) / (2.0 * sx * sx)
                - ((y - beamlet["y0"]) ** 2) / (2.0 * sy * sy)
            )
        return out
