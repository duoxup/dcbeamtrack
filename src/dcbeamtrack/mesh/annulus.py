from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .cell import CellGrid


def _elliptic_annular_sector_area_centroid(
    *,
    a: float,
    b: float,
    r_in: float,
    r_out: float,
    theta0: float,
    theta1: float,
    xc0: float,
    yc0: float,
) -> tuple[float, float, float]:
    dtheta = theta1 - theta0
    if dtheta <= 0.0:
        dtheta = dtheta % (2.0 * np.pi)
        if dtheta == 0.0:
            dtheta = 2.0 * np.pi

    area = a * b * 0.5 * (r_out * r_out - r_in * r_in) * dtheta
    if np.isclose(dtheta, 2.0 * np.pi):
        return area, xc0, yc0

    r3 = (r_out**3 - r_in**3) / 3.0
    icos = np.sin(theta1) - np.sin(theta0)
    isin = np.cos(theta0) - np.cos(theta1)
    cx = xc0 + (a * a * b * r3 * icos) / area
    cy = yc0 + (a * b * b * r3 * isin) / area
    return area, cx, cy


class AnnulusGrid(CellGrid):
    """Elliptic annular-sector cell grid."""

    def __init__(
        self,
        *,
        domain_bounds: tuple[tuple[float, float], tuple[float, float]],
        radial_scales,
        n_angular,
        center: Optional[Tuple[float, float]] = None,
        theta_offset: float = 0.0,
    ) -> None:
        (xmin, xmax), (ymin, ymax) = domain_bounds
        if not (xmax > xmin and ymax > ymin):
            raise ValueError("Invalid domain_bounds.")
        a = 0.5 * (xmax - xmin)
        b = 0.5 * (ymax - ymin)
        if center is None:
            xc0 = 0.5 * (xmin + xmax)
            yc0 = 0.5 * (ymin + ymax)
        else:
            xc0, yc0 = center

        radial_scales = np.asarray(radial_scales, dtype=float).reshape(-1)
        if radial_scales.size < 2:
            raise ValueError("radial_scales must contain at least two entries.")
        if np.any(radial_scales < 0.0) or np.any(radial_scales > 1.0):
            raise ValueError("radial_scales must lie in [0, 1].")
        if np.any(np.diff(radial_scales) <= 0.0):
            raise ValueError("radial_scales must be strictly increasing.")

        n_rings = radial_scales.size - 1
        if np.isscalar(n_angular):
            n_ang = np.full(n_rings, int(n_angular), dtype=int)
        else:
            n_ang = np.asarray(n_angular, dtype=int).reshape(-1)
            if n_ang.size != n_rings:
                raise ValueError("n_angular must be scalar or have one entry per ring.")
        if np.any(n_ang < 1):
            raise ValueError("All angular cell counts must be >= 1.")

        polygons = []
        areas = []
        centers = []
        for ring in range(n_rings):
            r_in = float(radial_scales[ring])
            r_out = float(radial_scales[ring + 1])
            theta_edges = (
                np.linspace(0.0, 2.0 * np.pi, int(n_ang[ring]) + 1) + float(theta_offset)
            )
            for k in range(int(n_ang[ring])):
                th0 = theta_edges[k]
                th1 = theta_edges[k + 1]
                poly = self._sector_polygon(a, b, r_in, r_out, th0, th1, xc0, yc0)
                area, cx, cy = _elliptic_annular_sector_area_centroid(
                    a=a,
                    b=b,
                    r_in=r_in,
                    r_out=r_out,
                    theta0=th0,
                    theta1=th1,
                    xc0=xc0,
                    yc0=yc0,
                )
                polygons.append(poly)
                areas.append(area)
                centers.append((cx, cy))

        centers_arr = np.asarray(centers, dtype=float)
        super().__init__(
            grid_type="annulus",
            xc=centers_arr[:, 0],
            yc=centers_arr[:, 1],
            areas=np.asarray(areas, dtype=float),
            polygons=tuple(polygons),
        )
        object.__setattr__(self, "domain_bounds", domain_bounds)
        object.__setattr__(self, "radial_scales", radial_scales)
        object.__setattr__(self, "n_angular", n_ang)

    @staticmethod
    def _sector_polygon(a, b, r_in, r_out, th0, th1, xc0, yc0) -> np.ndarray:
        x_out = xc0 + a * r_out * np.cos([th0, th1])
        y_out = yc0 + b * r_out * np.sin([th0, th1])
        x_in = xc0 + a * r_in * np.cos([th1, th0])
        y_in = yc0 + b * r_in * np.sin([th1, th0])
        return np.array(
            [
                [x_out[0], y_out[0]],
                [x_out[1], y_out[1]],
                [x_in[0], y_in[0]],
                [x_in[1], y_in[1]],
            ],
            dtype=float,
        )

    @classmethod
    def from_equal_area(
        cls,
        *,
        radius: float,
        n_rings: int,
        n_angular,
        center: Tuple[float, float] = (0.0, 0.0),
        theta_offset: float = 0.0,
        inner_radius: float = 0.0,
    ) -> "AnnulusGrid":
        if radius <= 0.0:
            raise ValueError("radius must be positive.")
        if inner_radius < 0.0 or inner_radius >= radius:
            raise ValueError("inner_radius must satisfy 0 <= inner_radius < radius.")
        if np.isscalar(n_angular):
            if n_rings < 1:
                raise ValueError("n_rings must be >= 1.")
            n_ang = np.full(int(n_rings), int(n_angular), dtype=int)
        else:
            n_ang = np.asarray(n_angular, dtype=int).reshape(-1)
            if n_ang.size < 1:
                raise ValueError("n_angular must have at least one entry.")
            n_rings = int(n_ang.size)

        total = int(np.sum(n_ang))
        frac = np.concatenate([[0.0], np.cumsum(n_ang, dtype=float) / float(total)])
        inner_scale = inner_radius / radius
        scales2 = inner_scale * inner_scale + (1.0 - inner_scale * inner_scale) * frac
        radial_scales = np.sqrt(scales2)
        cx, cy = center
        bounds = ((cx - radius, cx + radius), (cy - radius, cy + radius))
        return cls(
            domain_bounds=bounds,
            radial_scales=radial_scales,
            n_angular=n_ang,
            center=center,
            theta_offset=theta_offset,
        )
