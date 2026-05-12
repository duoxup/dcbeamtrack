from __future__ import annotations

import locale
from typing import Optional, Tuple

import numpy as np

from .cell import CellGrid


class _NumericCLocale:
    """Context manager forcing LC_NUMERIC=C; needed by meshpy on some locales."""

    def __enter__(self):
        try:
            self.prev = locale.setlocale(locale.LC_NUMERIC)
        except locale.Error:
            self.prev = None
        try:
            locale.setlocale(locale.LC_NUMERIC, "C")
        except locale.Error:
            pass
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.prev is not None:
            try:
                locale.setlocale(locale.LC_NUMERIC, self.prev)
            except locale.Error:
                pass


class TriGrid(CellGrid):
    """Unstructured triangular cell grid over a rectangular domain.

    Backed by :mod:`meshpy.triangle` (Delaunay + Steiner refinement) when
    available, falling back to :class:`scipy.spatial.Delaunay` over a regular
    lattice. ``hmax`` (maximum triangle edge length) or ``target_num_cells``
    determines the resolution; structured 2-triangles-per-cell layouts are
    intentionally not produced here because they bias the sampling along
    grid axes.
    """

    def __init__(
        self,
        *,
        domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]],
        hmax: Optional[float] = None,
        target_num_cells: Optional[int] = None,
        min_angle: float = 30.0,
        allow_boundary_steiner: bool = True,
        allow_volume_steiner: bool = True,
        fallback_lattice_n: int = 30,
    ) -> None:
        (xmin, xmax), (ymin, ymax) = domain_bounds
        if not (xmax > xmin and ymax > ymin):
            raise ValueError("Invalid domain_bounds.")
        if hmax is None and target_num_cells is None:
            raise ValueError("Provide hmax or target_num_cells.")
        if hmax is not None and hmax <= 0.0:
            raise ValueError("hmax must be positive.")
        if target_num_cells is not None and int(target_num_cells) < 1:
            raise ValueError("target_num_cells must be >= 1.")

        polygons = self._build_triangles(
            xmin, xmax, ymin, ymax,
            hmax=hmax,
            target_num_cells=target_num_cells,
            min_angle=min_angle,
            allow_boundary_steiner=allow_boundary_steiner,
            allow_volume_steiner=allow_volume_steiner,
            fallback_lattice_n=fallback_lattice_n,
        )

        base = CellGrid.from_polygons(grid_type="tri", polygons=polygons)
        super().__init__(
            grid_type=base.grid_type,
            xc=base.xc,
            yc=base.yc,
            areas=base.areas,
            polygons=base.polygons,
        )
        object.__setattr__(self, "domain_bounds", ((xmin, xmax), (ymin, ymax)))

    @staticmethod
    def _max_volume(hmax, target_num_cells, xmin, xmax, ymin, ymax):
        if hmax is not None:
            return 0.5 * hmax * hmax
        return (xmax - xmin) * (ymax - ymin) / float(target_num_cells)

    @classmethod
    def _build_triangles(
        cls,
        xmin, xmax, ymin, ymax,
        *, hmax, target_num_cells, min_angle,
        allow_boundary_steiner, allow_volume_steiner, fallback_lattice_n,
    ) -> list:
        max_volume = cls._max_volume(hmax, target_num_cells, xmin, xmax, ymin, ymax)

        try:
            import meshpy.triangle as triangle  # type: ignore
        except ImportError:
            triangle = None

        if triangle is not None:
            with _NumericCLocale():
                info = triangle.MeshInfo()
                info.set_points([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
                info.set_facets([(0, 1), (1, 2), (2, 3), (3, 0)])
                mesh = triangle.build(
                    info,
                    min_angle=float(min_angle),
                    allow_boundary_steiner=bool(allow_boundary_steiner),
                    allow_volume_steiner=bool(allow_volume_steiner),
                    max_volume=max_volume,
                )
            verts = np.array(mesh.points, dtype=float)
            elems = np.array(mesh.elements, dtype=int)
        else:
            from scipy.spatial import Delaunay

            n = max(3, int(fallback_lattice_n))
            # Choose lattice density so total cells ~ 2 * (n-1)^2 ≈ target.
            if max_volume is not None:
                total_area = (xmax - xmin) * (ymax - ymin)
                target = max(4, int(np.ceil(total_area / max_volume)))
                n = max(3, int(np.ceil(np.sqrt(target / 2.0))) + 1)
            gx = np.linspace(xmin, xmax, n)
            gy = np.linspace(ymin, ymax, n)
            X, Y = np.meshgrid(gx, gy, indexing="xy")
            verts = np.column_stack([X.ravel(), Y.ravel()])
            elems = Delaunay(verts).simplices

        eps_area = 1e-14 * (xmax - xmin) * (ymax - ymin)
        polygons = []
        for (i0, i1, i2) in elems:
            p0, p1, p2 = verts[i0], verts[i1], verts[i2]
            signed = 0.5 * (
                p0[0] * (p1[1] - p2[1])
                + p1[0] * (p2[1] - p0[1])
                + p2[0] * (p0[1] - p1[1])
            )
            if abs(signed) <= eps_area:
                continue
            polygons.append(np.array([p0, p1, p2], dtype=float))
        if not polygons:
            raise RuntimeError("Triangulation produced no non-degenerate cells.")
        return polygons

    @classmethod
    def from_size(
        cls,
        *,
        width: float,
        height: float,
        hmax: Optional[float] = None,
        target_num_cells: Optional[int] = None,
        center: Tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ) -> "TriGrid":
        if width <= 0.0 or height <= 0.0:
            raise ValueError("width and height must be positive.")
        cx, cy = center
        bounds = (
            (cx - 0.5 * width, cx + 0.5 * width),
            (cy - 0.5 * height, cy + 0.5 * height),
        )
        return cls(
            domain_bounds=bounds,
            hmax=hmax,
            target_num_cells=target_num_cells,
            **kwargs,
        )

    @classmethod
    def from_bounds(
        cls,
        *,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        hmax: Optional[float] = None,
        target_num_cells: Optional[int] = None,
        **kwargs,
    ) -> "TriGrid":
        return cls(
            domain_bounds=((xmin, xmax), (ymin, ymax)),
            hmax=hmax,
            target_num_cells=target_num_cells,
            **kwargs,
        )
