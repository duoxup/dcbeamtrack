from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from partdist import SliceDistribution

from dcbeamtrack.boundary import CircularBoundary, RectBoundary, TransverseBoundary
from dcbeamtrack.config import TrackConfig
from dcbeamtrack.fields import MagneticField
from dcbeamtrack.mesh import RectGrid
from dcbeamtrack.spacecharge import (
    AnalyticMirrorChargeSpaceChargeSolver,
    CircularGreenSpaceChargeSolver,
    RectFdmCircularBoundarySolver,
    RectFdmSpaceChargeSolver,
    SpaceChargeSolver,
)
from dcbeamtrack.tracking import SliceTrackResult, track_slice
from dcbeamtrack.tracking.result import _config_summary, _safe_summary


@dataclass(frozen=True)
class DcBeamSimulation:
    """Configured DC-beam slice simulation.

    The simulation object wires a space-charge solver, a magnetic field, and an
    optional transverse boundary into the fixed-z slice tracker.
    """

    magnetic_field: MagneticField
    config: TrackConfig
    spacecharge_solver: SpaceChargeSolver
    boundary: TransverseBoundary | None = None

    @classmethod
    def from_solver(
        cls,
        *,
        spacecharge_solver: SpaceChargeSolver,
        magnetic_field: MagneticField,
        config: TrackConfig,
        boundary: TransverseBoundary | None = None,
    ) -> "DcBeamSimulation":
        return cls(
            magnetic_field=magnetic_field,
            config=config,
            spacecharge_solver=spacecharge_solver,
            boundary=boundary,
        )

    @classmethod
    def circular_green(
        cls,
        *,
        radius: float,
        magnetic_field: MagneticField,
        config: TrackConfig,
        n_r: int = 96,
        n_theta: int = 720,
        r_eps: float = 1e-9,
    ) -> "DcBeamSimulation":
        solver = CircularGreenSpaceChargeSolver(
            radius=radius,
            n_r=n_r,
            n_theta=n_theta,
            r_eps=r_eps,
        )
        return cls(
            magnetic_field=magnetic_field,
            config=config,
            spacecharge_solver=solver,
            boundary=CircularBoundary(radius=radius),
        )

    @classmethod
    def analytic_mirror_charge(
        cls,
        *,
        radius: float,
        magnetic_field: MagneticField,
        config: TrackConfig,
    ) -> "DcBeamSimulation":
        solver = AnalyticMirrorChargeSpaceChargeSolver(radius=radius)
        return cls(
            magnetic_field=magnetic_field,
            config=config,
            spacecharge_solver=solver,
            boundary=CircularBoundary(radius=radius),
        )

    @classmethod
    def rect_fdm_circular(
        cls,
        *,
        grid: RectGrid,
        radius: float,
        magnetic_field: MagneticField,
        config: TrackConfig,
        deposition: str | None = None,
    ) -> "DcBeamSimulation":
        solver = RectFdmCircularBoundarySolver(
            grid,
            radius=radius,
            deposition=deposition or config.deposition,
        )
        return cls(
            magnetic_field=magnetic_field,
            config=config,
            spacecharge_solver=solver,
            boundary=CircularBoundary(radius=radius),
        )

    @classmethod
    def rect_fdm(
        cls,
        *,
        grid: RectGrid,
        magnetic_field: MagneticField,
        config: TrackConfig,
        spacecharge_solver: RectFdmSpaceChargeSolver | None = None,
    ) -> "DcBeamSimulation":
        solver = spacecharge_solver or RectFdmSpaceChargeSolver(
            grid,
            deposition=config.deposition,
        )
        return cls(
            magnetic_field=magnetic_field,
            config=config,
            spacecharge_solver=solver,
            boundary=RectBoundary.from_grid(grid),
        )

    def run(self, distribution: SliceDistribution) -> SliceTrackResult:
        result = track_slice(
            distribution,
            magnetic_field=self.magnetic_field,
            config=self.config,
            spacecharge_solver=self.spacecharge_solver,
            boundary=self.boundary,
        )
        n_live_final = int(result.live_mask()[-1].sum()) if len(result) else 0
        result_metadata = {
            "solver": _safe_summary(self.spacecharge_solver),
            "magnetic_field": _safe_summary(self.magnetic_field),
            "boundary": _safe_summary(self.boundary),
            "config": _config_summary(self.config),
            "particle_counts": {
                "n_initial": int(result.n_particles),
                "n_live_final": n_live_final,
                "n_wall_hit_final": int(result.n_particles - n_live_final),
            },
        }
        return dataclasses.replace(result, metadata=result_metadata)
