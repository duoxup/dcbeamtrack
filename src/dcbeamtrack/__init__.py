from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dcbeamtrack")
except PackageNotFoundError:
    __version__ = "0.0.0.dev"

from .config import TrackConfig
from .boundary import CircularBoundary, RectBoundary, TransverseBoundary, UnboundedBoundary
from .fields import AxialMagneticField, MagneticField, UniformMagneticField
from .mesh import AnnulusGrid, RectGrid, TriGrid
from .simulation import DcBeamSimulation
from .spacecharge import (
    AnalyticMirrorChargeFieldMap,
    AnalyticMirrorChargeSpaceChargeSolver,
    CircularGreenFieldMap,
    CircularGreenSpaceChargeSolver,
    FieldMap2D,
    FieldMap2DProtocol,
    RectFdmCircularBoundarySolver,
    RectFdmSpaceChargeSolver,
    SpaceChargeSolver,
)
from .sources import CallableDensity, GaussianMixture, UniformDisk, sample_slice_from_density_on_grid
from .tracking import ParticleTrajectories, SliceTrackResult, track_slice

__all__ = [
    "__version__",
    "AnalyticMirrorChargeFieldMap",
    "AnalyticMirrorChargeSpaceChargeSolver",
    "AxialMagneticField",
    "CallableDensity",
    "CircularBoundary",
    "CircularGreenFieldMap",
    "CircularGreenSpaceChargeSolver",
    "DcBeamSimulation",
    "FieldMap2D",
    "FieldMap2DProtocol",
    "GaussianMixture",
    "MagneticField",
    "ParticleTrajectories",
    "RectFdmCircularBoundarySolver",
    "RectFdmSpaceChargeSolver",
    "AnnulusGrid",
    "RectBoundary",
    "RectGrid",
    "SliceTrackResult",
    "SpaceChargeSolver",
    "TriGrid",
    "TrackConfig",
    "TransverseBoundary",
    "UniformDisk",
    "UniformMagneticField",
    "UnboundedBoundary",
    "sample_slice_from_density_on_grid",
    "track_slice",
]
