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
    CircularGreenFieldMap,
    CircularGreenSpaceChargeSolver,
    FieldMap2D,
    FieldMap2DProtocol,
    RectFdmSpaceChargeSolver,
    SpaceChargeSolver,
)
from .sources import CallableDensity, GaussianMixture, UniformDisk, sample_slice_from_density_on_grid
from .tracking import SliceTrackResult, track_slice

__all__ = [
    "__version__",
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
