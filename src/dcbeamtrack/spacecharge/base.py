from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class FieldMap2DProtocol(Protocol):
    """Interpolatable transverse electric field map.

    Returned by :meth:`SpaceChargeSolver.solve` and consumed by the tracker
    each step at the current particle positions.

    See ``dcbeamtrack.spacecharge`` for built-in implementations and
    ``examples/04_custom_solver.py`` for a custom walkthrough.
    """

    def field_at(self, x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Ex, Ey)`` in V/m at the requested transverse coordinates.

        Parameters
        ----------
        x, y : array-like
            Identical shape. The tracker passes 1-D arrays of length
            ``n_particles``; implementations should accept any shape that
            NumPy broadcasts naturally.

        Returns
        -------
        tuple of numpy.ndarray
            ``(Ex, Ey)`` with the same shape as the inputs, units V/m.

        Notes
        -----
        Off-source behavior is implementation-defined. The built-in
        :class:`CircularGreenFieldMap` raises :class:`ValueError` for
        off-source queries (it only caches values at source-particle
        positions); :class:`FieldMap2D` (rect FDM) interpolates bilinearly
        across the grid. Document the chosen behavior in the implementing
        class. Implementations must not mutate the input arrays.
        """


class SpaceChargeSolver(Protocol):
    """Space-charge solver consumed by the tracker (``track_slice`` directly
    or via ``DcBeamSimulation.run``).

    Each tracker step calls :meth:`solve` with the current slice
    distribution and uses the returned field map to push particles.

    See ``dcbeamtrack.spacecharge`` for built-in implementations and
    ``examples/04_custom_solver.py`` for a custom walkthrough.
    """

    def solve(self, distribution: object) -> FieldMap2DProtocol:
        """Solve the transverse space-charge field for a fixed-z slice.

        Parameters
        ----------
        distribution : SliceDistribution
            Duck-typed: implementations may rely on ``.x``, ``.y``,
            ``.lam``, ``.z``, and ``.get_data(key)`` (used to read extras
            such as ``wall_hit``). The tracker passes a fresh
            ``SliceDistribution`` per step.

        Returns
        -------
        FieldMap2DProtocol
            Any object with ``field_at(x, y) -> (Ex, Ey)``.

        Notes
        -----
        Wall-hit handling is not required but recommended for solvers whose
        field depends on per-particle position. The built-in
        :class:`CircularGreenSpaceChargeSolver` skips particles flagged
        via ``distribution.get_data("wall_hit") > 0.5`` from the source
        sum so they neither contribute to nor receive force. Rect FDM
        solvers deposit the (snapped) wall-hit positions like any other
        particle — acceptable because the boundary class also zeros their
        velocities. Implementations must not mutate the input distribution.
        """
