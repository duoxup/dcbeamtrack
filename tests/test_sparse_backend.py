import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

from dcbeamtrack.spacecharge.poisson import make_sparse_solve


_HAS_PYPARDISO = importlib.util.find_spec("pypardiso") is not None
_skip_if_no_pypardiso = pytest.mark.skipif(
    not _HAS_PYPARDISO, reason="pypardiso not installed"
)


def test_make_sparse_solve_rejects_invalid_backend():
    matrix = sp.eye(4, format="csc")
    with pytest.raises(ValueError):
        make_sparse_solve(matrix, backend="bogus")
