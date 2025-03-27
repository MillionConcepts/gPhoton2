"""
numba-compiled fast ndarray-partitioning methods
"""

import numpy as np

from gPhoton.numba_utilz import jit
from gPhoton.types import NDArray, NNumX, NNumY


@jit
def between(
    reference: NDArray[NNumX],
    t0: NNumX,
    t1: NNumX,
) -> tuple[NDArray[np.intp], ...]:
    return np.nonzero((reference >= t0) & (reference < t1))


@jit
def slice_between(
    subject: NDArray[NNumY],
    reference: NDArray[NNumX],
    t0: NNumX,
    t1: NNumX,
) -> NDArray[NNumY]:
    return subject[between(reference, t0, t1)]
