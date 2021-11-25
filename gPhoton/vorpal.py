"""
numba-compiled fast ndarray-partitioning methods
"""

from numba import njit
import numpy as np


@njit(cache=True)
def between(reference, t0, t1):
    return np.where((reference >= t0) & (reference < t1))


@njit(cache=True)
def slice_between(subject, reference, t0, t1):
    return subject[between(reference, t0, t1)]
