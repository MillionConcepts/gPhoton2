from pathlib import Path
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

Pathlike = str | Path
GalexBand = Literal["NUV", "FUV"]


# Type variables for code that is generic over numpy dtype as long as
# all the arguments (or groups of the arguments) have consistent dtype.
# For typical usage patterns see gPhoton/vorpal.py and
# gPhoton/coords/gnomonic.py.
NFloat = TypeVar("NFloat", bound=np.floating[Any])
NFloatX = TypeVar("NFloatX", bound=np.floating[Any])
NFloatY = TypeVar("NFloatY", bound=np.floating[Any])

NInt = TypeVar("NInt", bound=np.integer[Any])
NIntX = TypeVar("NIntX", bound=np.integer[Any])
NIntY = TypeVar("NIntY", bound=np.integer[Any])

NNum = TypeVar("NNum", bound=np.number[Any])
NNumX = TypeVar("NNumX", bound=np.number[Any])
NNumY = TypeVar("NNumY", bound=np.number[Any])


__all__ = [
    "Pathlike",
    "GalexBand",
    "NDArray",
    "NFloat", "NFloatX", "NFloatY",
    "NInt", "NIntX", "NIntY",
    "NNum", "NNumX", "NNumY",
]
