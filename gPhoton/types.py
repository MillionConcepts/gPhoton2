from pathlib import Path
from typing import TypeVar, Literal

Pathlike = TypeVar("Pathlike", str, Path)
GalexBand = Literal["NUV", "FUV"]
