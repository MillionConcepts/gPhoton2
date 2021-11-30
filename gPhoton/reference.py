"""
utilities for generating shared reference points between pipeline components:
canonical file paths, timer objects, etc.
this module is supposed to be essentially free to import: only standard
library modules should be used, at least as module-level imports.
"""

from time import time
from typing import Optional

from gPhoton.types import Pathlike


def eclipse_to_paths(
    eclipse: int,
    data_directory: Pathlike = "data",
    depth: Optional[int] = None
) -> dict[str, dict[str, str]]:
    """
    generate canonical paths for files associated with a given eclipse,
    optionally including files at a specific depth
    """
    zpad = str(eclipse).zfill(5)
    eclipse_path = f"{data_directory}/e{zpad}/"
    eclipse_base = f"{eclipse_path}e{zpad}"
    bands = "NUV", "FUV"
    band_initials = "n", "f"
    file_dict = {}
    for band, initial in zip(bands, band_initials):
        prefix = f"{eclipse_base}-{initial}d"
        band_dict = {
            "raw6": f"{prefix}-raw6.fits.gz",
            "photonfile": f"{prefix}.parquet",
            "image": f"{prefix}-full.fits.gz",
        }
        if depth is not None:
            band_dict |= {
                "movie": f"{prefix}-{depth}s.fits.gz",
                # stem -- multiple aperture sizes possible
                "photomfile": f"{prefix}-{depth}s-photom-",
                "expfile": f"{prefix}-{depth}s-exptime.csv",
            }
        file_dict[band] = band_dict
    return file_dict


class FakeStopwatch:
    """fake simple timer object"""

    def click(self):
        return


class Stopwatch(FakeStopwatch):
    """
    simple timer object
    """

    def __init__(self, digits=2):
        self.digits = digits
        self.last_time = None
        self.start_time = None

    def peek(self):
        return round(time() - self.last_time, self.digits)

    def start(self):
        print("starting timer")
        now = time()
        self.start_time = now
        self.last_time = now

    def click(self):
        if self.last_time is None:
            return self.start()
        print(f"{self.peek()} elapsed seconds, restarting timer")
        self.last_time = time()
