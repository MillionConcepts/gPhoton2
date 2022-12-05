"""
utilities for generating shared reference points between pipeline components:
canonical file paths, timer objects, etc.
this module is supposed to be essentially free to import: only standard
library modules should be used, at least as module-level imports.

# TODO: the leg count lookup I've added militates against the 'free to import'
# intent, but there's a ton of complexity it potentially reduces
"""
import functools
import subprocess
import time
from inspect import getmodule
from typing import Optional, Literal, Any, Callable

from gPhoton.types import Pathlike


def eclipse_to_paths(
    eclipse: int,
    data_directory: Pathlike = "data",
    depth: Optional[int] = None,
    compression: Literal["none", "gzip", "rice"] = "gzip",
) -> dict[str, dict[str, str]]:
    """
    generate canonical paths for files associated with a given eclipse,
    optionally including files at a specific depth
    """
    data_directory = "data" if data_directory is None else data_directory
    from gPhoton.aspect import aspect_tables
    zpad = str(eclipse).zfill(5)
    eclipse_path = f"{data_directory}/e{zpad}/"
    eclipse_base = f"{eclipse_path}e{zpad}"
    bands = "NUV", "FUV"
    band_initials = "n", "f"
    file_dict = {}
    comp = {
        "gzip": ".fits.gz", "none": ".fits", "rice": "-rice.fits"
    }[compression]
    leg_number = aspect_tables(eclipse, ("metadata",))[0]['legs'][0].as_py()
    legs = (0,) if leg_number == 0 else tuple(range(leg_number))
    for band, initial in zip(bands, band_initials):
        prefix = f"{eclipse_base}-{initial}d"
        band_dict = {
            "raw6": f"{prefix}-raw6.fits.gz",
            "photonfiles": [f"{prefix}.parquet" for _ in legs],
            "images": [f"{prefix}-full{comp}" for _ in legs],
            "extended_catalogs": [
                f"{prefix}-extended-sources.csv" for leg in legs
            ]
        }
        if depth is not None:
            band_dict |= {
                "movies": [f"{prefix}-{depth}s{comp}" for leg in legs],
                # stem -- multiple aperture sizes possible
                "photomfiles": [
                    f"{prefix}-{depth}s-photom-" for leg in legs
                ],
                "expfiles": [
                    f"{prefix}-{depth}s-exptime.csv" for leg in legs
                ]
            }
        else:
            band_dict |= {
                "photomfiles": [f"{prefix}-full-photom-" for leg in legs]
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

    def __init__(self, digits=2, silent=False):
        self.digits = digits
        self.last_time = None
        self.start_time = None
        self.silent = silent

    def peek(self):
        if self.last_time is None:
            return 0
        return round(time.time() - self.last_time, self.digits)

    def start(self):
        if self.silent is False:
            print("starting timer")
        now = time.time()
        self.start_time = now
        self.last_time = now

    def click(self):
        if self.last_time is None:
            return self.start()
        if self.silent is False:
            print(f"{self.peek()} elapsed seconds, restarting timer")
        self.last_time = time.time()

    def total(self):
        if self.last_time is None:
            return 0
        return round(time.time() - self.start_time, self.digits)


PROC_NET_DEV_FIELDS = (
    "bytes",
    "packets",
    "errs",
    "drop",
    "fifo",
    "frame",
    "compressed",
    "multicast",
)


def catprocnetdev():
    return subprocess.run(
        ("cat", "/proc/net/dev"), stdout=subprocess.PIPE
    ).stdout.decode()


def parseprocnetdev(procnetdev, rejects=("lo",)):
    interface_lines = filter(
        lambda l: ":" in l[:12], map(str.strip, procnetdev.split("\n"))
    )
    entries = []
    for interface, values in map(lambda l: l.split(":"), interface_lines):
        if interface in rejects:
            continue
        records = {
            field: int(number)
            for field, number in zip(
                PROC_NET_DEV_FIELDS, filter(None, values.split(" "))
            )
        }
        entries.append({"interface": interface} | records)
    return entries


class Netstat:
    # TODO: monitor TX as well as RX, etc.
    def __init__(self, rejects=("lo",)):
        self.rejects = rejects
        self.absolute, self.last, self.interval, self.total = None, {}, {}, {}
        self.update()

    def update(self):
        self.absolute = parseprocnetdev(catprocnetdev(), self.rejects)
        for line in self.absolute:
            interface, bytes_ = line["interface"], line["bytes"]
            if interface not in self.interval.keys():
                self.total[interface] = 0
                self.interval[interface] = 0
                self.last[interface] = bytes_
            else:
                self.interval[interface] = bytes_ - self.last[interface]
                self.total[interface] += self.interval[interface]
                self.last[interface] = bytes_

    def __repr__(self):
        return str(self.absolute)


def crudely_find_library(obj: Any) -> str:
    if isinstance(obj, functools.partial):
        if len(obj.args) > 0:
            if isinstance(obj.args[0], Callable):
                return crudely_find_library(obj.args[0])
        return crudely_find_library(obj.func)
    return getmodule(obj).__name__.split(".")[0]
