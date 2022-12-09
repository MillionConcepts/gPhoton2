"""
utilities for generating shared reference points between pipeline components:
canonical file paths, timer objects, etc.
this module is supposed to be essentially free to import: only standard
library modules should be used, at least as module-level imports.
the leg-count lookup militates against this idea some, but it reduces a great
deal of complexity.

# TODO: the leg count lookup I've added militates against the 'free to import'
# intent, but there's a ton of complexity it potentially reduces
"""
import functools
import subprocess
import time
from functools import cache
from inspect import getmodule
from typing import Any, Callable
from typing import Optional, Literal

from gPhoton.types import Pathlike, GalexBand


@cache
def get_legs(eclipse):
    from gPhoton.aspect import aspect_tables

    leg_count = aspect_tables(eclipse, ("metadata",))[0]["legs"][0].as_py()
    return (0,) if leg_count == 0 else tuple(range(leg_count))


def eclipse_to_paths(
    eclipse: int,
    band: GalexBand = "NUV",
    depth: Optional[int] = None,
    compression: Literal["none", "gzip", "rice"] = "gzip",
    root: Pathlike = "data",
    frame: Optional[int] = None,
    mode: str = "direct",
    leg: int = 0,
    aperture: Optional[float] = None,
    **kwargs,
) -> dict[str, str]:
    """
    generate canonical paths for files associated with a given eclipse,
    optionally including files at a specific depth
    """
    root = "data" if root is None else root
    zpad, leg = str(eclipse).zfill(5), str(leg).zfill(2)
    eclipse_path = f"{root}/e{zpad}/"
    eclipse_base = f"{eclipse_path}e{zpad}"
    if kwargs.get("emoji") is True:
        from gPhoton.__emoji import emojified

        return emojified(compression, depth, leg, eclipse_base, frame)
    ext = {"gzip": ".fits.gz", "none": ".fits", "rice": ".fits"}[compression]
    comp = {"gzip": "g", "none": "u", "rice": "r"}[compression]
    mode = {"direct": "d", "grism": "g", "opaque": "o"}[mode]
    frame = "movie" if frame is None else f"f{str(frame).zfill(4)}"
    depth = None if depth is None else f"t{str(depth).zfill(4)}"
    prefix = f"{eclipse_base}-{band[0].lower()}{mode}"
    aper = "" if aperture is None else str(aperture).replace(".", "_")
    file_dict = {
        "raw6": f"{prefix}-raw6.fits.gz",
        "photonfile": f"{prefix}-b{leg}.parquet",
        "image": f"{prefix}-tfull-b{leg}-image-{comp}{ext}",
        # TODO: frames, etc. -- decide exactly how once we are using
        #  extended source detection on movies
        "extended_catalog": f"{prefix}-b{leg}-extended-sources.csv",
    }
    if depth is not None:
        file_dict |= {
            "movie": f"{prefix}-{depth}-{leg}-{frame}-{comp}{ext}",
            "photomfile": f"{prefix}-{depth}-b{leg}-{frame}-photom-{aper}.csv",
            "expfile": f"{prefix}-{depth}-b{leg}-{frame}-exptime.csv",
        }
    else:
        file_dict[
            "photomfile"
        ] = f"{prefix}-tfull-b{leg}-image-photom-{aper}.csv"
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
