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
from math import floor
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Sequence, Mapping, Optional, Literal

from gPhoton.types import Pathlike, GalexBand


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


@cache
def get_legs(eclipse):
    from gPhoton.aspect import aspect_tables

    return tuple(range(len(aspect_tables(eclipse, ("boresight",))[0]["time"])))


@cache
def titular_legs(eclipse):
    from gPhoton.aspect import aspect_tables

    actual = len(get_legs(eclipse))
    nominal = aspect_tables(eclipse, ("metadata",))[0]["legs"][0].as_py()
    if (actual == 1) and (nominal == 0):
        return 0, 0
    return actual, nominal


def intfill(obj, zfill=4):
    obj = float(obj)
    integer = str(floor(obj)).zfill(zfill)
    if (decimal := obj - floor(obj)) > 0:
        # round for floating-point error
        return integer + str(round(decimal, 4))[2:]
    return integer


def eclipse_to_paths(
    eclipse: int,
    band: GalexBand = "NUV",
    depth: Optional[int] = None,
    compression: Literal["none", "gzip", "rice"] = "gzip",
    root: Pathlike = "data",
    start: Optional[float] = None,
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

        return emojified(compression, depth, leg, band, eclipse_base, start)
    ext = {"gzip": ".fits.gz", "none": ".fits", "rice": ".fits"}[compression]
    comp = {"gzip": "g", "none": "u", "rice": "r"}[compression]
    mode = {"direct": "d", "grism": "g", "opaque": "o"}[mode]
    start = "movie" if start is None else f"t{intfill(start)}"
    depth = None if depth is None else f"f{intfill(depth)}"
    prefix = f"{eclipse_base}-{band[0].lower()}{mode}"
    aper = "" if aperture is None else str(aperture).replace(".", "_")
    file_dict = {
        "raw6": f"{prefix}-raw6.fits.gz",
        "photonfile": f"{prefix}-b{leg}.parquet",
        "image": f"{prefix}-b{leg}-ffull-image-{comp}{ext}",
        # TODO: frames, etc. -- decide exactly how once we are using
        #  extended source detection on movies
        "extended_catalog": f"{prefix}-b{leg}-extended-sources.csv",
    }
    if depth is not None:
        file_dict |= {
            "movie": f"{prefix}-b{leg}-{depth}-{start}-{comp}{ext}",
            "photomfile": f"{prefix}-{depth}-b{leg}-{start}-photom-{aper}.csv",
            "expfile": f"{prefix}-{depth}-b{leg}-{start}-exptime.csv",
        }
    else:
        file_dict[
            "photomfile"
        ] = f"{prefix}-b{leg}-ffull-image-photom-{aper}.csv"
    return file_dict


class PipeContext:
    """
    simple class for tracking identifying options / elements of a pipeline
    execution and constructing paths based on those options.
    """
    def __init__(
        self,
        eclipse: int,
        band: GalexBand = "NUV",
        depth: Optional[int] = None,
        compression: Literal["none", "gzip", "rice"] = "gzip",
        local: Pathlike = "data",
        remote: Optional[Pathlike] = None,
        aperture_sizes: Sequence[float] = (12.8,),
        leg: int = 0,
        frame: Optional[int] = None,
        mode: str = "direct",
        download: bool = True,
        recreate: bool = False,
        verbose: int = 2,
        source_catalog_file: Optional[str] = None,
        write: Mapping = MappingProxyType({"image": True, "movie": True}),
        lil: bool = True,
        coregister_lightcurves: bool = False,
        stop_after: Optional[Literal["photonpipe", "moviemaker"]] = None,
        min_exptime: Optional[float] = None,
        photometry_only: bool = False,
        burst: bool = False,
        hdu_constructor_kwargs: Mapping = MappingProxyType({}),
        threads: Optional[int] = None,
        watch: Optional[Stopwatch] = None,
        share_memory: Optional[bool] = None,
        chunksz: Optional[int] = 1000000,
        extended_photonlist: bool = False,
        aspect: str = "aspect",
        start_time: Optional[float] = None,
        snippet: Optional[tuple] = None
    ):
        self.eclipse = eclipse
        self.band = band
        self.depth = depth
        self.compression = compression
        self.local = local
        self.remote = remote
        self.frame = frame
        self.mode = mode
        self.leg = leg
        self.aperture_sizes = aperture_sizes
        self.download = download
        self.recreate = recreate
        self.verbose = verbose
        self.source_catalog_file = source_catalog_file
        self.write = write
        self.lil = lil
        self.coregister_lightcurves = coregister_lightcurves
        self.stop_after = stop_after
        self.min_exptime = min_exptime
        self.photometry_only = photometry_only
        self.burst = burst
        self.hdu_constructor_kwargs = hdu_constructor_kwargs
        self.threads = threads
        self.watch = watch if watch is not None else Stopwatch()
        self.extended_photonlist = extended_photonlist
        self.chunksz = chunksz
        self.share_memory = share_memory
        self.aspect = aspect
        self.start_time = start_time
        self.snippet = snippet


    def __repr__(self):
        return (
            f"PipeContext(eclipse={self.eclipse}, band={self.band}, "
            f"depth={self.depth}, compression={self.compression}, "
            f"frame={self.frame}, mode={self.mode}, leg={self.leg}, "
            f"apertures={self.aperture_sizes}, local={self.local}, "
            f"remote={self.remote}"
        )

    def __str__(self):
        return repr(self)

    def pathdict(self) -> dict[str, Any]:
        return {
            "eclipse": self.eclipse,
            "band": self.band,
            "depth": self.depth,
            "compression": self.compression,
            "local": self.local,
            "remote": self.remote,
            "frame": self.frame,
            "mode": self.mode,
            "leg": self.leg,
            "aperture_sizes": self.aperture_sizes,
        }

    def asdict(self) -> dict[str, Any]:
        return {
            "eclipse": self.eclipse,
            "band": self.band,
            "depth": self.depth,
            "compression": self.compression,
            "local": self.local,
            "remote": self.remote,
            "frame": self.frame,
            "mode": self.mode,
            "leg": self.leg,
            "aperture_sizes": self.aperture_sizes,
            "download": self.download,
            "recreate": self.recreate,
            "verbose": self.verbose,
            "source_catalog_file": self.source_catalog_file,
            "write": self.write,
            "lil": self.lil,
            "coregister_lightcurves": self.coregister_lightcurves,
            "stop_after": self.stop_after,
            "min_exptime": self.min_exptime,
            "photometry_only": self.photometry_only,
            "burst": self.burst,
            "hdu_constructor_kwargs": self.hdu_constructor_kwargs,
            "threads": self.threads,
            "watch": self.watch,
            "chunksz": self.chunksz,
            "share_memory": self.share_memory,
            "extended_photonlist": self.extended_photonlist,
            "start_time": self.start_time
        }

    def eclipse_path(self, remote=False):
        root = self.local if remote is False else self.remote
        return Path(root, f"e{str(self.eclipse).zfill(5)}")

    def temp_path(self, remote=False):
        root = self.local if remote is False else self.remote
        return Path(root, "temp", f"e{str(self.eclipse).zfill(5)}")

    def __call__(self, remote=False, **kwargs):
        kwargs = self.pathdict() | kwargs
        apertures = kwargs.pop("aperture_sizes")
        if "aperture" not in kwargs.keys():
            kwargs["aperture"] = apertures[0]
        if "root" not in kwargs.keys():
            kwargs["root"] = self.remote if remote is True else self.local
        kwargs.pop("local"), kwargs.pop("remote")
        return eclipse_to_paths(**kwargs)

    def __getitem__(self, key):
        return self()[key]

    def explode_legs(self):
        """
        generate a list of PipeContexts, one for each leg of the eclipse
        """
        legs = get_legs(self.eclipse)
        kdict = self.asdict()
        del kdict['leg']
        return [PipeContext(leg=leg, **kdict) for leg in legs]


def check_eclipse(eclipse):
    from gPhoton.aspect import aspect_tables
    e_warn, e_error = [], []
    if eclipse > 47000:
        e_error.append("CAUSE data w/eclipse>47000 are not yet supported.")
    meta = aspect_tables(eclipse, ("metadata",))[0]
    if len(meta) == 0:
        e_error.append(f"No metadata found for e{eclipse}.")
        return e_warn, e_error
    obstype = meta['obstype'].to_pylist()[0]
    actual, nominal = titular_legs(eclipse)
    if obstype == 'CAI':
        e_error.append('CAI mode is not yet supported.')
    elif (obstype in ("MIS", "GII")) and (actual == 1) and (nominal > 0):
        e_error.append('petal pattern is not yet supported.')
    elif actual != nominal:
        e_warn.append(
            f"Note: e{eclipse} observation-level metadata specifies "
            f"{nominal} legs, but only {actual} appear(s) to have "
            f"been completed."
        )
    return e_warn, e_error
