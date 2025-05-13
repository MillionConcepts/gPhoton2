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
from pathlib import Path
from types import MappingProxyType
from typing import Any, Sequence, Mapping, Optional, Literal

from gPhoton.eclipse import eclipse_to_paths
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
        if obj.args and callable(obj.args[0]):
            return crudely_find_library(obj.args[0])
        return crudely_find_library(obj.func)
    mod = getmodule(obj)
    if mod is None:
        raise ImportError("could not find a library for " + repr(obj))
    return mod.__name__.split(".")[0]


@cache
def get_legs(eclipse, aspect_dir: None | str | Path = None):
    from gPhoton.aspect import aspect_tables

    return tuple(range(len(aspect_tables(
        eclipse=eclipse,
        tables="boresight",
        columns=["time"],
        aspect_dir=aspect_dir,
    )[0]["time"])))


@cache
def titular_legs(eclipse, aspect_dir: None | str | Path = None):
    from gPhoton.aspect import aspect_tables

    actual = len(get_legs(eclipse, aspect_dir=aspect_dir))
    nominal = aspect_tables(
        eclipse=eclipse,
        tables="metadata",
        columns=["legs"],
        aspect_dir=aspect_dir
    )[0]["legs"][0].as_py()
    if (actual == 1) and (nominal == 0):
        return 0, 0
    return actual, nominal


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
        extended_flagging: bool = False,
        aspect: Literal["aspect", "aspect2"] = "aspect",
        start_time: Optional[float] = None,
        snippet: Optional[tuple] = None,
        suffix: Optional[str] = None,
        aspect_dir: None | str | Path = None,
        ftype: str = "csv",
        wide_edge_thresh: int = 340,
        narrow_edge_thresh: int = 360,
        single_leg: Optional[int] = None
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
        self.extended_flagging = extended_flagging
        self.chunksz = chunksz
        self.share_memory = share_memory
        self.aspect = aspect
        self.start_time = start_time
        self.snippet = snippet
        self.suffix = suffix
        self.aspect_dir = aspect_dir
        self.ftype = ftype
        self.wide_edge_thresh = wide_edge_thresh
        self.narrow_edge_thresh = narrow_edge_thresh
        self.single_leg = single_leg

    def __repr__(self):
        params = [ f"{k}={v!r}" for k,v in self.__dict__ ]
        params.sort()
        return "PipeContext(" + ", ".join(params) + ")"

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
            "suffix": self.suffix,
            "ftype": self.ftype
        }

    def asdict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def eclipse_path(self, remote=False):
        root = self.local if remote is False else self.remote
        return Path(root, f"e{str(self.eclipse).zfill(5)}")

    def temp_path(self, remote=False):
        root = self.local if remote is False else self.remote
        return Path(root, "temp", f"e{str(self.eclipse).zfill(5)}")

    def __call__(self, remote=False, **kwargs):
        kwargs = self.pathdict() | kwargs

        apertures = kwargs.pop("aperture_sizes")
        kwargs.setdefault("aperture", apertures[0])

        local_base = kwargs.pop("local")
        remote_base = kwargs.pop("remote")
        root = kwargs.pop("root", None)
        if root is None:
            root = remote_base if remote is True else local_base
        if not isinstance(root, Path):
            root = Path(root)
        kwargs["root"] = root

        frame = kwargs.pop("frame", None)
        if frame is not None and kwargs.get("start") is None:
            kwargs["start"] = frame

        return eclipse_to_paths(**kwargs)

    def __getitem__(self, key):
        return self()[key]

    def explode_legs(self):
        """
        generate a list of PipeContexts, one for each leg of the eclipse
        """
        legs = get_legs(self.eclipse, aspect_dir=self.aspect_dir)
        kdict = self.asdict()
        del kdict['leg']
        return [PipeContext(leg=leg, **kdict) for leg in legs]


def check_eclipse(eclipse, aspect_dir: None | str | Path = None):
    from gPhoton.aspect import aspect_tables
    e_warn: list[str] = []
    e_error: list[str] = []
    if eclipse > 47000:
        e_error.append("CAUSE data w/eclipse>47000 are not yet supported.")
    if 37423 < eclipse <= 38149:
        e_error.append(f"Eclipse {eclipse} is post-CSP and pre-TAC switch, "
                       "suspect data that should not be processed.")
    meta = aspect_tables(
        eclipse=eclipse, tables="metadata", aspect_dir=aspect_dir
    )[0]
    if len(meta) == 0:
        e_error.append(f"No metadata found for e{eclipse}.")
        return e_warn, e_error
    actual, nominal = titular_legs(eclipse, aspect_dir=aspect_dir)
    if actual != nominal:
        obstype = meta['obstype'].to_pylist()[0]
        e_warn.append(
            f"Note: e{eclipse} observation-level metadata specifies "
            f"{nominal} legs, but only {actual} appear(s) to have "
            f"been completed. Obstype is {obstype}."
        )
    return e_warn, e_error
