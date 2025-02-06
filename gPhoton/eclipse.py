"""
shared parameters that depend on the eclipse and on how gphoton
is being run but not on anything else
"""

from collections.abc import Mapping
from functools import cache
from math import modf
from pathlib import Path
from typing import Any, Callable, Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from _typeshed import ConvertibleToFloat

from gPhoton.types import Pathlike, GalexBand


MODE_TAG = {
    "direct": ["d", "⭐"],
    "grism":  ["g", "🌈"],
    "opaque": ["o", "🔲"],
}

BAND_TAG = {
    "NUV": ["n", "🌠"],
    "FUV": ["f", "👻"],
}

COMP_TAG_EXT = {
    "gzip": ["g.fits.gz", "🤐.fits.gz"],
    "none": ["n.fits",    "🍵.fits"],
    "rice": ["r.fits",    "🍚.fits"],
}

# 'b' is for 'boresight' and is used because 'l' is often too
# hard to tell apart from '1' or 'I'.
LEG_TAG   = ["b", "🦿"]
DEPTH_TAG = ["f", "⏱️"]
START_TAG = ["t", "t"]

MOVIE_TAG = ["movie", "🎥"]
IMAGE_TAG = ["image", "🖼"]


def intfill(obj: 'ConvertibleToFloat', zfill: int = 4) -> str:
    """
    Zero-fill the integer part of `obj` to `zfill` places, then
    append the fractional part of `obj`, rounded to four decimal
    places, *without* an intervening decimal point.

    >>> intfill(123)
    '0123'
    >>> intfill(123.0)
    '0123'
    >>> intfill(123.45)
    '012345'
    """
    if isinstance(obj, int):
        # shortcut - save some conversions
        return str(obj).zfill(zfill)

    if not isinstance(obj, float):
        obj = float(obj)

    decimal, integer = modf(obj)
    # both values returned by modf are floats, and if the input was
    # negative, both are negative.
    integer = str(int(integer)).zfill(zfill)
    if not decimal:
        return integer
    return integer + str(round(abs(decimal), 4))[2:]


def format_leg(leg: int, emoji: bool) -> str:
    return LEG_TAG[emoji] + str(leg).zfill(2)


def format_start(start: float | None, depth: int | None, emoji: bool) -> str:
    if depth is None:
        return IMAGE_TAG[emoji]
    elif start is None:
        return MOVIE_TAG[emoji]
    else:
        return START_TAG[emoji] + intfill(start)


def format_depth(depth: int | None, emoji: bool) -> str:
    return DEPTH_TAG[emoji] + ("full" if depth is None else str(depth).zfill(4))


def format_aperture(aperture: float | None, emoji: bool) -> str:
    return "" if aperture is None else str(aperture).replace(".", "_")


@cache
def eclipse_prefix(
    eclipse: int,
    band: GalexBand,
    mode: str,
    emoji: bool,
) -> tuple[Path, str]:
    eclipse = "e" + str(eclipse).zfill(5)
    band = BAND_TAG[band][emoji]
    mode = MODE_TAG[mode][emoji]
    return (Path(eclipse), f"{eclipse}-{band}{mode}")


@cache
def raw6_path(
    eclipse: int,
    band: GalexBand,
    mode: str,
    *,
    emoji: bool = False
) -> Path:
    d, p = eclipse_prefix(eclipse, band, mode, emoji)
    return d / f"{p}raw6.fits.gz"


@cache
def photonfile_path(
    eclipse: int,
    leg: int,
    band: GalexBand,
    mode: str,
    *,
    emoji: bool = False,
) -> Path:
    leg = format_leg(leg, emoji)
    d, p = eclipse_prefix(eclipse, band, mode, emoji)
    return d / f"{p}{leg}.parquet"


@cache
def image_path(
    eclipse: int,
    leg: int,
    band: GalexBand,
    mode: str,
    *,
    compression: str = "fits",
    emoji: bool = False,
) -> Path:
    leg = format_leg(leg, emoji)
    comp = COMP_TAG_EXT[compression][emoji]
    img = IMAGE_TAG[emoji]
    depth = format_depth(None, emoji)
    d, p = eclipse_prefix(eclipse, band, mode, emoji)
    return d / f"{p}{depth}-{leg}-{img}-{comp}"


@cache
def extended_shapes_path(
    eclipse: int,
    leg: int,
    band: GalexBand,
    mode: str,
    *,
    emoji: bool = False,
) -> Path:
    # TODO: frames, etc. -- decide exactly how once we are using
    #  extended source detection on movies
    # TODO: is this being used?
    leg = format_leg(leg, emoji)
    d, p = eclipse_prefix(eclipse, band, mode, emoji)
    return d / f"{p}{leg}-extended-shapes.csv"


@cache
def extended_catalog_path(
    eclipse: int,
    leg: int,
    band: GalexBand,
    mode: str,
    *,
    emoji: bool = False,
) -> Path:
    leg = format_leg(leg, emoji)
    d, p = eclipse_prefix(eclipse, band, mode, emoji)
    return d / f"{p}{leg}-extended-sources.csv"


@cache
def photomfile_path(
    eclipse: int,
    leg: int,
    band: GalexBand,
    mode: str,
    *,
    depth: int | None = None,
    start: float | None = None,
    aperture: float | None = None,
    suffix: str | None = None,
    emoji: bool = False,
):
    leg = format_leg(leg, emoji)
    start = format_start(start, depth, emoji)
    depth = format_depth(depth, emoji)
    aper = format_aperture(aperture, emoji)
    suffix = "" if suffix is None else "-" + suffix
    d, p = eclipse_prefix(eclipse, band, mode, emoji)
    return d / f"{p}{depth}-{leg}-{start}-photom-{aper}{suffix}.csv"


@cache
def movie_path(
    eclipse: int,
    leg: int,
    band: GalexBand,
    mode: str,
    *,
    depth: int,
    start: float | None = None,
    compression: str = "fits",
    emoji: bool = False,
):
    if depth is None:
        raise ValueError("movies must have a depth")
    leg = format_leg(leg, emoji)
    start = format_start(start, depth, emoji)
    depth = format_depth(depth, emoji)
    comp = COMP_TAG_EXT[compression][emoji]
    d, p = eclipse_prefix(eclipse, band, mode, emoji)
    return d / f"{p}{depth}-{leg}-{start}-{comp}"


@cache
def expfile_path(
    eclipse: int,
    leg: int,
    band: GalexBand,
    mode: str,
    *,
    depth: int,
    start: float | None = None,
    emoji: bool = False,
):
    if depth is None:
        raise ValueError("expfiles must have a depth")
    leg = format_leg(leg, emoji)
    start = format_start(start, depth, emoji)
    depth = format_depth(depth, emoji)
    d, p = eclipse_prefix(eclipse, band, mode, emoji)
    return d / f"{p}{depth}-{leg}-{start}-exptime.csv"


class EclipsePaths(Mapping):
    """lazy mapping that constructs paths for a specific eclipse's
    data files"""
    def __init__(
        self,
        eclipse: int,
        *,
        band: GalexBand = "NUV",
        depth: int | None = None,
        compression: Literal["none", "gzip", "rice"] = "gzip",
        root: Pathlike | None = None,
        start: float | None = None,
        mode: str = "direct",
        leg: int = 0,
        aperture: float | None = None,
        suffix: str | None = None, # suffix for file names
        emoji: bool = False,
    ):
        if root is not None and not isinstance(root, Path):
            root = Path(root)

        self._eclipse = eclipse
        self._band = band
        self._depth = depth
        self._compression = compression
        self._root = root
        self._start = start
        self._mode = mode
        self._leg = leg
        self._aperture = aperture
        self._suffix = suffix
        self._emoji = emoji

        def raw6():
            s = self
            return raw6_path(s._eclipse, s._band, s._mode,
                             emoji=s._emoji)

        def photonfile():
            s = self
            return photonfile_path(s._eclipse, s._leg, s._band, s._mode,
                                   emoji=s._emoji)

        def image():
            s = self
            return raw6_path(s._eclipse, s._leg, s._band, s._mode,
                             compression=s._compression,
                             emoji=s._emoji)

        def extended_shapes():
            s = self
            return extended_shapes_path(s._eclipse, s._leg, s._band, s._mode,
                                        emoji=s._emoji)


        def extended_catalog():
            s = self
            return extended_catalog_path(s._eclipse, s._leg, s._band, s._mode,
                                         emoji=s._emoji)

        def photomfile():
            s = self
            return photomfile_path(s._eclipse, s._leg, s._band, s._mode,
                                   depth=s._depth, start=s._start,
                                   aperture=s._aperture, suffix=s.suffix,
                                   emoji=s._emoji)

        def movie():
            s = self
            return movie_path(s._eclipse, s._leg, s._band, s._mode,
                              depth=s._depth, start=s._start,
                              compression=s._compression, emoji=s._emoji)

        def expfile():
            s = self
            return expfile_path(s._eclipse, s._leg, s._band, s._mode,
                                s._depth, s._start, s._emoji)


        self._paths: dict[str, Path | Callable[[], Path]] = {
            "raw6": raw6,
            "photonfile": photonfile,
            "image": image,
            "extended_shapes": extended_shapes,
            "extended_catalog": extended_catalog,
            "photomfile": photomfile,
        }
        if self._depth is not None:
            self._paths["movie"] = movie
            self._paths["expfile"] = expfile

    # must use 'key: Any' here because that's what the superclass does
    # (but this is safe; if it's not a string, it will not be in _paths)
    def __contains__(self, key: Any) -> bool:
        return key in self._paths

    def __getitem__(self, key: str) -> Path:
        lazy_path = self._paths[key]
        if not isinstance(lazy_path, Path):
            lazy_path = lazy_path()
            if self._root is not None:
                lazy_path = self._root / lazy_path
            self._paths[key] = lazy_path
        return lazy_path

    def __iter__(self):
        return iter(self._paths)

    def __len__(self):
        return len(self._paths)

    def __eq__(self, other):
        # equality is based only on the input parameters, not the
        # paths we may or may not have emitted yet (all of which are
        # pure functions of the input parameters, anyway)
        return (
            isinstance(other, type(self))
            and self._eclipse     == other._eclipse
            and self._band        == other._band
            and self._depth       == other._depth
            and self._compression == other._compression
            and self._start       == other._start
            and self._mode        == other._mode
            and self._leg         == other._leg
            and self._aperture    == other._aperture
            and self._suffix      == other._suffix
            and self._emoji       == other._emoji
        )


def eclipse_to_paths(
    eclipse: int,
    band: GalexBand = "NUV",
    depth: int | None = None,
    compression: Literal["none", "gzip", "rice"] = "gzip",
    root: Pathlike | None = None,
    start: float | None = None,
    mode: str = "direct",
    leg: int = 0,
    aperture: float | None = None,
    suffix: str | None = None, # suffix for file names
    emoji: bool = False,
) -> EclipsePaths:
    """
    generate canonical paths for files associated with a given eclipse,
    optionally including files at a specific depth
    """
    return EclipsePaths(
        eclipse,
        band=band,
        depth=depth,
        compression=compression,
        root=root,
        start=start,
        mode=mode,
        leg=leg,
        aperture=aperture,
        suffix=suffix,
        emoji=emoji,
    )
