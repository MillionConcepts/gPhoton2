"""
shared parameters that depend on the eclipse and on how gphoton
is being run but not on anything else
"""

from math import floor
from typing import Literal, Optional

from gPhoton.types import Pathlike, GalexBand


def intfill(obj, zfill=4):
    obj = float(obj)
    integer = str(floor(obj)).zfill(zfill)
    if (decimal := obj - floor(obj)) > 0:
        # round for floating-point error
        return integer + str(round(decimal, 4))[2:]
    return integer


def emojified(compression, depth, leg, band, eclipse_base, frame):
    band_emoji = {"NUV": "🌠", "FUV": "👻"}[band]
    ext = {"gzip": ".fits.gz", "none": ".fits", "rice": ".fits"}[compression]
    comp = {"gzip": "🤐", "none": "🍵", "rice": "🍚"}[compression]
    frame = "🎥" if frame is None else f"f{str(frame).zfill(4)}"
    prefix = f"{eclipse_base}-{band_emoji}"
    file_dict = {
        "raw6": f"{prefix}-raw6.fits.gz",
        "photonfile": f"{prefix}-🦿{leg}.parquet",
        "image": f"{prefix}️-🦿{leg}-🖼-{comp}{ext}",
        "extended_catalogs": f"{prefix}-🦿{leg}-extended-sources.csv"
    }
    if depth is not None:
        depth = str(depth).zfill(4)
        file_dict |= {
            "movie": f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-{comp}{ext}",
            # stem -- multiple aperture sizes possible
            "photomfile": f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-photom-",
            "expfiles": f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-exptime.csv"
        }
    else:
        file_dict |= {"photomfiles": [f"{prefix}-⏱️full-{leg}-🖼-photom-"]}
    return file_dict


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
    suffix: Optional[str] = None, # suffix for file names
    **kwargs,
) -> dict[str, str]:
    """
    generate canonical paths for files associated with a given eclipse,
    optionally including files at a specific depth
    """
    root = "data" if root is None else root
    zpad = str(eclipse).zfill(5)
    leg = str(leg).zfill(2)
    eclipse_path = f"{root}/e{zpad}/"
    eclipse_base = f"{eclipse_path}e{zpad}"

    if kwargs.get("emoji") is True:
        return emojified(compression, depth, leg, band, eclipse_base, start)

    ext = {"gzip": ".fits.gz", "none": ".fits", "rice": ".fits"}[compression]
    comp = {"gzip": "g", "none": "u", "rice": "r"}[compression]
    mode = {"direct": "d", "grism": "g", "opaque": "o"}[mode]
    start = "movie" if start is None else f"t{intfill(start)}"
    depth = None if depth is None else f"f{intfill(depth)}"
    prefix = f"{eclipse_base}-{band[0].lower()}{mode}"
    suffix = "" if suffix is None else f"-{suffix}"
    aper = "" if aperture is None else str(aperture).replace(".", "_")
    file_dict = {
        "raw6": f"{prefix}-raw6.fits.gz",
        "photonfile": f"{prefix}-b{leg}.parquet",
        "image": f"{prefix}-ffull-b{leg}-image-{comp}{ext}",
        # TODO: frames, etc. -- decide exactly how once we are using
        #  extended source detection on movies
        "extended_catalog": f"{prefix}-b{leg}-extended-sources.csv", # TODO: is this being used?
        "extended_shapes":f"{prefix}-b{leg}-extended-shapes.csv"

    }
    if depth is not None:
        file_dict |= {
            "movie": f"{prefix}-{depth}-b{leg}-{start}-{comp}{ext}",
            "photomfile": f"{prefix}-{depth}-b{leg}-{start}-photom-{aper}{suffix}.csv",
            "expfile": f"{prefix}-{depth}-b{leg}-{start}-exptime.csv",
        }
    else:
        file_dict[
            "photomfile"
        ] = f"{prefix}-ffull-b{leg}-image-photom-{aper}{suffix}.csv"
    return file_dict
