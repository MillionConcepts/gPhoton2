"""utilities that retrieve and yield GALEX cal files / objects."""

import importlib.resources
from typing import Any, Literal

import numpy as np
from astropy.io.fits.header import Header as FITSHeader

from gPhoton.io.fits_utils import pyfits_open_igzip, get_tbl_data
from gPhoton.types import NDArray, GalexBand

def check_band(band: str) -> str:
    if band not in ("NUV", "FUV"):
        raise ValueError("Band must be NUV or FUV")
    return band


def check_xy(xy: str) -> str:
    if xy not in ("x", "y"):
        raise ValueError("xy must be x or y.")
    return xy


def enforce_native_byteorder(arr: NDArray[Any]) -> NDArray[Any]:
    # NOTE: I'm not sure if np.dtype.isnative's behavior for structured data
    #  types is consistent across versions, so it would be nice if we could do
    #  a single check here in that case, but I don't know that we can
    if arr.dtype.fields is None and arr.dtype.isnative:
        return arr
    elif arr.dtype.fields is None:
        return arr.byteswap().view(arr.dtype.newbyteorder("="))
    # structured array case
    swap_targets, swapped_dtype = [], []
    for name, field in arr.dtype.fields.items():
        if field[0].isnative is False:
            swap_targets.append(name)
            swapped_dtype.append((name, field[0].newbyteorder("=")))
        elif "V" not in str(field[0]):
            swapped_dtype.append((name, field[0]))
        else:
            # NOTE: should never happen here
            swapped_dtype.append((name, np.dtype("O")))
    return arr.astype(swapped_dtype)


def read_cal_fits(fn: str, dim: int = 0) -> tuple[NDArray[Any], FITSHeader]:
    files = importlib.resources.files("gPhoton.cal_data")
    with importlib.resources.as_file(files / fn) as path:
        with pyfits_open_igzip(path, memmap=1) as hdulist:
            data = hdulist[dim].data
            # Should this be hdulist[dim].header?
            header = hdulist[0].header
            return enforce_native_byteorder(data), header


def read_cal_tbl(fn: str) -> NDArray[Any]:
    files = importlib.resources.files("gPhoton.cal_data")
    with importlib.resources.as_file(files / fn) as path:
        return get_tbl_data(path)


def wiggle(
    band: GalexBand,
    xy: Literal["x", "y"]
) -> tuple[NDArray[np.float32], FITSHeader]:
    b = check_band(band)
    d = check_xy(xy)
    return read_cal_fits(f"{b}_wiggle_{d}.fits")


# actual dtype for this one is a mess:
# [('YA', '<i2'), ('YB', '<i2'), ('XB', '<i2'), ('yy', '<i2'), ('ycor', '<f8')]
def wiggle2() -> tuple[NDArray[Any], FITSHeader]:
    """The post-CSP wiggle file."""
    return read_cal_fits("WIG2_Sep2010.fits", dim=1)


def walk(
    band: GalexBand,
    xy: Literal["x", "y"]
) -> tuple[NDArray[np.float32], FITSHeader]:
    b = check_band(band)
    d = check_xy(xy)
    return read_cal_fits(f"{b}_walk_{d}.fits")


# [('Q', '<i2'), ('YB', '<i2'), ('yy', '<i2'), ('ycor', '<f8')]
def walk2() -> tuple[NDArray[Any], FITSHeader]:
    """The post-CSP walk file."""
    return read_cal_fits("WLK2_Sep2010.fits", dim=1)

# [('YB', '<i2'), ('yy', '<i2'), ('ycor', '<f8')]
def clock2() -> tuple[NDArray[Any], FITSHeader]:
    """The post-CSP clock file."""
    return read_cal_fits("CLK2_Sep2010.fits", dim=1)


def linearity(
    band: GalexBand,
    xy: Literal["x", "y"]
) -> tuple[NDArray[np.float32], FITSHeader]:
    b = check_band(band)
    d = check_xy(xy)
    return read_cal_fits(f"{b}_NLC_{d}_det2sky.fits")


def flat(band: GalexBand) -> tuple[NDArray[np.float32], FITSHeader]:
    b = check_band(band)
    return read_cal_fits(f"{b}_flat.fits")


def distortion(
    band: GalexBand,
    xy: Literal["x", "y"],
    eclipse: int,
    raw_stimsep: float,
) -> tuple[NDArray[np.float32], FITSHeader]:
    b = check_band(band).lower()
    d = check_xy(xy)
    index = ""
    if band == "NUV" and eclipse > 37423:
        if raw_stimsep < 5136.3:
            index = "a"
        elif raw_stimsep < 5137.25:
            index = "b"
        else:
            index = "c"
    return read_cal_fits(f"{b}_distortion_cube_d{d}{index}.fits")


def offset(xy: Literal["x", "y"]) -> NDArray[np.float64]:
    d = check_xy(xy)
    return read_cal_tbl(f"fuv_d{d}_fdttdc_coef_0.tbl")


def mask(band: GalexBand) -> tuple[NDArray[np.float32], FITSHeader]:
    b = check_band(band)
    return read_cal_fits(f"{b}_mask.fits")
