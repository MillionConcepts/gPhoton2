"""utilities that retrieve and yield GALEX cal files / objects."""

import importlib.resources

import numpy as np

from gPhoton.io.fits_utils import get_fits_data, get_fits_header, get_tbl_data


def check_band(band):
    if band not in ["NUV", "FUV"]:
        raise ValueError("Band must be NUV or FUV")
    return band


def check_xy(xy):
    if xy not in ["x", "y"]:
        raise ValueError("xy must be x or y.")
    return xy


def enforce_native_byteorder(arr: np.ndarray) -> np.ndarray:
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
            swapped_dtype.append((name, "O"))
    return arr.astype(swapped_dtype)


def read_data(fn, dim=0):
    files = importlib.resources.files("gPhoton.cal_data")
    with importlib.resources.as_file(files / fn) as path:
        if ".fits" in fn:
            data = get_fits_data(path, dim=dim)
            header = get_fits_header(path, ext=0)
            return enforce_native_byteorder(data), header
        elif ".tbl" in fn:
            return get_tbl_data(path)
        else:
            raise ValueError("Unrecognized data type: {ext}"
                             .format(ext=fn[-4:]))


def wiggle(band, xy):
    fn = "{b}_wiggle_{d}.fits".format(b=check_band(band), d=check_xy(xy))
    return read_data(fn)


def wiggle2():
    """The post-CSP wiggle file."""
    return read_data("WIG2_Sep2010.fits", dim=1)


def avgwalk(band, xy):
    fn = "{b}_avgwalk_{d}.fits".format(b=check_band(band), d=check_xy(xy))
    return read_data(fn)


def walk(band, xy):
    fn = "{b}_walk_{d}.fits".format(b=check_band(band), d=check_xy(xy))
    return read_data(fn)


def walk2():
    """The post-CSP walk file."""
    return read_data("WLK2_Sep2010.fits", dim=1)


def clock2():
    """The post-CSP clock file."""
    return read_data("CLK2_Sep2010.fits", dim=1)


def linearity(band, xy):
    fn = "{b}_NLC_{d}_det2sky.fits".format(b=check_band(band), d=check_xy(xy))
    return read_data(fn)


def flat(band):
    return read_data(f"{check_band(band)}_flat.fits")


def distortion(band, xy, eclipse, raw_stimsep):
    index = ""
    if band == "NUV":
        if eclipse > 37423:
            if raw_stimsep < 5136.3:
                index = "a"
            elif raw_stimsep < 5137.25:
                index = "b"
            else:
                index = "c"
    fn = "{b}_distortion_cube_d{d}{i}.fits".format(
        b=check_band(band).lower(), d=check_xy(xy), i=index
    )
    return read_data(fn)


def offset(xy):
    fn = "fuv_d{d}_fdttdc_coef_0.tbl".format(d=check_xy(xy))
    return read_data(fn)


def mask(band):
    return read_data(f"{check_band(band)}_mask.fits")
