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
    # use >0 not >=0 because '.cshrc' does not have an extension
    if (dot := fn.rfind(".")) > 0:
        ext = fn[dot:]
        if ext == ".gz" and (dot2 := fn[:dot].rfind(".")) > 0:
            ext = fn[dot2:dot]
    else:
        ext = ''
    files = importlib.resources.files("gPhoton.cal_data")
    with importlib.resources.as_file(files / fn) as path:
        if ext == ".fits":
            data = get_fits_data(path, dim=dim)
            header = get_fits_header(path)
            return enforce_native_byteorder(data), header
        elif ext == ".tbl":
            return get_tbl_data(path)
        else:
            raise ValueError(f"Unrecognized data type: {ext}")


def wiggle(band, xy):
    b = check_band(band)
    d = check_xy(xy)
    return read_data(f"{b}_wiggle_{d}")


def wiggle2():
    """The post-CSP wiggle file."""
    return read_data("WIG2_Sep2010.fits", dim=1)


def avgwalk(band, xy):
    b = check_band(band)
    d = check_xy(xy)
    return read_data(f"{b}_avgwalk_{d}.fits")


def walk(band, xy):
    b = check_band(band)
    d = check_xy(xy)
    return read_data(f"{b}_walk_{d}.fits")


def walk2():
    """The post-CSP walk file."""
    return read_data("WLK2_Sep2010.fits", dim=1)


def clock2():
    """The post-CSP clock file."""
    return read_data("CLK2_Sep2010.fits", dim=1)


def linearity(band, xy):
    b = check_band(band)
    d = check_xy(xy)
    return read_data(f"{b}_NLC_{d}_det2sky.fits")


def flat(band):
    b = check_band(band)
    return read_data(f"{b}_flat.fits")


def distortion(band, xy, eclipse, raw_stimsep):
    b = check_band(band)
    d = check_xy(xy)
    index = ""
    if band == "NUV" and eclipse > 37423:
        if raw_stimsep < 5136.3:
            index = "a"
        elif raw_stimsep < 5137.25:
            index = "b"
        else:
            index = "c"
    return read_data(f"{b}_distortion_cube_d{d}{index}.fits")


def offset(xy):
    d = check_xy(xy)
    return read_data(f"fuv_d{d}_fdttdc_coef_0.tbl")


def mask(band):
    b = check_band(band)
    return read_data(f"{b}_mask.fits")
