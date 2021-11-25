"""utilities that retrieve and yield GALEX cal files / objects."""

import os

import numpy as np

from gPhoton import cal_dir
from gPhoton.io.fits_utils import get_fits_data, get_fits_header, get_tbl_data
from gPhoton.io.netutils import download_with_progress_bar


# Remote repository for GALEX calibration files.
CAL_URL = "https://archive.stsci.edu/prepds/gphoton/cal/cal/"


def check_band(band):
    if band not in ["NUV", "FUV"]:
        raise ValueError("Band must be NUV or FUV")
    return band


def check_xy(xy):
    if xy not in ["x", "y"]:
        raise ValueError("xy must be x or y.")
    return xy


def read_data(fn, dim=0):
    path = os.path.join(cal_dir, fn)
    # Download the file if it doesn't exist locally.
    if not os.path.exists(path):
        data_url = "{b}/{f}".format(b=CAL_URL, f=fn)
        download_with_progress_bar(data_url, path)
    if ".fits" in fn:
        data = get_fits_data(path, dim=dim)
        header = get_fits_header(path)
        if isinstance(data, np.recarray):
            for name in data.names:
                data[name] = data[name].byteswap().newbyteorder()
        else:
            data = data.byteswap().newbyteorder()
        return data, header
    elif ".tbl" in fn:
        return get_tbl_data(path)
    else:
        raise ValueError("Unrecognized data type: {ext}".format(ext=fn[-4:]))


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
        if eclipse > 37460:
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
