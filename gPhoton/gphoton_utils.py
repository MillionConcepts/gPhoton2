"""
.. module:: gphoton_utils
   :synopsis: Read, plot, time conversion, and other functionality useful when
       dealing with gPhoton data.
"""

from __future__ import absolute_import, division, print_function

from typing import Sequence

import astropy.wcs
import numpy as np

import gPhoton.constants as c
from gPhoton import cal
from gPhoton.MCUtils import get_fits_header


# Core and Third Party imports.
# TODO: comment these back in if necessary
# import scipy.stats
# import matplotlib.pyplot as plt
# gPhoton imports.
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------

def make_wcs(
    skypos: Sequence,
    pixsz: float=0.000416666666666667,
    imsz: Sequence[int, int] = (3200, 3200),
) -> astropy.wcs.WCS:
    """
    makes a WCS object from passed center ra/dec, scale, and image size
    parameters. by default, uses the nominal image size and pixel scale
    values from the internal mission intensity map products.
    """
    wcs = astropy.wcs.WCS(naxis=2)
    wcs.wcs.cdelt = np.array([-pixsz, pixsz])
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [(imsz[1] / 2.0) + 0.5, (imsz[0] / 2.0) + 0.5]
    wcs.wcs.crval = skypos
    return wcs


def make_bounding_wcs(radec: np.ndarray) -> astropy.wcs.WCS:
    """
    makes a WCS solution for a given range of ra/dec values,
    at the constant degree-per-pixel scale set in gPhoton.constants.DEGPERPIXEL
    """
    real_ra = radec[:, 0][np.isfinite(radec[:, 0])]
    real_dec = radec[:, 1][np.isfinite(radec[:, 1])]
    ra_range = real_ra.min(), real_ra.max()
    dec_range = real_dec.min(), real_dec.max()
    center_skypos = (np.mean(ra_range), np.mean(dec_range))
    imsz = (
        int(np.ceil((dec_range[1] - dec_range[0]) / c.DEGPERPIXEL)),
        int(np.ceil((ra_range[1] - ra_range[0]) / c.DEGPERPIXEL)),
    )
    return make_wcs(center_skypos, imsz=imsz, pixsz=c.DEGPERPIXEL)


def find_fuv_offset(scstfile, raise_invalid = True):
    """
    Computes NUV->FUV center offset based on a lookup table.

    :param scstfile: Name of the spacecraft state (-scst) FITS file.

    :type scstfile: str

    :returns: tuple -- A two-element tuple containing the x and y offsets.
    """

    fodx_coef_0, fody_coef_0, fodx_coef_1, _ = (0.0, 0.0, 0.0, 0.0)

    scsthead = get_fits_header(scstfile)

    print("Reading header values from scst file: ", scstfile)

    try:
        eclipse = int(scsthead["eclipse"])
    except:
        print("WARNING: ECLIPSE is not defined in SCST header.")
        try:
            eclipse = int(scstfile.split("/")[-1].split("-")[0][1:])
            print("         Using {e} from filename.".format(e=eclipse))
        except:
            print("         Unable to infer eclipse from filename.")
            return 0.0, 0.0

    try:
        fdttdc = float(scsthead["FDTTDC"])
    except KeyError:
        print("WARNING: FUV temperature value missing from SCST.")
        print("         This is probably not a valid FUV observation.")
        if raise_invalid is True:
            raise ValueError("This is probably not a valid FUV observation.")
        return 0.0, 0.0

    print(
        "Offsetting FUV image for eclipse {e} at {t} degrees.".format(
            e=eclipse, t=fdttdc
        )
    )

    fodx_coef_0 = cal.offset("x")[eclipse - 1, 1]
    fody_coef_0 = cal.offset("y")[eclipse - 1, 1]

    fodx_coef_1 = 0.0
    fody_coef_1 = 0.3597

    if (fdttdc <= 20.0) or (fdttdc >= 40.0):
        print("ERROR: FDTTDC is out of range at {t}".format(t=fdttdc))
        if raise_invalid is True:
            raise ValueError("FUV temperature out of range.")
        return 0.0, 0.0
    else:
        xoffset = fodx_coef_0 - (fodx_coef_1 * (fdttdc - 29.0))
        yoffset = fody_coef_0 - (fody_coef_1 * (fdttdc - 29.0))
        print(
            "Setting FUV offsets to x={x}, y={y}".format(x=xoffset, y=yoffset)
        )

    return xoffset, yoffset

