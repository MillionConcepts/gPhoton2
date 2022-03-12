"""
.. module:: wcs
   :synopsis: Functions for generating World Coordinate System (WCS) objects.
"""


from typing import Sequence

import astropy.wcs
import numpy as np

import gPhoton.constants as c


# ------------------------------------------------------------------------------


def make_wcs(
    skypos: Sequence,
    pixsz: float = 0.000416666666666667,
    imsz: Sequence[int] = (3200, 3200),
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
    assumes gnomonically-projected ra/dec values; scales ra bounds to
    approximate distortion in pixel size
    """
    import math
    real_ra = radec[:, 0][np.isfinite(radec[:, 0])]
    real_dec = radec[:, 1][np.isfinite(radec[:, 1])]
    ra_range = real_ra.min(), real_ra.max()
    dec_range = real_dec.min(), real_dec.max()
    # viewports in which ra wraps around 360
    if ra_range[1] - ra_range[0] > 350:
        real_ra[real_ra > 180] -= 360
        ra_range = real_ra.min(), real_ra.max()
    # WCS center pixel in sky coordinates
    ra0, dec0 = (np.mean(ra_range), np.mean(dec_range))
    ra0 = ra0 if ra0 > 0 else ra0 + 360
    # scale by approximate ra-direction distortion of gnomonic projection
    ra_offset = (ra_range[1] - ra_range[0]) * math.cos(math.radians(dec0))
    imsz = (
        int(np.ceil((dec_range[1] - dec_range[0]) / c.DEGPERPIXEL)),
        int(np.ceil(ra_offset / c.DEGPERPIXEL)),
    )
    return make_wcs((ra0, dec0), imsz=imsz, pixsz=c.DEGPERPIXEL)
