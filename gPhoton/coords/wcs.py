"""
.. module:: wcs
   :synopsis: Functions for generating World Coordinate System (WCS) objects.
"""
from itertools import product
import math
from operator import add, sub
from typing import Sequence

import astropy.wcs
import numpy as np

import gPhoton.constants as c


# ------------------------------------------------------------------------------


def make_wcs(
    skypos: Sequence,
    pixsz: float = 0.000416666666666667,
    imsz: Sequence[int] = (3200, 3200),
    proj=("RA---TAN", "DEC--TAN")
) -> astropy.wcs.WCS:
    """
    makes a WCS object from passed center ra/dec, scale, and image size
    parameters. by default, uses the nominal image size and pixel scale
    values from the internal mission intensity map products, and a gnomonic
    projection.
    """
    system = astropy.wcs.WCS(naxis=2)
    system.wcs.cdelt = np.array([-pixsz, pixsz])
    system.wcs.ctype = list(proj)
    system.wcs.crpix = [(imsz[1] / 2.0) + 0.5, (imsz[0] / 2.0) + 0.5]
    system.wcs.crval = skypos
    return system


def make_bounding_wcs(
    radec: np.ndarray,
    pixsz: float = c.DEGPERPIXEL,
    proj: Sequence[str] = ("RA---TAN", "DEC--TAN")
) -> astropy.wcs.WCS:
    """
    makes a WCS solution for a given range of ra/dec values
    by default, assumes gnomonically-projected ra/dec values; scales ra bounds
    to approximate distortion in pixel size
    radec: n x 2 array with ra in first column and dec in second
    pixsz: size of returned WCS's pixels in square degrees;
    defaults to degree-per-pixel scale set in gPhoton.constants.DEGPERPIXEL
    """
    real_ra = radec[:, 0][np.isfinite(radec[:, 0])]
    real_dec = radec[:, 1][np.isfinite(radec[:, 1])]
    ra_range = real_ra.min(), real_ra.max()
    dec_range = real_dec.min(), real_dec.max()
    # handle viewports in which ra wraps around 360
    if ra_range[1] - ra_range[0] > 350:
        real_ra[real_ra > 180] -= 360
        ra_range = real_ra.min(), real_ra.max()
    # WCS center pixel in sky coordinates
    ra0, dec0 = (np.mean(ra_range), np.mean(dec_range))
    ra0 = ra0 if ra0 > 0 else ra0 + 360
    # scale ra-axis pixel size using cos(declination) to approximate
    # ra-direction distortion introduced by gnomonic projection
    ra_offset = (ra_range[1] - ra_range[0]) * math.cos(math.radians(dec0))
    imsz = (
        int(np.ceil((dec_range[1] - dec_range[0]) / pixsz)),
        int(np.ceil(ra_offset / pixsz)),
    )
    return make_wcs((ra0, dec0), imsz=imsz, pixsz=pixsz, proj=proj)


def translate_pc_keyword(keyword: str):
    """
    convert old-style fits wcs transformation keywords. this is not strictly
    necessary for any GALEX products, but is useful for some data fusion
    applications.
    """
    # note: i suppose this will fail for headers with hundreds
    # of dimensions -- they may not exist, though, and deserve special-purpose
    # code if they do
    if not keyword.startswith("PC0"):
        return keyword
    return keyword.replace("PC00", "PC").replace("00", "_")


def extract_wcs_keywords(header):
    """
    header formatting and WCS keyword handling can make astropy.wcs upset,
    it handles validation and fixes gracefully, but not quickly. faster
    to trim irrelevant keywords and fix old-style ones before feeding them to
    astropy.wcs.
    """
    wcs_words = ('CTYPE', 'CRVAL', 'CRPIX', 'CDELT', 'NAXIS', 'PC')
    return {
        translate_pc_keyword(k): header[k] for k in header.keys()
        if any([k.startswith(w) for w in wcs_words])
    }


def corners_of_a_square(ra, dec, side_length):
    """
    corners of a square centered at ra, dec with side length side_length
    in the order: upper right, lower right, upper left, lower left
    """
    return [
        (op1(ra, side_length / 2), op2(dec, side_length / 2))
        for op1, op2 in product((add, sub), (add, sub))
    ]


def sky_box_to_image_box(corners, system):
    """
    get image coordinates that correspond to a sky-coordinate square with
    specified corners (in whatever units wcs axes 1 and 2 are in, most likely
    degrees)
    """
    cuts = system.world_to_pixel_values(
        np.array(corners)[:, 0], np.array(corners)[:, 1]
    )
    rounded_index_map = map(
        round, (cuts[0].min(), cuts[0].max(), cuts[1].min(), cuts[1].max())
    )
    return tuple(rounded_index_map)
