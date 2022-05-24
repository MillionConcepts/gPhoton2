from itertools import product
from pathlib import Path

import astropy.wcs
import fast_histogram as fh
import fitsio
import numpy as np
from cytoolz import first

from gPhoton.coords.wcs import make_bounding_wcs, extract_wcs_keywords, \
    corners_of_a_square, sky_box_to_image_box
from gPhoton.io.fits_utils import pyfits_open_igzip, read_wcs_from_fits
from gPhoton.pretty import mb
from gPhoton.reference import eclipse_to_paths, Stopwatch, Netstat


def get_image_fns(*eclipses, band="NUV", root="test_data"):
    return [
        eclipse_to_paths(eclipse, root)[band]["image"] for eclipse in eclipses
    ]


def wcs_imsz(system: astropy.wcs.WCS):
    """
    image size associated with a WCS object. WARNING: not universally
    applicable! works if and only if the reference pixel is at the center of
    the image.
    """
    return (
        int((system.wcs.crpix[1] - 0.5) * 2),
        int((system.wcs.crpix[0] - 0.5) * 2),
    )


def wcs_ra_dec_corners(wcs_system):
    """
    corners, in sky coordinates, of the image associated with a WCS object.
    WARNING: not universally applicable! works if and only if the reference
    pixel of the WCS object is at the center of the image.
    """
    imsz = wcs_imsz(wcs_system)
    ymax, xmax = imsz[1], imsz[0]
    return wcs_system.pixel_to_world_values(
        (0, ymax, ymax, 0), (0, xmax, 0, xmax)
    )


def bounds_from_corners(corners):
    extremes = []
    for pred, coord in product((np.max, np.min), np.arange(len(corners[0]))):
        local_extrema = map(pred, [corner[coord] for corner in corners])
        extremes.append(pred(tuple(local_extrema)))
    return extremes


def make_shared_wcs(wcs_sequence):
    """
    WARNING: not universally applicable! works if and only if the reference
    pixels of the WCS objects in wcs_sequence are at the center of the images.
    if this is not relevant to your use case, explicitly construct
    sky-coordinate bounds and feed them to make_bounding_wcs.
    """
    corners = tuple(map(wcs_ra_dec_corners, wcs_sequence))
    ra_min, dec_min, ra_max, dec_max = map(
        np.float32, bounds_from_corners(corners)
    )
    return make_bounding_wcs(np.array([[ra_min, dec_min], [ra_max, dec_max]]))


def zero_flag_and_edge(cnt, flag, edge):
    cnt[~np.isfinite(cnt)] = 0
    cnt[np.nonzero(flag)] = 0
    cnt[np.nonzero(edge)] = 0
    return cnt


def project_to_shared_wcs(gphoton_fits, shared_wcs, nonzero=True):
    # TODO: rewrite to use fitsio.FITS / handle rice (range(-3, 0), etc.)
    cnt, flag, edge = [gphoton_fits[ix].data for ix in range(3)]
    cnt = zero_flag_and_edge(cnt, flag, edge)
    if nonzero is True:
        y_ix, x_ix = np.nonzero(cnt)
    else:
        indices = np.indices((cnt.shape[0], cnt.shape[1]), dtype=np.int16)
        y_ix, x_ix = indices[0].ravel(), indices[1].ravel()
    system = astropy.wcs.WCS(gphoton_fits[0].header)
    ra_input, dec_input = system.pixel_to_world_values(x_ix, y_ix)
    x_shared, y_shared = shared_wcs.wcs_world2pix(ra_input, dec_input, 1)
    return {
        "x": x_shared,
        "y": y_shared,
        "weight": cnt[y_ix, x_ix],
        "exptime": np.float32(gphoton_fits[0].header["EXPTIME"]),
    }


def bin_projected_weights(x, y, weights, imsz):
    binned = fh.histogram2d(
        y - 0.5,
        x - 0.5,
        bins=imsz,
        range=([[0, imsz[0]], [0, imsz[1]]]),
        weights=weights,
    )
    return binned


def get_full_frame_coadd_layer(gphoton_fits, shared_wcs):
    projection = project_to_shared_wcs(gphoton_fits, shared_wcs)
    return bin_projected_weights(
        projection["x"],
        projection["y"],
        projection["weight"] / projection["exptime"],
        wcs_imsz(shared_wcs),
    )


def coadd_image_files(image_files):
    headers, systems = read_wcs_from_fits(*image_files)
    shared_wcs = make_shared_wcs(systems)
    coadd = np.zeros(wcs_imsz(shared_wcs), dtype=np.float64)
    for image_file in image_files:
        print(image_file)
        coadd += get_full_frame_coadd_layer(
            pyfits_open_igzip(image_file), shared_wcs
        )
    return coadd


def project_slice_to_shared_wcs(
    values, individual_wcs, shared_wcs, ra_min, dec_min
):
    """
    Args:
        values: sliced values from source image
        individual_wcs: WCS object for full-frame source image
        shared_wcs: WCS object for coadd
        ra_min: minimum RA of pixels in values
        dec_min: minimum DEC of pixels in values
    """
    indices = np.indices((values.shape[0], values.shape[1]), dtype=np.int16)
    y_ix, x_ix = indices[0].ravel() + dec_min, indices[1].ravel() + ra_min
    ra_input, dec_input = individual_wcs.pixel_to_world_values(x_ix, y_ix)
    x_shared, y_shared = shared_wcs.wcs_world2pix(ra_input, dec_input, 1)
    return {
        "x": x_shared,
        "y": y_shared,
        "weight": values.ravel(),
    }


def coadd_galex_rice_slices(
    image_paths, ra, dec, side_length, stat=None, watch=None
):
    """
    top-level handler function for coadding slices from rice-compressed GALEX
    images.
    TODO: not fully integrated yet.
    """
    if watch is None:
        watch = Stopwatch()
    if stat is None:
        stat = Netstat()
    print(f"... planning cuts on {len(image_paths)} galex image(s) ...")
    hduls = [fitsio.FITS(file) for file in image_paths]
    headers = [hdul[1].read_header() for hdul in hduls]
    systems = [
        astropy.wcs.WCS(extract_wcs_keywords(header))
        for header in headers
    ]
    corners = corners_of_a_square(ra, dec, side_length)
    cutout_coords = [
        sky_box_to_image_box(corners, system)
        for system in systems
    ]
    if len(image_paths) > 0:
        shared_wcs = make_bounding_wcs(
            np.array(
                [
                    [corners[2][0], corners[1][1]],
                    [corners[0][0], corners[0][1]]
                ]
            )
        )
    else:
        shared_wcs = None
    watch.click(), stat.update()
    print(
        f"{watch.peek()} s; "
        f"{mb(round(first(stat.interval.values())))} MB"
    )
    binned_images = []
    for header, hdul, coords, system in zip(
        headers, hduls, cutout_coords, systems
    ):
        print(f"slicing data from {Path(hdul._filename).name}")
        cnt, flag, edge = [
            hdul[ix][coords[2]:coords[3] + 1, coords[0]:coords[1] + 1]
            for ix in (1, 2, 3)
        ]
        cnt = zero_flag_and_edge(cnt, flag, edge)
        watch.click(), stat.update()
        print(
            f"{watch.peek()} s; "
            f"{mb(round(first(stat.interval.values())))} MB"
        )
        watch.click()
        if len(image_paths) > 0:
            projection = project_slice_to_shared_wcs(
                cnt, system, shared_wcs, coords[0], coords[2]
            )
            binned_images.append(
                bin_projected_weights(
                    projection['x'],
                    projection['y'],
                    projection['weight'] / header['EXPTIME'],
                    wcs_imsz(shared_wcs)
                )
            )
        else:
            return cnt / header['EXPTIME'], system
    return np.sum(binned_images, axis=0), shared_wcs
