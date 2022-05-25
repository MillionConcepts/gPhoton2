from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import astropy.wcs
import fast_histogram as fh
import fitsio
import numpy as np

from gPhoton.coords.wcs import make_bounding_wcs, extract_wcs_keywords, \
    corners_of_a_square, sky_box_to_image_box
from gPhoton.io.fits_utils import pyfits_open_igzip, read_wcs_from_fits, \
    logged_fits_initializer
from gPhoton.pretty import notary, print_stats
from gPhoton.reference import (
    eclipse_to_paths, Stopwatch, Netstat, crudely_find_library
)


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


def cut_skybox(array_handles, target, side_length, wcs_object):
    corners = corners_of_a_square(target['ra'], target['dec'], side_length)
    coords = sky_box_to_image_box(corners, wcs_object)
    return [
        handle[coords[2]:coords[3] + 1, coords[0]:coords[1] + 1]
        for handle in array_handles
    ], corners, coords


def skybox_cuts_from_file(
    path, loader, targets, side_length, hdu_indices, wcs_object=None, verbose=0
):
    import astropy.wcs
    from gPhoton.coords.wcs import extract_wcs_keywords
    array_handles, header, log, stat = logged_fits_initializer(
        hdu_indices, loader, path, verbose
    )
    note = notary(log)
    # permit sharing wcs between pre-coregistered images
    if wcs_object is None:
        wcs_object = astropy.wcs.WCS(extract_wcs_keywords(header))
    cuts = []
    for target in targets:
        try:
            target_cuts, corners, coords = cut_skybox(
                array_handles, target, side_length, wcs_object
            )
        except ValueError as error:
            if "negative dimensions are not allowed" in str(error):
                print(f"coordinates out of bounds of image, skipping")
                continue
            raise
        cut_record = {
            "arrays": target_cuts, "coords": coords, "corners": corners
        } | target
        cuts.append(cut_record)
        note(
            f"got cuts at ra={round(target['ra'], 3)} "
            f"dec={round(target['dec'], 3)},c{path},{stat()}",
            loud=verbose > 1
        )
    note(
        f"got {len(targets)} cuts,{path},{stat(total=True)}", loud=verbose > 0
    )
    return cuts, wcs_object, header, log


def coadd_image_slices(
    image_slices: Sequence[Mapping], systems: Mapping[int, astropy.wcs.WCS]
):
    """
    operates on records output by skybox_cuts_from_file() or routines w/
    similar signatures.

    TODO: not fully integrated yet.
    """
    if len(image_slices) == 1:
        return image_slices[0]['array']
    corners = image_slices[0]['corners']
    shared_wcs = make_bounding_wcs(
        np.array(
            [[corners[2][0], corners[1][1]], [corners[0][0], corners[0][1]]]
        )
    )
    binned_images = []
    for image in image_slices:
        projection = project_slice_to_shared_wcs(
            image['array'],
            systems[image['eclipse']],
            shared_wcs,
            image['coords'][0],
            image['coords'][2]
        )
        binned_images.append(
            bin_projected_weights(
                projection['x'],
                projection['y'],
                projection['weight'],
                wcs_imsz(shared_wcs)
            )
        )
    return np.sum(binned_images, axis=0)
