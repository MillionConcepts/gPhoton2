from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Mapping, Sequence, Union, Optional, Literal

import astropy.wcs
import fast_histogram as fh
import numpy as np

from gPhoton.coords.wcs import (
    make_bounding_wcs, corners_of_a_square, sky_box_to_image_box
)
from gPhoton.io.fits_utils import pyfits_open_igzip, read_wcs_from_fits
from gPhoton.reference import (
    eclipse_to_paths, crudely_find_library
)


def get_image_fns(*eclipses, band="NUV", root="test_data"):
    """
    Construct list of filenames for a series of eclipse images.
    Args:
        *eclipses: eclipses for which to construct filenames
        band: GALEX band of images
        root: root directory that contains per-eclipse subdirectories

    Returns:

    """
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


# TODO: this version is compatible with RICE compression, but is relatively
#  inefficient. needs to be juiced up.
def project_to_shared_wcs(
    fits_path: Union[str, Path],
    shared_wcs: astropy.wcs.WCS,
    hdu_offset: Literal[0, 1] = 0,
    nonzero: bool = True,
    system: Optional[astropy.wcs.WCS] = None
):
    """
    fits_path: path to fits file
    shared_wcs: WCS object
    hdu_offset: number of HDUs to skip at the beginning of the file. this will
        be 1 for RICE-compressed GALEX images and 0 otherwise.
    nonzero: sparsify returned weights?
    system: precomputed WCS object for this image (only for optimization)
    """
    import fitsio

    hdul = fitsio.FITS(fits_path)
    cnt, flag, edge = [hdul[ix + hdu_offset].read() for ix in range(3)]
    cnt = zero_flag_and_edge(cnt, flag, edge)
    if nonzero is True:
        y_ix, x_ix = np.nonzero(cnt)
    else:
        indices = np.indices((cnt.shape[0], cnt.shape[1]), dtype=np.int16)
        y_ix, x_ix = indices[0].ravel(), indices[1].ravel()
    header = hdul[1].read_header()
    if system is None:
        system = astropy.wcs.WCS(header)
    ra_input, dec_input = system.pixel_to_world_values(x_ix, y_ix)
    x_shared, y_shared = shared_wcs.wcs_world2pix(ra_input, dec_input, 1)
    return {
        "x": x_shared,
        "y": y_shared,
        "weight": cnt[y_ix, x_ix],
        "exptime": header["EXPTIME"]
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


def cut_skybox(loader, target, hdu_indices, side_length):
    hdul = loader(target['path'])
    library = crudely_find_library(loader)
    array_handles = [
        hdul[hdu_ix]
        if library == "fitsio"
        else hdul[hdu_ix].data
        for hdu_ix in hdu_indices
    ]
    corners = corners_of_a_square(target['ra'], target['dec'], side_length)
    try:
        coords = sky_box_to_image_box(corners, target['wcs'])
        return {
            "arrays": [
                handle[coords[2]:coords[3] + 1, coords[0]:coords[1] + 1]
                for handle in array_handles
            ],
            "corners": corners,
            "coords": coords
        } | target
    except ValueError as ve:
        if ("NaN" in str(ve)) or ("negative dimensions" in str(ve)):
            return 
        raise


# TODO, maybe: granular logging here -- maybe because Netstat is basically
#  useless on this level if cuts are parallelized, and diagnostics would
#  probably be better done on fake data using other workflows
def cut_skyboxes(plans, threads, cut_kwargs):
    pool = Pool(threads) if threads is not None else None
    cuts = []
    for cut_plan in plans:
        if pool is None:
            cuts.append(cut_skybox(target=cut_plan, **cut_kwargs))
        else:
            cuts.append(
                pool.apply_async(
                    cut_skybox, kwds={"target": cut_plan} | cut_kwargs
                )
            )
    if pool is not None:
        pool.close()
        pool.join()
        cuts = [cut_result.get() for cut_result in cuts]
    # None results are cuts that edged out of the WCS or image bounds
    return list(filter(None, cuts))


def coadd_image_slices(image_slices: Sequence[Mapping]):
    """
    operates on records output by get_galex_cutouts() or routines w/
    similar signatures.

    TODO: not fully integrated yet.
    """
    if len(image_slices) == 1:
        return (
            zero_flag_and_edge(*image_slices[0]['arrays']), 
            image_slices[0]['wcs']
        )
    corners = image_slices[0]['corners']
    shared_wcs = make_bounding_wcs(
        np.array(
            [[corners[2][0], corners[1][1]], [corners[0][0], corners[0][1]]]
        )
    )
    binned_images = []
    for image in image_slices:
        projection = project_slice_to_shared_wcs(
            zero_flag_and_edge(*image['arrays']),
            image['wcs'],
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
    return np.sum(binned_images, axis=0), shared_wcs
