import warnings
from functools import reduce
from itertools import product
from multiprocessing import Pool
from operator import or_
from pathlib import Path
from typing import Mapping, Sequence, Union, Optional, Literal, Callable

import astropy.wcs
import fast_histogram as fh
import numpy as np
from dustgoggles.structures import listify

from gPhoton.coords.wcs import (
    make_bounding_wcs,
    corners_of_a_rectangle,
    sky_box_to_image_box,
)
from gPhoton.io.fits_utils import (
    pyfits_open_igzip,
    read_wcs_from_fits,
    AgnosticHDUL,
    AgnosticHDU,
)
from gPhoton.reference import eclipse_to_paths
from gPhoton.types import Pathlike


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
    system: Optional[astropy.wcs.WCS] = None,
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
        "exptime": header["EXPTIME"],
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


def get_full_frame_coadd_layer(gphoton_fits, shared_wcs, hdu_offset=0):
    projection = project_to_shared_wcs(gphoton_fits, shared_wcs, hdu_offset)
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


def _check_oob(bounds, shape):
    if not reduce(
        or_,
        (
            bounds[2] < 0,
            bounds[3] > shape[0],
            bounds[0] < 0,
            bounds[1] > shape[1],
        ),
    ):
        return
    warnings.warn(
        f"{bounds} larger than array ({shape}) output will be clipped"
    )


def cut_skybox(hdus, ra, dec, ra_x=None, dec_x=None, system=None):
    """
    assumes all hdus are spatially coregistered. if not, call the function
    once for each cluster of spatially-coregistered hdus.
    """
    hdus = listify(hdus)
    if system is None:
        if not isinstance(hdus[0], AgnosticHDU):
            system = AgnosticHDU(hdus[0]).wcs_
        else:
            system = hdus[0].wcs_
    corners = corners_of_a_rectangle(ra, dec, ra_x, dec_x)
    coords = sky_box_to_image_box(corners, system)
    arrays = []
    for hdu in hdus:
        _check_oob(coords, hdu.shape)
        arrays.append(
            hdu.data[coords[2] : coords[3] + 1, coords[0] : coords[1] + 1]
        )
    return {
        "arrays": arrays,
        "corners": corners,
        "coords": coords,
        "ra": ra,
        "dec": dec,
    }


def cut_skybox_from_file(
    path: Pathlike,
    ra: float,
    dec: float,
    ra_x: float = None,
    dec_x: float = None,
    system: Optional[astropy.wcs.WCS] = None,
    loader: Optional[Callable] = None,
    hdu_indices: tuple[int] = (0,),
    **_,
):
    """
    assumes all hdus are spatially coregistered. if not, call the function
    multiple times.
    """
    if loader is None:
        import fitsio

        loader = fitsio.FITS
    hdul = AgnosticHDUL(loader(path))
    try:
        return cut_skybox(
            [hdul[ix] for ix in hdu_indices], ra, dec, ra_x, dec_x, system
        )
    except ValueError as ve:
        if ("NaN" in str(ve)) or ("negative dimensions" in str(ve)):
            return
        raise


# TODO, maybe: granular logging here -- maybe because Netstat is basically
#  useless on this level if cuts are parallelized, and diagnostics would
#  probably be better done on fake data using other workflows
def cut_skyboxes(plans, **kwargs):
    if kwargs.get("threads") is not None:
        cuts = _cut_skyboxes_threaded([p | kwargs for p in plans])
    else:
        cuts = _cut_skyboxes_unthreaded([p | kwargs for p in plans])
    # None results are cuts that edged out of the WCS or image bounds
    return list(filter(None, cuts))


def _cut_skyboxes_unthreaded(cut_plans):
    return [cut_skybox_from_file(**plan) for plan in cut_plans]


def _cut_skyboxes_threaded(cut_plans):
    pool = Pool(cut_plans[0]["threads"])
    cuts = []
    for plan in cut_plans:
        cuts.append(pool.apply_async(cut_skybox_from_file, kwds=plan))
    pool.close()
    pool.join()
    return [cut_result.get() for cut_result in cuts]


def coadd_image_slices(image_slices: Sequence[Mapping]):
    """
    operates on records output by get_galex_cutouts() or routines w/
    similar signatures.

    TODO: not fully integrated yet.
    """
    if len(image_slices) == 1:
        return (
            zero_flag_and_edge(*image_slices[0]["arrays"]),
            image_slices[0]["system"],
        )
    corners = image_slices[0]["corners"]
    shared_wcs = make_bounding_wcs(
        np.array(
            [[corners[2][0], corners[1][1]], [corners[0][0], corners[0][1]]]
        )
    )
    binned_images = []
    for image in image_slices:
        projection = project_slice_to_shared_wcs(
            zero_flag_and_edge(*image["arrays"]),
            image["system"],
            shared_wcs,
            image["coords"][0],
            image["coords"][2],
        )
        binned_images.append(
            bin_projected_weights(
                projection["x"],
                projection["y"],
                projection["weight"],
                wcs_imsz(shared_wcs),
            )
            / image["exptime"]
        )
    return np.sum(binned_images, axis=0), shared_wcs
