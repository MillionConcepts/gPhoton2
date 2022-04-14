from itertools import product

from astropy.wcs import wcs
import fast_histogram as fh
import numpy as np

from gPhoton.coords.wcs import make_bounding_wcs
from gPhoton.io.fits_utils import pyfits_open_igzip, read_wcs_from_fits
from gPhoton.reference import eclipse_to_paths


def get_image_fns(*eclipses, band="NUV", root="test_data"):
    return [
        eclipse_to_paths(eclipse, root)[band]["image"] for eclipse in eclipses
    ]


def corner_ra_dec(wcs_system):
    imsz = wcs_imsz(wcs_system)
    ymax, xmax = imsz[1], imsz[0]
    return wcs_system.pixel_to_world_values(
        (0, ymax, ymax, 0), (0, xmax, 0, xmax)
    )


def bounding_corners(corners):
    extremes = []
    for pred, coord in product((np.max, np.min), np.arange(len(corners[0]))):
        local_extrema = map(pred, [corner[coord] for corner in corners])
        extremes.append(pred(tuple(local_extrema)))
    return extremes


def make_shared_wcs(wcs_sequence):
    corners = tuple(map(corner_ra_dec, wcs_sequence))
    ra_min, dec_min, ra_max, dec_max = map(
        np.float32, bounding_corners(corners)
    )
    return make_bounding_wcs(np.array([[ra_min, dec_min], [ra_max, dec_max]]))


def project_to_shared_wcs(gphoton_fits, shared_wcs, nonzero=True):
    cnt, flag, edge = [gphoton_fits[ix].data for ix in range(3)]
    cnt[~np.isfinite(cnt)] = 0
    cnt[np.nonzero(flag)] = 0
    cnt[np.nonzero(edge)] = 0
    if nonzero is True:
        y_ix, x_ix = np.nonzero(cnt)
    else:
        indices = np.indices((cnt.shape[0], cnt.shape[1]), dtype=np.int16)
        y_ix, x_ix = indices[0].ravel(), indices[1].ravel()
    system = wcs.WCS(gphoton_fits[0].header)
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


# note: this contains an assumption about the origin of the wcs
def wcs_imsz(system: wcs.WCS):
    return (
        int((system.wcs.crpix[1] - 0.5) * 2),
        int((system.wcs.crpix[0] - 0.5) * 2),
    )


def get_coadd_slice(gphoton_fits, shared_wcs):
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
        coadd += get_coadd_slice(pyfits_open_igzip(image_file), shared_wcs)
    return coadd
