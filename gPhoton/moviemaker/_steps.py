"""
.. module:: _steps
   :synopsis: individual components of gPhoton movie- and image-making
   processes. generally should not be called on their own.
"""
from pathlib import Path
from typing import Mapping, Sequence
import warnings

import pyarrow
from astropy.io import fits as pyfits
from dustgoggles.structures import NestingDict
import fast_histogram as fh
import numpy as np
from pyarrow import parquet
import scipy.sparse
import sh

from gPhoton import __version__
from gPhoton.calibrate import compute_exptime
from gPhoton.coords.wcs import make_bounding_wcs
from gPhoton.parquet_utils import parquet_to_ndarray
from gPhoton.pretty import print_inline
from gPhoton.sharing import (
    reference_shared_memory_arrays,
    slice_into_memory,
    unlink_nested_block_dict,
    send_to_shared_memory,
)
from gPhoton.types import Pathlike
from gPhoton.vorpal import between, slice_between


def booleanize_for_fits(array: np.ndarray) -> np.ndarray:
    """
    many things in FITS arrays are too big. it would be nice if FITS had a
    1-bit data type. It does not. Furthermore, astropy.io.fits will not even
    cast boolean arrays to unsigned integers, which is the closest thing FITS
    does have. This function does that, after fake-booleanizing them.
    :param array: numpy array to faux-booleanize
    :return: faux-booleanized array
    """
    mask = array.astype(bool)
    return np.ones(shape=array.shape, dtype=np.uint8) * mask


def make_frame(
    foc: np.ndarray,
    weights: np.ndarray,
    imsz: tuple[int, int],
    booleanize: bool = False,
) -> np.ndarray:
    """

    :param foc: 2-column array containing positions in detector space
    :param weights: 1-D array containing counts at each position
    :param imsz: x/y dimensions of output image
    :param booleanize: reduce this to "boolean" (uint8 valued as 0 or 1)?
    :return: ndarray containing single image made from 2D histogram of inputs
    """
    frame = fh.histogram2d(
        foc[:, 1] - 0.5,
        foc[:, 0] - 0.5,
        bins=imsz,
        range=([[0, imsz[0]], [0, imsz[1]]]),
        weights=weights,
    )
    if booleanize:
        return booleanize_for_fits(frame)
    return frame


def shared_compute_exptime(
    event_block_info: Mapping, band: str, trange: tuple[float, float]
) -> float:
    """
    unlike unshared_compute_exptime(),
    this retrieves exptime in presliced chunks from shared memory; thus
    no time indexing needs to happen inside this function.
    """
    _, event_dict = reference_shared_memory_arrays(event_block_info)
    events = event_dict["exposure"]
    return compute_exptime(events[:, 0], events[:, 1], band, trange)


def unshared_compute_exptime(
    events: np.ndarray, band: str, trange: Sequence[float]
) -> float:
    times = events[:, 0]
    tix = np.where((times >= trange[0]) & (times < trange[1]))
    return compute_exptime(times[tix], events[:, 1][tix], band, trange)


def select_on_detector(
    event_table: pyarrow.Table, threshold: int = 400
) -> pyarrow.Table:
    """
    select events "on" the detector
    :param event_table: pyarrow Table that contains a detector radius column
    :param threshold: how many pixels away from center still counts as "on"?
    :return: Table consisting of rows of event_table "on" detector
    """
    detrad = event_table["detrad"].to_numpy()
    return event_table.take(
        # TODO: is isfinite() necessary?
        np.where(np.isfinite(detrad) & (detrad < threshold))[0]
    )


def prep_image_inputs(photonfile, edge_threshold):
    event_table, exposure_array = load_image_tables(photonfile)
    foc, wcs = generate_wcs_components(event_table)
    with warnings.catch_warnings():
        # don't bother us about divide-by-zero errors
        warnings.simplefilter("ignore")
        weights = 1.0 / event_table["response"].to_numpy()
    mask_ix = np.where(event_table["mask"].to_numpy())
    edge_ix = np.where(event_table["detrad"].to_numpy() > edge_threshold)
    t = event_table["t"].to_numpy()
    map_ix_dict = generate_indexed_values(edge_ix, foc, mask_ix, t, weights)
    total_trange = (t.min(), t.max())
    return exposure_array, map_ix_dict, total_trange, wcs


def generate_indexed_values(edge_ix, foc, mask_ix, t, weights):
    indexed = NestingDict()
    for value, value_name in zip((t, foc, weights), ("t", "foc", "weights")):
        for map_ix, map_name in zip(
            (edge_ix, mask_ix, slice(None)), ("edge", "flag", "cnt")
        ):
            indexed[map_name][value_name] = value[map_ix]
    return indexed


def slice_frame_into_memory(
    exposure_directory, map_ix_dict, map_name, frame_ix, trange
):
    # 0-count exposure times have 'None' entries assigned in
    # slice_exposure_into_memory
    if exposure_directory[frame_ix] is None:
        return
    frame_time_ix = between(map_ix_dict[map_name]["t"], *trange)[0]
    if len(frame_time_ix) == 0:
        return None
    try:
        return slice_into_memory(
            {k: v for k, v in map_ix_dict[map_name].items() if k != "t"},
            (frame_time_ix.min(), frame_time_ix.max()),
        )
    except ValueError as error:
        # case in which there are no events for this map in this
        # time range
        if (
            str(error).lower()
            == "'size' must be a positive number different from zero"
        ):
            return None
        raise


def sm_compute_movie_frame(
    band, map_block_info, exposure_block_info, trange, imsz, lil, headline
):
    print_inline(headline)
    if exposure_block_info is None:
        exptime = 0
        cntmap, edgemap, flagmap = zero_frame(imsz, lil)
    else:
        exptime = shared_compute_exptime(exposure_block_info, band, trange)
        # todo: make this cleaner...?
        expblock, _ = reference_shared_memory_arrays(
            exposure_block_info, fetch=False
        )
        expblock["exposure"].unlink()
        expblock["exposure"].close()
        # noinspection PyTypeChecker
        cntmap, flagmap, edgemap = sm_make_maps(map_block_info, imsz, lil)
        unlink_nested_block_dict(map_block_info)
        # TODO: slice this into shared memory to reduce serialization cost?
        #  not worth doing if they're sparse
    return {
        "cnt": cntmap,
        "edge": edgemap,
        "flag": flagmap,
        "exptime": exptime,
    }


def slice_exposure_into_memory(exposure_array, tranges):
    exposure_directory = {}
    times = exposure_array[:, 0]
    for frame_ix, trange in enumerate(tranges):
        exposure = slice_between(exposure_array, times, trange[0], trange[1])
        if len(exposure) > 0:
            exposure_directory[frame_ix] = send_to_shared_memory(
                {"exposure": exposure}
            )
        else:
            exposure_directory[frame_ix] = None
    return exposure_directory


def zero_frame(imsz, lil=False):
    maps = (
        np.zeros(imsz),
        np.zeros(imsz, dtype="uint8"),
        np.zeros(imsz, dtype="uint8"),
    )
    if lil is True:
        return [scipy.sparse.coo_matrix(moviemap) for moviemap in maps]
    return maps


def sm_make_maps(block_directory, imsz, lil=False):
    maps = [
        sm_make_map(block_directory, map_name, imsz)
        for map_name in ("cnt", "flag", "edge")
    ]
    if lil is True:
        return [scipy.sparse.coo_matrix(moviemap) for moviemap in maps]
    return maps


def sm_make_map(block_directory, map_name, imsz):
    if block_directory[map_name] is None:
        dtype = np.uint8 if map_name in ("edge", "flag") else np.float64
        return np.zeros(imsz, dtype)
    _, map_arrays = reference_shared_memory_arrays(block_directory[map_name])
    return make_frame(
        map_arrays["foc"],
        map_arrays["weights"],
        imsz,
        booleanize=map_name in ("edge", "flag"),
    )


def generate_wcs_components(event_table):
    wcs = make_bounding_wcs(parquet_to_ndarray(event_table, ["ra", "dec"]))
    # This is a bottleneck, so only do it once.
    # TODO: do we actually want these 1-indexed?
    foc = wcs.sip_pix2foc(
        wcs.wcs_world2pix(parquet_to_ndarray(event_table, ["ra", "dec"]), 1), 1
    )
    return foc, wcs


def load_image_tables(
    photonfile: Pathlike,
) -> tuple[pyarrow.Table, np.ndarray]:
    event_table = parquet.read_table(
        photonfile,
        columns=[
            "ra",
            "dec",
            "t",
            "response",
            "flags",
            "mask",
            "detrad"
        ],
    )
    # retain time and flag for every event for exposure time computations
    exposure_array = parquet_to_ndarray(event_table, ["t", "flags"])
    # but otherwise only deal with data actually on the 800x800 detector grid
    event_table = select_on_detector(event_table)
    return event_table, exposure_array


def populate_fits_header(band, wcs, tranges, exptimes):
    header = pyfits.Header()
    header["CDELT1"], header["CDELT2"] = wcs.wcs.cdelt
    header["CTYPE1"], header["CTYPE2"] = wcs.wcs.ctype
    header["CRPIX1"], header["CRPIX2"] = wcs.wcs.crpix
    header["CRVAL1"], header["CRVAL2"] = wcs.wcs.crval
    header["EQUINOX"], header["EPOCH"] = 2000.0, 2000.0
    header["BAND"] = 1 if band == "NUV" else 2
    header["VERSION"] = "v{v}".format(v=__version__)
    header["EXPSTART"] = np.array(tranges).min()
    header["EXPEND"] = np.array(tranges).max()
    header["EXPTIME"] = sum(t1 - t0 for (t0, t1) in tranges)
    header["N_FRAME"] = len(tranges)
    for i, trange in enumerate(tranges):
        header["T0_{i}".format(i=i)] = trange[0]
        header["T1_{i}".format(i=i)] = trange[1]
        header["EXPT_{i}".format(i=i)] = exptimes[i]
    return header


def write_fits_array(
    band,
    depth,
    moviefile,
    movie_dict,
    wcs,
    compress=True,
    clean_up=False,
):
    # TODO, maybe: rewrite this to have to not assemble the primary hdu in
    #  order to make the header
    header = populate_fits_header(
        band, wcs, movie_dict["tranges"], movie_dict["exptimes"]
    )
    if depth is None:
        movie_name = "full-depth image"
    else:
        movie_name = f"{depth}-second depth movie"
    movie_path = Path(moviefile)
    if movie_path.exists():
        print(f"overwriting {movie_path} with {movie_name}")
        movie_path.unlink()
    else:
        print(f"writing {movie_name} to {movie_path}")
    # TODO: write names / descriptions into the headers
    for key in ["cnt", "flag", "edge"]:
        print(f"writing {key} map")
        add_movie_to_fits_file(movie_path, movie_dict[key], header)
        if clean_up:
            del movie_dict[key]
    if clean_up:
        del movie_dict
    if compress is not True:
        return
    gzip_path = Path(f"{movie_path}.gz")
    if gzip_path.exists():
        print(f"overwriting {gzip_path}")
        gzip_path.unlink()
    print(f"gzipping {movie_path}")
    # try various gzip commands in order of perceived goodness
    for gzipper, gzip_command in (
        ("igzip", [movie_path, "-T 4", "--rm"]),
        ("libdeflate_gzip", [movie_path]),
        ("gzip", [movie_path]),
    ):
        try:
            getattr(sh, gzipper)(*gzip_command)
            break
        except sh.CommandNotFound:
            continue


def add_movie_to_fits_file(writer, movie, header):
    if isinstance(movie[0], scipy.sparse.spmatrix):
        pyfits.append(
            writer,
            np.stack([frame.toarray() for frame in movie]),
            header=header,
        )
    else:
        pyfits.append(writer, np.stack(movie), header=header)


def predict_movie_memory(imsz, n_frames, nbytes=8):
    """
    predict memory size in bytes of movie during write. nbytes is equal to the
    per-element size of _only_ the largest plane, which should be the cntmap
    with the current execute_pipeline configuration. this will be something of an
    undercount due to handling costs.
    """
    return imsz[0] * imsz[1] * n_frames * nbytes


def predict_sparse_movie_memory(imsz, n_frames, threads, nbytes=17):
    """
    nbytes here is equal to the sum of the per-element sizes of all movie
    planes -- cnt / flag / edge, in the worst-case where one has not yet
    been reduced to uint8.
    sparsification means that, for large arrays, the process will usually
    get cheaper as it goes on unless there are a truly huge number of frames,
    so we're sort of ignoring framesize in that calculation for now.
    """
    if threads is None:
        threads = 1
    threads = min(threads, n_frames)
    framesize = imsz[0] * imsz[1] * nbytes
    # we need to be able to hold one full frame in memory for each thread
    base_cost = framesize * threads
    # and also there's some small amount of overhead from frames
    slice_cost = n_frames * (15 * 1024 ** 2)
    return base_cost + slice_cost
