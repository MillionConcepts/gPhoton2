"""
.. module:: _steps
   :synopsis: individual components of gPhoton movie- and image-making
   processes. generally should not be called on their own.
"""
from pathlib import Path
from typing import Mapping, Sequence, Literal
import warnings

import fitsio
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
from gPhoton.reference import PipeContext
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
    return frame.astype("f4")


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
    """
    retrieve count, flag, and map event arrays from shared memory and
    transform them into "maps" (image / movie frame) by taking their 2-D
    histograms; optionally return them sparsified
    """
    maps = [
        sm_make_map(block_directory, map_name, imsz)
        for map_name in ("cnt", "flag", "edge")
    ]
    if lil is True:
        return [scipy.sparse.coo_matrix(moviemap) for moviemap in maps]
    return maps


def sm_make_map(block_directory, map_name, imsz):
    """
    retrieve a count, flag, or map event array from shared memory and
    transform it into a "map" by taking its 2-D hstogram
    """
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
    # TODO: are we supposed to have SIP correction? we don't appear to.
    foc = wcs.sip_pix2foc(
        wcs.wcs_world2pix(parquet_to_ndarray(event_table, ["ra", "dec"]), 1), 1
    )
    return foc, wcs


def load_image_tables(
    photonfile: Pathlike,
) -> tuple[pyarrow.Table, np.ndarray]:
    """
    read a photonlist file produced by `photonpipe` from a raw6 telemetry file;
    return event and exposure tables appropriate for making images / movies
    and performing photometry
    """
    relevant_columns = ["ra", "dec", "response", "flags", "mask", "t", "detrad"]
    event_table = parquet.read_table(photonfile, columns=relevant_columns)
    # retain time and flag for every event for exposure time computations
    exposure_array = parquet_to_ndarray(event_table, ["t", "flags"])
    # but otherwise only deal with data actually on the 800x800 detector grid
    event_table = select_on_detector(event_table)
    return event_table, exposure_array


def populate_fits_header(band, wcs, tranges, exptimes):
    """
    create an astropy.io.fits.Header object containing our canonical
    metadata values
    """
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


def initialize_fits_file(array_path):
    if Path(array_path).exists():
        Path(array_path).unlink()
    stream = fitsio.FITS(array_path, "rw")
    stream.write_image(None)
    stream.close()


def write_fits_array(
    ctx: PipeContext, arraymap, wcs, is_movie=True, clean_up=True
):
    """
    convert an intermediate movie or image dictionary, perhaps previously
    used for photometry, into a FITS object; write it to disk; compress it
    using the best available gzip-compatible compression binary
    """
    if is_movie is False:
        array_name = "full-depth image"
    else:
        array_name = f"{ctx.depth}-second depth movie"
    outpaths = []
    # TODO: write names / descriptions into the headers
    if (ctx.burst is True) and (is_movie is True):
        # burst mode writes each frame as a separate file w/cnt, flag, and edge
        for frame in range(len(arraymap["cnt"])):
            start = arraymap['tranges'][frame][0] - ctx.start_time
            array_path = ctx(start=start)["movie"].replace(".gz", "")
            print(f"writing {array_name} frame {frame} to {array_path}")
            initialize_fits_file(array_path)
            for key in ["cnt", "flag", "edge"]:
                print(f"writing frame {frame} {key} map")
                header = populate_fits_header(
                    ctx.band, wcs, arraymap["tranges"], arraymap["exptimes"]
                )
                header['EXTNAME'] = key.upper()
                add_array_to_fits_file(
                    array_path,
                    arraymap[key][frame],
                    header,
                    ctx.compression,
                    **ctx.hdu_constructor_kwargs
                )
            outpaths.append(array_path)
    else:
        array_file = ctx["movie"] if is_movie is True else ctx["image"]
        array_path = Path(array_file.replace(".gz", ""))
        print(f"writing {array_name} to {array_path}")
        initialize_fits_file(array_path)
        for key in ["cnt", "flag", "edge"]:
            #print(f"writing {key} map")
            header = populate_fits_header(
                ctx.band, wcs, arraymap["tranges"], arraymap["exptimes"]
            )
            header['EXTNAME'] = key.upper()
            add_array_to_fits_file(
                array_path,
                arraymap[key],
                header,
                ctx.compression,
                **ctx.hdu_constructor_kwargs
            )
            if clean_up:
                del arraymap[key]
        outpaths = [array_path]
    if clean_up:
        del arraymap
    if ctx.compression != "gzip":
        return
    for path in outpaths:
        gzip_path = Path(f"{path}.gz")
        if gzip_path.exists():
            print(f"overwriting {gzip_path}")
            gzip_path.unlink()
        print(f"gzipping {path}")
        # try various gzip commands in order of perceived goodness
        for gzipper, gzip_command in (
            ("igzip", [path, "-T 4", "--rm"]),
            ("libdeflate_gzip", [path]),
            ("gzip", [path]),
        ):
            try:
                getattr(sh, gzipper)(*gzip_command)
                break
            except sh.CommandNotFound:
                continue


def add_array_to_fits_file(
    fits_path,
    array,
    header,
    compression_type: Literal["none", "gzip", "rice"] = "none",
    **fitsio_write_kwargs
):
    if compression_type not in ('none', 'gzip', 'rice'):
        raise ValueError(f"{compression_type} is not supported.")
    if isinstance(array, np.ndarray):
        data = array
    elif array.__class__.__module__.startswith('scipy.sparse'):
        data = array.toarray()
    elif array[0].__class__.__module__.startswith('scipy.sparse'):
        data = np.stack([frame.toarray() for frame in array])
    else:
        raise ValueError("I don't understand this so-called 'array'.")
    # this pipeline supports monolithic gzipping after file construction,
    # not gzipping individual HDUs.
    fits_stream = fitsio.FITS(fits_path, "rw")
    try:
        if compression_type in ("none", "gzip"):
            fits_stream.write(
                data,
                extname=header['EXTNAME'],
                header=dict(header),
                **fitsio_write_kwargs
            )
            return
        if 'tile_size' in fitsio_write_kwargs:
            tile_size = fitsio_write_kwargs.pop('tile_size')
        else:
            tile_size = (1, 100, 100) if len(data.shape) == 3 else (100, 100)
        if len(tile_size) > len(data.shape):
            tile_size = tile_size[:len(data.shape)]
        if len(tile_size) < len(data.shape):
            tile_size = [
                1 for _ in range(len(data.shape) - len(tile_size))
            ] + list(tile_size)
        if 'qlevel' in fitsio_write_kwargs:
            qlevel = fitsio_write_kwargs.pop('qlevel')
        else:
            qlevel = 10
        fits_stream.write(
            data,
            header=dict(header),
            extname=header['EXTNAME'],
            compress='RICE',
            tile_dims=tile_size,
            qlevel=qlevel,
            qmethod=2,
            **fitsio_write_kwargs
        )
    finally:
        fits_stream.close()
