import gc
import gzip
import os
import warnings
from collections.abc import Sequence
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Optional, Union

import astropy.io.fits as pyfits
import astropy.wcs.wcs
import fast_histogram as fh
import numpy as np
from more_itertools import windowed
from pyarrow import parquet
import scipy.sparse
from zstandard import ZstdCompressor

import gPhoton.constants as c
import gfcat.gfcat_utils as gfu
from gPhoton import MCUtils as mc
from gPhoton import __version__
from gPhoton._shared_memory_pipe_components import (
    reference_shared_memory_arrays,
    unlink_nested_block_dict,
    slice_into_memory,
    send_to_shared_memory,
)
from gPhoton.gphoton_utils import (
    table_values,
    where_between,
    NestingDict,
)


# @profile
def optimize_wcs(radec):
    real_ra = radec[:, 0][np.isfinite(radec[:, 0])]
    real_dec = radec[:, 1][np.isfinite(radec[:, 1])]
    ra_range = real_ra.min(), real_ra.max()
    dec_range = real_dec.min(), real_dec.max()
    center_skypos = (np.mean(ra_range), np.mean(dec_range))
    imsz = (
        int(np.ceil((dec_range[1] - dec_range[0]) / c.DEGPERPIXEL)),
        int(np.ceil((ra_range[1] - ra_range[0]) / c.DEGPERPIXEL)),
    )
    # imsz = (3200, 3200)
    return gfu.make_wcs(center_skypos, imsz=imsz, pixsz=c.DEGPERPIXEL)


def make_frame(foc, weights, imsz):
    frame = fh.histogram2d(
        foc[:, 1] - 0.5,
        foc[:, 0] - 0.5,
        bins=imsz,
        range=([[0, imsz[0]], [0, imsz[1]]]),
        weights=weights,
    )
    return frame


def optimize_compute_shutter(timeslice, flagslice, trange, shutgap=0.05):
    ix = np.where(flagslice == 0)
    t = np.hstack([trange[0], np.unique(timeslice[ix]), +trange[1]])
    ix = np.where(t[1:] - t[:-1] >= shutgap)
    shutter = np.array(t[1:] - t[:-1])[ix].sum()
    return shutter


def sm_compute_exptime(event_block_info, band, trange):
    """
    this retrieves exptime in presliced chunks from shared memory; thus
    no time indexing needs to happen inside this function.
    """
    _, event_dict = reference_shared_memory_arrays(event_block_info)
    events = event_dict["exposure"]
    rawexpt = trange[1] - trange[0]
    shutter = optimize_compute_shutter(
        events[:, 0], flagslice=events[:, 1], trange=trange
    )
    # Calculate deadtime
    model = {
        "NUV": [-0.000434730599193, 77.217817988],
        "FUV": [-0.000408075976406, 76.3000943221],
    }
    # TODO: THIS IS A CORRECTION THAT NEEDS TO BE
    #  IMPLEMENTED IN gPhoton!!!
    rawexpt -= shutter

    if rawexpt == 0:
        return rawexpt
    gcr = len(events[:, 0]) / rawexpt
    feeclkratio = 0.966
    refrate = model[band][1] / feeclkratio
    scr = model[band][0] * gcr + model[band][1]
    deadtime = 1 - scr / feeclkratio / refrate
    return rawexpt * (1.0 - deadtime)


def optimize_compute_exptime(
    events: np.ndarray, band: str, trange: Sequence[int]
) -> float:
    rawexpt = trange[1] - trange[0]
    times = events[:, 0]
    tix = np.where((times >= trange[0]) & (times < trange[1]))
    timeslice = times[tix]
    shutter = optimize_compute_shutter(
        timeslice, flagslice=events[:, 1][tix], trange=trange
    )

    # Calculate deadtime
    model = {
        "NUV": [-0.000434730599193, 77.217817988],
        "FUV": [-0.000408075976406, 76.3000943221],
    }

    # TODO: THIS IS A CORRECTION THAT NEEDS TO BE
    #  IMPLEMENTED IN gPhoton!!!
    rawexpt -= shutter

    if rawexpt == 0:
        return rawexpt
    gcr = len(timeslice) / rawexpt
    feeclkratio = 0.966
    refrate = model[band][1] / feeclkratio
    scr = model[band][0] * gcr + model[band][1]
    deadtime = 1 - scr / feeclkratio / refrate
    return rawexpt * (1.0 - deadtime)


def select_on_detector(event_table, threshold=400):
    detrad = event_table["detrad"].to_numpy()
    return event_table.take(
        # TODO: is isfinite() necessary?
        np.where(np.isfinite(detrad) & (detrad < threshold))[0]
    )


# @profile
def make_images(
    photonfile: Union[str, os.PathLike],
    depth: Sequence[Union[int, None]],
    band: str = "NUV",
    edge_threshold: int = 350,
    lil: bool = False,
):
    print("making exposure tables & WCS solution")
    exposure_array, indexed, trange, wcs = prep_image_inputs(
        photonfile, edge_threshold
    )
    imsz = (
        int((wcs.wcs.crpix[1] - 0.5) * 2),
        int((wcs.wcs.crpix[0] - 0.5) * 2),
    )
    # TODO: write each framesize out mid_loop
    movies, tranges, exptimes = {}, [], []
    for framesize in depth:
        print(f"making {framesize}-second depth movie")
        movies, tranges, exptimes = make_frames_at_depth(
            framesize, exposure_array, indexed, trange, imsz, band, lil
        )
        mc.print_inline("")
    return movies, wcs, tranges, exptimes

    # TODO: Write the images.


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
    indexed = generate_indexed_values(edge_ix, foc, mask_ix, t, weights)
    trange = (
        np.min(event_table["t"].to_numpy()),
        np.max(event_table["t"].to_numpy()),
    )
    return exposure_array, indexed, trange, wcs


def generate_indexed_values(edge_ix, foc, mask_ix, t, weights):
    indexed = NestingDict()
    for value, value_name in zip((t, foc, weights), ("t", "foc", "weights")):
        for map_ix, map_name in zip(
            (edge_ix, mask_ix, slice(None)), ("edge", "flag", "cnt")
        ):
            indexed[map_name][value_name] = value[map_ix]
    return indexed


def make_frames_at_depth(
    framesize: Optional[int],
    exposure_array: np.ndarray,
    map_ix_dict: dict,
    total_trange: tuple[int, int],
    imsz: Sequence[int, int],
    band: str,
    lil: bool = False,
    threads: Optional[int] = 4,
) -> tuple[dict[str, list[np.ndarray]], list, list]:
    """
    :param framesize: framesize in seconds; None for full range
    :param exposure_array: t and flags, _including_ off-detector, for exptime
    :param map_ix_dict: cnt, edge, and mask indices for weights, t, and foc
    :param total_trange: (time minimum, time maximum) for on-detector events
    :param imsz: size of each frame, as given by upstream WCS object
    :param band: "FUV" or "NUV"
    :param lil: sparsify matrices to reduce memory footprint, at compute cost
    :param threads: how many threads to use to calculate frames
    """
    if threads is None:
        # TODO: write this part
        raise NotImplementedError("oops, I didn't write this part yet")
    interval = total_trange[1] - total_trange[0]
    t0s = np.arange(
        total_trange[0],
        total_trange[1] + framesize
        if framesize is not None
        else total_trange[1] + interval,
        framesize
        if framesize is not None
        else total_trange[1] - total_trange[0],
    )
    tranges = list(windowed(t0s, 2))
    exposure_directory = slice_exposure_into_memory(exposure_array, tranges)
    del exposure_array
    map_directory = NestingDict()
    # TODO: these _are_ always sorted by time, right? check
    for map_name in list(map_ix_dict.keys()):
        for frame_ix, trange in enumerate(tranges):
            frame_time_ix = where_between(map_ix_dict[map_name]["t"], *trange)
            map_directory[frame_ix][map_name] = slice_into_memory(
                {k: v for k, v in map_ix_dict[map_name].items() if k != "t"},
                (frame_time_ix.min(), frame_time_ix.max()),
            )
        del map_ix_dict[map_name]
    del map_ix_dict
    # pool = Pool(threads)
    results = {}
    for frame_ix, trange in enumerate(tranges):
        headline = f"Integrating frame {frame_ix + 1} of {len(tranges)}"
        # noinspection PyTypeChecker
        results[frame_ix] = sm_compute_movie_frames(
            band,
            map_directory[frame_ix],
            exposure_directory[frame_ix],
            trange,
            imsz,
            lil,
            headline
        )
        # time.sleep(10)
        # raise
        # results[frame_ix] = pool.apply_async(
        #     sm_compute_movie_frames,
        #     (
        #         band,
        #         map_directory[frame_ix],
        #         exposure_directory[frame_ix],
        #         trange,
        #         imsz,
        #         lil,
        #         headline,
        #     ),
        # )
    # pool.close()

    mc.print_inline("... gathering results ...")
    # pool.join()
    # results = {task: result.get() for task, result in results.items()}
    frame_indices = sorted(results.keys())
    movies = {"cnt": [], "flag": [], "edge": []}
    exptimes = []
    for frame_ix in frame_indices:
        movies["cnt"].append(results[frame_ix]["cnt"])
        movies["flag"].append(results[frame_ix]["cnt"])
        movies["edge"].append(results[frame_ix]["cnt"])
        exptimes.append(results[frame_ix]["exptime"])
        del results[frame_ix]
    return movies, tranges, exptimes


def sm_compute_movie_frames(
    band, map_block_info, exposure_block_info, trange, imsz, lil, headline
):
    mc.print_inline(headline)
    exptime = sm_compute_exptime(exposure_block_info, band, trange)
    # todo: make this cleaner...?
    expblock, _ = reference_shared_memory_arrays(
        exposure_block_info, fetch=False
    )
    expblock["exposure"].unlink()
    expblock["exposure"].close()
    # noinspection PyTypeChecker
    cntmap, edgemap, flagmap = sm_make_maps(map_block_info, imsz, lil)
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
        exposure_directory[frame_ix] = send_to_shared_memory(
            {
                "exposure": exposure_array[
                    np.where((times >= trange[0]) & (times < trange[1]))
                ]
            }
        )
    return exposure_directory


def sm_make_maps(block_directory, imsz, lil=False):
    maps = [
        sm_make_map(block_directory, map_name, imsz)
        for map_name in ("cnt", "flag", "edge")
    ]
    if lil is True:
        return [scipy.sparse.lil_matrix(moviemap) for moviemap in maps]
    return maps


def sm_make_map(block_directory, map_name, imsz):
    _, map_arrays = reference_shared_memory_arrays(block_directory[map_name])
    return make_frame(map_arrays["foc"], map_arrays["weights"], imsz)


# def make_maps(indexed, trange, wcs):
#     t0, t1 = trange
#     t, foc, weights = [indexed[n]["det"] for n in ("t", "foc", "weights")]
#     cnt_tix = where_between(t, t0, t1)
#     cntmap = make_frame(foc[cnt_tix], weights[cnt_tix], wcs)
#     flag_tix = where_between(indexed["t"]["mask"], t0, t1)
#     flagmap = make_frame(
#         indexed["foc"]["mask"][flag_tix],
#         indexed["weights"]["mask"][flag_tix],
#         wcs,
#     )
#     edge_tix = where_between(indexed["t"]["edge"], t0, t1)
#     edgemap = make_frame(
#         indexed["foc"]["edge"][edge_tix],
#         indexed["weights"]["edge"][edge_tix],
#         wcs,
#     )
#     return cntmap, edgemap, flagmap
#


def generate_wcs_components(event_table):
    wcs = optimize_wcs(table_values(event_table, ["ra", "dec"]))
    # This is a bottleneck, so only do it once.
    foc = wcs.sip_pix2foc(
        wcs.wcs_world2pix(table_values(event_table, ["ra", "dec"]), 1), 1
    )
    return foc, wcs


def load_image_tables(photonfile):
    event_table = parquet.read_table(
        photonfile,
        columns=[
            "ra",
            "dec",
            "t",
            "response",
            "flags",
            "mask",
            "detrad",
        ],
    )
    # Only deal with data actually on the 800x800 detector grid
    exposure_array = table_values(event_table, ["t", "flags"])
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


def write_movie(
    band, depth, eclipse, exptimes, movies, tranges, wcs, compression="gz"
):
    # TODO: nicer depth filenaming in the inner loop, garbage hack
    # TODO: actually write all the frames
    if depth == [None]:
        frame_title = "-full"
    else:
        frame_title = f"-{depth[0]}s"
    movie_fn = f"test_data/e{eclipse}/e{eclipse}{frame_title}" \
               f"-cnt.fits.{compression}"
    if Path(movie_fn).exists():
        print(f"overwriting {movie_fn}")
        Path(movie_fn).unlink()
    if compression == "zstd":
        file_handle = open(movie_fn, "wb+")
        compressor = ZstdCompressor(level=-1)
        writer = compressor.stream_writer(file_handle)
    else:
        file_handle = gzip.open(movie_fn, "wb+", compresslevel=3)
        writer = file_handle
    # TODO, maybe: rewrite this to have to not assemble the primary hdu in
    #  order to make the header
    print(f"writing {depth[0]}-second depth movie to {movie_fn}")
    header = populate_fits_header(band, wcs, tranges, exptimes)
    for key in ["cnt", "flag", "edge"]:
        print(f"writing {key} map to file")
        add_movie_to_fits_file(writer, movies[key], header)
        del movies[key]
    del movies
    writer.close()


def add_movie_to_fits_file(writer, movie, header):
    if isinstance(movie[0], scipy.sparse.spmatrix):
        pyfits.append(
            writer,
            np.stack([frame.toarray() for frame in movie]),
            header=header
        )
    else:
        pyfits.append(writer, np.stack(movie), header=header)


def make_movies(eclipse, depths, band, lil=False, compression="gz"):
    photonfile = f"test_data/e{eclipse}/e{eclipse}-nd.parquet"
    print(f"making movies at {depths}-second depths from {photonfile}")
    movies, wcs, tranges, exptimes = make_images(
        photonfile, depth=depths, lil=lil
    )
    write_movie(
        band, depths, eclipse, exptimes, movies, tranges, wcs, compression
    )
