import gzip
import warnings
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import astropy.io.fits as pyfits
import fast_histogram as fh
import numpy as np
import scipy.sparse
from more_itertools import windowed
from pyarrow import parquet
from zstandard import ZstdCompressor

import gPhoton.constants as c
import gfcat.gfcat_utils as gfu
from gPhoton import MCUtils as mc
from gPhoton import __version__
from gPhoton._numbafied_pipe_components import between
from gPhoton._shared_memory_pipe_components import (
    reference_shared_memory_arrays,
    unlink_nested_block_dict,
    slice_into_memory,
    send_to_shared_memory,
)
from gPhoton.gphoton_utils import (
    table_values,
    NestingDict,
)


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


def _compute_exptime(timeslice, flagslice, band, trange):
    shutter = optimize_compute_shutter(timeslice, flagslice, trange)
    # Calculate deadtime
    model = {
        "NUV": [-0.000434730599193, 77.217817988],
        "FUV": [-0.000408075976406, 76.3000943221],
    }
    # TODO: THIS IS A CORRECTION THAT NEEDS TO BE
    #  IMPLEMENTED IN gPhoton!!!
    rawexpt = trange[1] - trange[0]
    rawexpt -= shutter
    if rawexpt == 0:
        return rawexpt
    gcr = len(timeslice) / rawexpt
    feeclkratio = 0.966
    refrate = model[band][1] / feeclkratio
    scr = model[band][0] * gcr + model[band][1]
    deadtime = 1 - scr / feeclkratio / refrate
    return rawexpt * (1.0 - deadtime)


def shared_compute_exptime(event_block_info, band, trange):
    """
    this retrieves exptime in presliced chunks from shared memory; thus
    no time indexing needs to happen inside this function.
    """
    _, event_dict = reference_shared_memory_arrays(event_block_info)
    events = event_dict["exposure"]
    return _compute_exptime(events[:, 0], events[:, 1], band, trange)


def unshared_compute_exptime(
    events: np.ndarray, band: str, trange: Sequence[int]
) -> float:
    times = events[:, 0]
    tix = np.where((times >= trange[0]) & (times < trange[1]))
    return _compute_exptime(times[tix], events[:, 1][tix], band, trange)


def select_on_detector(event_table, threshold=400):
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


def make_movies(
    depth: Optional[int],
    exposure_array: np.ndarray,
    map_ix_dict: dict,
    total_trange: tuple[int, int],
    imsz: Sequence[int, int],
    band: str,
    lil: bool = False,
    threads: Optional[int] = 4,
) -> tuple[dict[str, list[np.ndarray]], list, list]:
    """
    :param depth: framesize in seconds
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
    t0s = np.arange(total_trange[0], total_trange[1] + depth, depth)
    tranges = list(windowed(t0s, 2))
    exposure_directory = slice_exposure_into_memory(exposure_array, tranges)
    del exposure_array
    map_directory = NestingDict()
    # TODO: these _are_ always sorted by time, right? check
    for map_name in list(map_ix_dict.keys()):
        for frame_ix, trange in enumerate(tranges):
            frame_time_ix = between(map_ix_dict[map_name]["t"], *trange)[0]
            map_directory[frame_ix][map_name] = slice_into_memory(
                {k: v for k, v in map_ix_dict[map_name].items() if k != "t"},
                (frame_time_ix.min(), frame_time_ix.max()),
            )
        del map_ix_dict[map_name]
    del map_ix_dict
    pool = Pool(threads)
    results = {}
    for frame_ix, trange in enumerate(tranges):
        headline = f"Integrating frame {frame_ix + 1} of {len(tranges)}"
        # noinspection PyTypeChecker
        # results[frame_ix] = sm_compute_movie_frame(
        #     band,
        #     map_directory[frame_ix],
        #     exposure_directory[frame_ix],
        #     trange,
        #     imsz,
        #     lil,
        #     headline,
        # )
        results[frame_ix] = pool.apply_async(
            sm_compute_movie_frame,
            (
                band,
                map_directory[frame_ix],
                exposure_directory[frame_ix],
                trange,
                imsz,
                lil,
                headline,
            ),
        )
    pool.close()
    pool.join()
    results = {task: result.get() for task, result in results.items()}
    frame_indices = sorted(results.keys())
    movies = {"cnt": [], "flag": [], "edge": []}
    exptimes = []
    for frame_ix in frame_indices:
        movies["cnt"].append(results[frame_ix]["cnt"])
        movies["flag"].append(results[frame_ix]["flag"])
        movies["edge"].append(results[frame_ix]["edge"])
        exptimes.append(results[frame_ix]["exptime"])
        del results[frame_ix]
    return movies, tranges, exptimes


def sm_compute_movie_frame(
    band, map_block_info, exposure_block_info, trange, imsz, lil, headline
):
    mc.print_inline(headline)
    exptime = shared_compute_exptime(exposure_block_info, band, trange)
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
            {"exposure": exposure_array[between(times, trange[0], trange[1])]}
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


def open_compressed_stream(fn, compression):
    if Path(fn).exists():
        print(f"overwriting {fn}")
        Path(fn).unlink()
    if compression == "zstd":
        file_handle = open(fn, "wb+")
        compressor = ZstdCompressor(level=-1)
        writer = compressor.stream_writer(file_handle)
    else:
        file_handle = gzip.open(fn, "wb+", compresslevel=3)
        writer = file_handle
    return file_handle, writer


def write_movie(
    band,
    depth,
    eclipse,
    exptimes,
    movies,
    tranges,
    wcs,
    compression="gz",
    write_split=True,
):
    title = "-full" if depth is None else f"-{depth}s"

    # TODO, maybe: rewrite this to have to not assemble the primary hdu in
    #  order to make the header

    header = populate_fits_header(band, wcs, tranges, exptimes)

    if depth is None:
        movie_name = "full-depth image"
    else:
        movie_name = f"{depth}-second depth movie"
    if write_split is False:
        movie_fn = (
            f"test_data/e{eclipse}/e{eclipse}{title}" f".fits.{compression}"
        )
        file_handle, writer = open_compressed_stream(movie_fn, compression)
        print(f"writing {movie_name} to {movie_fn}")
    else:
        print("writing planes separately, pass write_split=False to not")
    for key in ["cnt", "flag", "edge"]:
        print(f"writing {key} map")
        if write_split is True:
            movie_fn = (
                f"test_data/e{eclipse}/e{eclipse}{title}-{key}.fits."
                f"{compression}"
            )
            file_handle, writer = open_compressed_stream(movie_fn, compression)
            print(f"writing to {movie_fn}")
        add_movie_to_fits_file(writer, movies[key], header)
        del movies[key]
        if write_split is True:
            writer.close()
    del movies
    writer.close()


def add_movie_to_fits_file(writer, movie, header):
    if isinstance(movie[0], scipy.sparse.spmatrix):
        pyfits.append(
            writer,
            np.stack([frame.toarray() for frame in movie]),
            header=header,
        )
    else:
        pyfits.append(writer, np.stack(movie), header=header)


def make_full_depth_image(
    exposure_array, map_ix_dict, total_trange, imsz, band="NUV"
):
    # TODO: feels weird to have the nominal range end up twice as long as
    #  the actual range, but the ending value can actually be arbitrarily
    #  far in the future, right?
    interval = total_trange[1] - total_trange[0]
    trange = np.arange(total_trange[0], total_trange[1] + interval, interval)
    exptime = unshared_compute_exptime(exposure_array, band, trange)
    images = {
        map_name: make_frame(
            map_ix_dict[map_name]["foc"],
            map_ix_dict[map_name]["weights"],
            imsz,
        )
        for map_name in ["cnt", "flag", "edge"]
    }
    return images, trange, exptime


def handle_movie_and_image_creation(
    eclipse,
    depth,
    band,
    lil=False,
    compression="gz",
    make_full=True,
    edge_threshold: int = 350,
    write_to_file=True,
):
    photonfile = f"test_data/e{eclipse}/e{eclipse}-nd.parquet"
    print(f"making images from {photonfile}")
    if write_to_file is True:
        print(
            "write_to_file=True passed, writing to disk and cleaning up "
            "in-memory values"
        )
    else:
        print("write_to_file=False passed, returning in-memory values")
    print("indexing data and making WCS solution")
    exposure_array, map_ix_dict, total_trange, wcs = prep_image_inputs(
        photonfile, edge_threshold
    )
    imsz = (
        int((wcs.wcs.crpix[1] - 0.5) * 2),
        int((wcs.wcs.crpix[0] - 0.5) * 2),
    )
    render_kwargs = {
        "exposure_array": exposure_array,
        "map_ix_dict": map_ix_dict,
        "total_trange": total_trange,
        "imsz": imsz,
        "band": band,
    }
    writer_kwargs = {
        "eclipse": eclipse,
        "wcs": wcs,
        "compression": compression,
        "band": band,
    }
    # TODO: this slightly less horrible format makes me unable to delete the
    #  exposure array inline. consider doing something about this.
    exptime, images, trange = handle_full_depth_image(
        depth, make_full, render_kwargs, write_to_file, writer_kwargs,
    )
    exptimes, movies, tranges = handle_movie(
        depth, lil, render_kwargs, write_to_file, writer_kwargs
    )
    if write_to_file is True:
        return
    output_dict = {"wcs": wcs}
    if movies is not None:
        output_dict["movie"] = {
            "cnt": movies["cnt"],
            "flag": movies["flag"],
            "edge": movies["edge"],
            "tranges": tranges,
            "exptimes": exptimes,
        }
        if images is not None:
            output_dict["image"] = {
                "cnt": images["cnt"],
                "flag": images["flag"],
                "edge": images["edge"],
                "trange": trange,
                "exptime": exptime,
            }
        return output_dict


def handle_movie(depth, lil, render_kwargs, write_to_file, writer_kwargs):
    if depth is not None:
        print(f"making {depth}-second movie")
        movies, tranges, exptimes = make_movies(
            depth=depth, lil=lil, **render_kwargs
        )
        del render_kwargs
        if write_to_file is True:
            write_movie(
                depth=depth,
                exptimes=exptimes,
                movies=movies,
                tranges=tranges,
                **writer_kwargs,
            )
    else:
        movies, tranges, exptimes = None, None, None
    return exptimes, movies, tranges


def handle_full_depth_image(
    depth, make_full_frame, render_kwargs, write_to_file, writer_kwargs
):
    if (depth is None) or (make_full_frame is True):
        print(f"making full-depth image")
        # special case: don't be careful about memory, just go for it.
        images, trange, exptime = make_full_depth_image(**render_kwargs)
        if write_to_file is True:
            write_movie(
                depth=None,
                exptimes=[exptime],
                movies=images,
                tranges=[trange],
                **writer_kwargs,
            )
            images = None
    else:
        images, trange, exptime = None, None, None
    return exptime, images, trange
