from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Union
import warnings

import astropy.io.fits as pyfits
import fast_histogram as fh
import numpy as np
import scipy.sparse
import sh
from more_itertools import windowed
from pyarrow import parquet

from gPhoton import MCUtils as mc
from gPhoton import __version__
from gPhoton._numbafied_pipe_components import between, slice_between
from gPhoton._shared_memory_pipe_components import (
    reference_shared_memory_arrays,
    unlink_nested_block_dict,
    slice_into_memory,
    send_to_shared_memory,
)
from gPhoton.gphoton_utils import make_wcs_from_radec
from gPhoton.pipeline_utils import (
    table_values,
    NestingDict,
)


def booleanize_for_fits(array):
    # not actually casting to bool because FITS does not have a boolean
    # data type and astropy will not cast boolean arrays to unsigned
    # integers, which is the closest thing FITS does have.
    mask = array.astype(bool)
    return np.ones(shape=array.shape, dtype=np.uint8) * mask


def make_frame(foc, weights, imsz, booleanize=False):
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
    ix = np.where(flagslice == 0)
    gcr = len(timeslice[ix]) / rawexpt
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


def make_movies(
    depth: Optional[int],
    exposure_array: np.ndarray,
    map_ix_dict: dict,
    total_trange: tuple[int, int],
    imsz: Sequence[int, int],
    band: str,
    lil: bool = False,
    threads: Optional[int] = 4,
    maxsize: Optional[int] = None,
) -> tuple[str, dict]:
    """
    :param depth: framesize in seconds
    :param exposure_array: t and flags, _including_ off-detector, for exptime
    :param map_ix_dict: cnt, edge, and mask indices for weights, t, and foc
    :param total_trange: (time minimum, time maximum) for on-detector events
    :param imsz: size of each frame, as given by upstream WCS object
    :param band: "FUV" or "NUV"
    :param lil: sparsify matrices to reduce memory footprint, at compute cost
    :param threads: how many threads to use to calculate frames
    :param maxsize: terminate if predicted size (in bytes) of cntmap > maxsize
    """
    # TODO, maybe: at very short depths, slicing arrays into memory becomes a
    #  meaningful single-core-speed-dependent bottleneck. overhead of
    #  distributing across processes may not be worth it anyway.
    #  also, jamming the entirety of the arrays into memory and indexing as
    #  needed _is_ an option if rigorous thread safety is practiced, although
    #  this will significantly increase the memory pressure of this portion
    #  of the pipeline.
    t0s = np.arange(total_trange[0], total_trange[1] + depth, depth)
    tranges = list(windowed(t0s, 2))
    if maxsize is not None:
        # we're ignoring most of the per-frame overhead at this point because
        # it is probably (?) trivial for large arrays.
        # sparse = 0.08 if lil else 1
        memory = predict_sparse_movie_memory(
            imsz, n_frames=len(tranges), threads=threads
        )
        # add a single-frame 'penalty' for unsparsified full-depth image
        # 10 = sum of element sizes across planes
        memory += imsz[0] * imsz[1] * 10
        if memory > maxsize:
            failure_string = (
                f"{round(memory / (1024 ** 3), 2)} GB needed to make movie "
                f"> size threshold {maxsize}"
            )
            print(failure_string + "; halting pipeline")
            return failure_string, {}
    print("slicing exposure array into memory")
    exposure_directory = slice_exposure_into_memory(exposure_array, tranges)
    del exposure_array
    map_directory = NestingDict()
    for map_name in list(map_ix_dict.keys()):
        for frame_ix, trange in enumerate(tranges):
            # 0-count exposure times have 'None' entries assigned in
            # slice_exposure_into_memory
            print(
                f"slicing {map_name} data {frame_ix + 1} of {len(tranges)} "
                f"into memory"
            )
            map_directory[frame_ix][map_name] = slice_frame_into_memory(
                exposure_directory, map_ix_dict, map_name, frame_ix, trange
            )
        del map_ix_dict[map_name]
    del map_ix_dict
    if threads is None:
        pool = None
    else:
        pool = Pool(threads)
    results = {}
    for frame_ix, trange in enumerate(tranges):
        headline = f"Integrating frame {frame_ix + 1} of {len(tranges)}"
        frame_params = (
            band,
            map_directory[frame_ix],
            exposure_directory[frame_ix],
            trange,
            imsz,
            lil,
            headline,
        )
        if pool is None:
            results[frame_ix] = sm_compute_movie_frame(*frame_params)
        else:
            results[frame_ix] = pool.apply_async(
                sm_compute_movie_frame, frame_params
            )
    if pool is not None:
        pool.close()
        pool.join()
        results = {task: result.get() for task, result in results.items()}
    frame_indices = sorted(results.keys())
    movies = {"cnt": [], "flag": [], "edge": []}
    exptimes = []
    for frame_ix in frame_indices:
        for map_name in ("cnt", "flag", "edge"):
            movies[map_name].append(results[frame_ix][map_name])
        exptimes.append(results[frame_ix]["exptime"])
        del results[frame_ix]
    return (
        "successfully made movies",
        {"tranges": tranges, "exptimes": exptimes} | movies,
    )


def sm_compute_movie_frame(
    band, map_block_info, exposure_block_info, trange, imsz, lil, headline
):
    mc.print_inline(headline)
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
    wcs = make_wcs_from_radec(table_values(event_table, ["ra", "dec"]))
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
    with the current pipeline configuration. this will be something of an
    undercount due to handling costs.
    """
    return imsz[0] * imsz[1] * n_frames * nbytes


def predict_sparse_movie_memory(
    imsz, n_frames, threads, nbytes=17
):
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
    # and also there's some amount of overhead from frames
    slice_cost = n_frames * nbytes * (40 * 1024**2)
    return base_cost + slice_cost


def make_full_depth_image(
    exposure_array, map_ix_dict, total_trange, imsz, maxsize=None, band="NUV"
) -> tuple[str, dict]:
    if maxsize is not None:
        # peak memory usage will be ~17 -- 8 + 1 + 8, before the final
        # map is cast from float64 to uint8.
        memory = predict_movie_memory(imsz, n_frames=1, nbytes=17)
        if memory > maxsize:
            failure_string = (
                f"failure: {round(memory/(1024**3), 2)} GB needed to make "
                f"image > size threshold {maxsize}"
            )
            print(failure_string + "; halting pipeline")
            return failure_string, {}
    interval = total_trange[1] - total_trange[0]
    trange = np.arange(total_trange[0], total_trange[1] + interval, interval)
    exptime = unshared_compute_exptime(exposure_array, band, trange)
    output_dict = {"tranges": [trange], "exptimes": [exptime]}
    for map_name in ("cnt", "flag", "edge"):
        output_dict[map_name] = make_frame(
            map_ix_dict[map_name]["foc"],
            map_ix_dict[map_name]["weights"],
            imsz,
            booleanize=map_name in ("flag", "edge"),
        )
    return "successfully made image", output_dict


def handle_movie_and_image_creation(
    photonfile,
    depth,
    band,
    lil=False,
    threads=None,
    maxsize=None,
    edge_threshold: int = 350,
) -> Union[dict, str]:
    print(f"making images from {photonfile}")
    print("indexing data and making WCS solution")
    movie_dict, image_dict, status = {}, {}, "started"
    exposure_array, map_ix_dict, total_trange, wcs = prep_image_inputs(
        photonfile, edge_threshold
    )
    imsz = (
        int((wcs.wcs.crpix[1] - 0.5) * 2),
        int((wcs.wcs.crpix[0] - 0.5) * 2),
    )
    print(f"image size: {imsz}")
    render_kwargs = {
        "exposure_array": exposure_array,
        "map_ix_dict": map_ix_dict,
        "total_trange": total_trange,
        "imsz": imsz,
        "maxsize": maxsize,
        "band": band,
    }

    print(f"making full-depth image")
    # don't be careful about memory wrt sparsification, just go for it
    status, image_dict = make_full_depth_image(**render_kwargs)
    if (depth is not None) and status.startswith("success"):
        print(f"making {depth}-second depth movies")
        status, movie_dict = make_movies(
            depth=depth, lil=lil, threads=threads, **render_kwargs
        )
    return {
        "wcs": wcs,
        "movie_dict": movie_dict,
        "image_dict": image_dict,
        "status": status,
    }


def write_fits(
    results,
    filenames,
    depth,
    band,
    write,
    maxsize,
    stopwatch,
):
    if write["image"] and (results["image_dict"] != {}):
        write_fits_array(
            band,
            None,
            filenames["image"].replace(".gz", ""),
            results["image_dict"],
            clean_up=True,
            wcs=results["wcs"],
        )
    del results["image_dict"]
    stopwatch.click()
    if write["movie"] and (results["movie_dict"] != {}):
        # we don't check size of the image, because if we can handle the image
        # in memory, we can write it, but we handle the movies frame by frame
        # earlier in the pipeline, so that doesn't hold true for them.
        if maxsize is not None:
            imsz = results["movie_dict"]["cnt"][0].shape
            n_frames = len(results["movie_dict"]["cnt"])
            memory = predict_movie_memory(imsz, n_frames)
            if memory > maxsize:
                failure_string = (
                    f"{round(memory / (1024 ** 3), 2)} GB needed to write "
                    f"movie > size threshold {maxsize}"
                )
                print(failure_string + "; not writing")
                return failure_string
        write_fits_array(
            band,
            depth,
            filenames["movie"].replace(".gz", ""),
            results["movie_dict"],
            clean_up=True,
            wcs=results["wcs"],
        )
        stopwatch.click()
    return "successful"
