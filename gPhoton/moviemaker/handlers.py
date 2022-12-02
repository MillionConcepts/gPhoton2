"""top-level handler module for gPhoton.moviemaker"""

from multiprocessing import Pool
from typing import Optional, Union

from dustgoggles.structures import NestingDict
from more_itertools import windowed
import numpy as np

from gPhoton.moviemaker._steps import (
    predict_sparse_movie_memory,
    slice_exposure_into_memory,
    slice_frame_into_memory,
    sm_compute_movie_frame,
    unshared_compute_exptime,
    make_frame,
    write_fits_array,
    predict_movie_memory,
    prep_image_inputs,
)
from gPhoton.types import Pathlike, GalexBand


def make_movies(
    depth: Optional[int],
    exposure_array: np.ndarray,
    map_ix_dict: dict,
    total_trange: tuple[int, int],
    imsz: tuple[int, int],
    band: str,
    lil: bool = False,
    threads: Optional[int] = 4,
    maxsize: Optional[int] = None,
    fixed_start_time: Optional[float] = None,
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
    :param fixed_start_time: externally-defined time for start of first frame,
    primarily intended to help coregister NUV and FUV frames
    """
    if fixed_start_time is not None:
        total_trange = (fixed_start_time, total_trange[1])
    t0s = np.arange(total_trange[0], total_trange[1] + depth, depth)
    tranges = list(windowed(t0s, 2))
    # TODO, maybe: at very short depths, slicing arrays into memory becomes a
    #  meaningful single-core-speed-dependent bottleneck. overhead of
    #  distributing across processes may not be worth it anyway.
    #  also, jamming the entirety of the arrays into memory and indexing as
    #  needed _is_ an option if rigorous thread safety is practiced, although
    #  this will significantly increase the memory pressure of this portion
    #  of the execute_pipeline.
    if maxsize is not None:
        # TODO: this is jury-rigged and ignores unsparsified cases etc. etc.
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
            print(failure_string + "; halting execute_pipeline")
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


def make_burst_movies(
    depth: Optional[int],
    exposure_array: np.ndarray,
    map_ix_dict: dict,
    total_trange: tuple[int, int],
    imsz: tuple[int, int],
    band: str,
    lil: bool = False,
    threads: Optional[int] = 4,
    maxsize: Optional[int] = None,
    fixed_start_time: Optional[float] = None,
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
    :param fixed_start_time: externally-defined time for start of first frame,
    primarily intended to help coregister NUV and FUV frames
    """
    if fixed_start_time is not None:
        total_trange = (fixed_start_time, total_trange[1])
    t0s = np.arange(total_trange[0], total_trange[1] + depth, depth)
    tranges = list(windowed(t0s, 2))
    # TODO, maybe: at very short depths, slicing arrays into memory becomes a
    #  meaningful single-core-speed-dependent bottleneck. overhead of
    #  distributing across processes may not be worth it anyway.
    #  also, jamming the entirety of the arrays into memory and indexing as
    #  needed _is_ an option if rigorous thread safety is practiced, although
    #  this will significantly increase the memory pressure of this portion
    #  of the execute_pipeline.
    if maxsize is not None:
        # TODO: this is jury-rigged and ignores unsparsified cases etc. etc.
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
            print(failure_string + "; halting execute_pipeline")
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


def make_full_depth_image(
    exposure_array, map_ix_dict, total_trange, imsz, maxsize=None, band="NUV"
) -> tuple[str, dict]:
    if maxsize is not None:
        # nominally, peak memory usage will be ~9 -- 4 + 1 + 4 -- just before
        # the final map is cast from float32 to uint8.
        memory = predict_movie_memory(imsz, n_frames=1, nbytes=9)
        if memory > maxsize:
            failure_string = (
                f"failure: {round(memory/(1024**3), 2)} GB needed to make "
                f"image > size threshold {maxsize}"
            )
            print(failure_string + "; halting execute_pipeline")
            return failure_string, {}
    # TODO: this weird arithmetic cartwheel doesn't seem _wrong_, but it can't
    #  be _necessary_, right?
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


def write_moviemaker_results(
    results,
    filenames,
    depth,
    band,
    leg,
    write,
    maxsize,
    stopwatch,
    compression,
    burst,
    hdu_constructor_kwargs,
    **unused_options
):
    if write["image"] and (results["image_dict"] != {}):
        write_fits_array(
            band,
            None,
            filenames["images"][leg].replace(".gz", ""),
            results["image_dict"],
            clean_up=True,
            wcs=results["wcs"],
            compression=compression,
            hdu_constructor_kwargs=hdu_constructor_kwargs
        )
    del results["image_dict"]
    stopwatch.click()
    if write["movie"] and (results["movie_dict"] != {}):
        # we don't check size of the image, because if we can handle the image
        # in memory, we can write it, but we handle the movies frame by frame
        # earlier in the execute_pipeline, so that doesn't hold true for them.
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
            filenames["movies"][leg].replace(".gz", ""),
            results["movie_dict"],
            clean_up=True,
            wcs=results["wcs"],
            compression=compression,
            burst=burst,
            hdu_constructor_kwargs=hdu_constructor_kwargs
        )
        stopwatch.click()
    return "successful"


def create_images_and_movies(
    photonfile: Pathlike,
    depth: int,
    band: GalexBand,
    lil=False,
    threads=None,
    maxsize=None,
    fixed_start_time: Optional[int] = None,
    edge_threshold: int = 350,
    min_exptime: Optional[float] = None,
    **_unused_options
) -> Union[dict, str]:
    print(f"making images from {photonfile}")
    print("indexing data and making WCS solution")
    movie_dict, image_dict, status = {}, {}, "started"
    exposure_array, map_ix_dict, total_trange, wcs = prep_image_inputs(
        photonfile, edge_threshold
    )
    imsz = (
        int((wcs.wcs.crpix[1] - 0.5) * 2),
        int((wcs.wcs.crpix[0] - 0.5) * 2)
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
    if (min_exptime is not None) and (image_dict["exptimes"][0] < min_exptime):
        return {
            "wcs": wcs,
            "movie_dict": {},
            "image_dict": image_dict,
            "status": (
                f"exptime {round(image_dict['exptimes'][0])} "
                f"< min_exptime {min_exptime}"
            )
        }
    if (depth is not None) and status.startswith("success"):
        print(f"making {depth}-second depth movies")
        burst_mode = True # temporary, need to add to pipeline
        if burst_mode:
            status, movie_dict = make_movies(
                depth=depth,
                lil=lil,
                threads=threads,
                fixed_start_time=fixed_start_time,
                **render_kwargs,
            )
        elif burst_mode:
            print("using burst mode to save individual movie frames.")
            status, movie_dict = make_burst_movies(
                depth=depth,
                lil=lil,
                threads=threads,
                fixed_start_time=fixed_start_time,
                **render_kwargs,
            )
    return {
        "wcs": wcs,
        "movie_dict": movie_dict,
        "image_dict": image_dict,
        "status": status,
    }
