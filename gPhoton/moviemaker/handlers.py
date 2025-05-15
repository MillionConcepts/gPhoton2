"""top-level handler module for gPhoton.moviemaker"""

from multiprocessing import Pool
from typing import Any, Optional, Union

from dustgoggles.structures import NestingDict
from more_itertools import windowed
import numpy as np

from gPhoton.moviemaker._steps import (
    slice_exposure_into_memory,
    slice_frame_into_memory,
    sm_compute_movie_frame,
    unshared_compute_exptime,
    make_frame,
    make_mask_frame,
    write_fits_array,
    prep_image_inputs,
)
from gPhoton.reference import PipeContext
from gPhoton.types import Pathlike


def make_movies(
    ctx: PipeContext,
    exposure_array: np.ndarray,
    map_ix_dict: dict,
    total_trange: tuple[int, int],
    imsz: tuple[int, int],
    fixed_start_time: Optional[int] = None,
) -> tuple[str, dict]:
    """
    :param ctx: PipeContext options handler
    :param exposure_array: t and flags, _including_ off-detector, for exptime
    :param map_ix_dict: cnt and mask indices for weights, t, and foc
    :param total_trange: (time minimum, time maximum) for on-detector events
    :param imsz: size of each frame, as given by upstream WCS object
    :param fixed_start_time: externally-defined time for start of first frame,
    primarily intended to help coregister NUV and FUV frames
    """
    if fixed_start_time is not None:
        total_trange = (fixed_start_time, total_trange[1])
    assert ctx.depth is not None
    t0s = np.arange(total_trange[0], total_trange[1] + ctx.depth, ctx.depth)
    tranges = list(windowed(t0s, 2))
    # TODO, maybe: at very short depths, slicing arrays into memory becomes a
    #  meaningful single-core-speed-dependent bottleneck. overhead of
    #  distributing across processes may not be worth it anyway.
    #  also, jamming the entirety of the arrays into memory and indexing as
    #  needed _is_ an option if rigorous thread safety is practiced, although
    #  this will significantly increase the memory pressure of this portion
    #  of the execute_pipeline.
    exposure_directory = slice_exposure_into_memory(exposure_array, tranges)
    del exposure_array
    map_directory = NestingDict()
    for map_name in list(map_ix_dict.keys()):
        for frame_ix, trange in enumerate(tranges):
            # 0-count exposure times have 'None' entries assigned in
            # slice_exposure_into_memory
            map_directory[frame_ix][map_name] = slice_frame_into_memory(
                exposure_directory, map_ix_dict, map_name, frame_ix, trange
            )
        del map_ix_dict[map_name]
    del map_ix_dict
    if ctx.threads is None:
        pool = None
    else:
        pool = Pool(ctx.threads)
    results = {}
    for frame_ix, trange in enumerate(tranges):
        headline = f"Integrating frame {frame_ix + 1} of {len(tranges)}"
        frame_params = (
            ctx.band,
            map_directory[frame_ix],
            exposure_directory[frame_ix],
            trange,
            imsz,
            ctx.lil,
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
    movies: dict[str, list[Any]] = {"cnt": [], "flag": []}
    exptimes = []
    for frame_ix in frame_indices:
        for map_name in ("cnt", "flag"):
            movies[map_name].append(results[frame_ix][map_name])
        exptimes.append(results[frame_ix]["exptime"])
        del results[frame_ix]
    return (
        "successfully made movies",
        {"tranges": tranges, "exptimes": exptimes} | movies,
    )


def make_full_depth_image(
    exposure_array, map_ix_dict, total_trange, imsz, band="NUV"
): # -> tuple[str, dict]:  we're not ready to typecheck this function yet
    interval = total_trange[1] - total_trange[0]
    trange = np.arange(total_trange[0], total_trange[1] + interval, interval)
    exptime = unshared_compute_exptime(exposure_array, band, trange)
    output_dict = {"tranges": [trange], "exptimes": [exptime]}
    output_dict["cnt"] = make_frame(
            map_ix_dict["cnt"]["foc"],
            map_ix_dict["cnt"]["weights"],
            imsz
    )
    output_dict["flag"] = make_mask_frame(
        map_ix_dict['flag']["foc"],
        map_ix_dict['flag']["weights"],
        imsz
        )
    return "successfully made image", output_dict


def write_moviemaker_results(results, ctx):
    if ctx.write["image"] and (results["image_dict"] != {}):
        write_fits_array(ctx, results["image_dict"], results["wcs"], False)
    del results["image_dict"]
    ctx.watch.click()
    if ctx.write["movie"] and (results["movie_dict"] != {}):
        write_fits_array(ctx, results["movie_dict"], results["wcs"])
        ctx.watch.click()
    return "successful"


def create_images_and_movies(
    ctx: PipeContext,
    photonfile: Pathlike,
    fixed_start_time: Optional[int] = None
) -> Union[dict, str]:
    print(f"making images from {photonfile}")
    print("indexing data and making WCS solution")
    movie_dict = {}
    status = "started"
    exposure_array, map_ix_dict, total_trange, wcs, photons = prep_image_inputs(
        photonfile,
        ctx
    )
    imsz = (
        int((wcs.wcs.crpix[1] - 0.5) * 2), int((wcs.wcs.crpix[0] - 0.5) * 2)
    )
    # to check that the dimensions of the image are at least 1x1
    # for images where everything is flagged it could get to this point
    # and the imsz is 0,0 and image making returns []. could maybe be a better fix
    # catching this earlier in the pipeline maybe?
    if sum(imsz) <= 1:
        raise ValueError("image size is less than 1x1, "
                         "photonlist is likely all flagged.")

    render_kwargs = {
        "exposure_array": exposure_array,
        "map_ix_dict": map_ix_dict,
        "total_trange": total_trange,
        "imsz": imsz,
    }
    print(f"making full-depth image")
    # don't be careful about memory wrt sparsification, just go for it
    status, image_dict = make_full_depth_image(**render_kwargs)
    if (
        (ctx.min_exptime is not None)
        and (image_dict["exptimes"][0] < ctx.min_exptime)
    ):
        return {
            "wcs": wcs,
            "movie_dict": {},
            "image_dict": image_dict,
            "photon_count": photons,
            "status": (
                f"exptime {round(image_dict['exptimes'][0])} "
                f"< min_exptime {ctx.min_exptime}"
            )
        }
    if (ctx.depth is not None) and status.startswith("success"):
        print(f"making {ctx.depth}-second depth movies")
        status, movie_dict = make_movies(
            ctx, fixed_start_time=fixed_start_time, **render_kwargs,
        )
    return {
        "wcs": wcs,
        "movie_dict": movie_dict,
        "image_dict": image_dict,
        "photon_count": photons,
        "status": status,
    }
