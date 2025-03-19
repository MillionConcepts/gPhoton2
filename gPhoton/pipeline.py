"""
primary handler module for the 'full' gPhoton pipeline. can:
1. retrieve data from MAST or another specified location
2. create calibrated photonlists
3. create full-depth FITS images and movies of any number of specified depths
4. generate lightcurves and photonlists using detected or specified sources
These steps can also be performed separately using methods from the
gPhoton.io, gPhoton.photonpipe, gPhoton.moviemaker, and gPhoton.lightcurve
modules. This module is intended to perform them with optimized transitions
and endpoints / output suitable for remote automation.
"""
import re
import shutil
import warnings
from pathlib import Path
from time import time
from types import MappingProxyType
from typing import Literal
from collections.abc import Sequence, Mapping

from cytoolz import identity, keyfilter
from more_itertools import chunked

import gPhoton.reference
from gPhoton.reference import PipeContext, check_eclipse
from gPhoton.types import GalexBand

# oh no! divide by zero! i am very distracting!
# (we love dividing by zero. dividing by zero is cool.)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def get_photonlists(ctx: PipeContext, raise_errors: bool = True):
    """
    Args:
        ctx: pipeline context management object
        raise_errors: if we encounter errors during pipeline execution,
            shall we raise an exception (True) or return structured text
            describing that error and its context (False)? Typically this
            should be True for ad-hoc local execution (and especially for
            debugging), and False for managed remote execution.
    """
    # TODO, maybe: parameter to run only certain legs
    create_local_paths(ctx)
    if ctx.recreate is False:
        photonpaths = find_photonfiles(ctx)
        if all(path.exists() for path in photonpaths):
            print(
                f"using existing photon list(s): "
                f"{[str(path) for path in photonpaths]}"
            )
            return photonpaths
    raw6path = _look_for_raw6(ctx)
    if not raw6path.exists():
        print("couldn't find raw6 file.")
        if raise_errors is True:
            raise OSError("couldn't find raw6 file.")
        return "return code: couldn't find raw6 file."
    from gPhoton.photonpipe import execute_photonpipe

    try:
        return execute_photonpipe(ctx, str(raw6path))
    except ValueError as value_error:
        if raise_errors:
            raise
        return parse_photonpipe_error(value_error)


def find_photonfiles(context: PipeContext):
    photonpaths = []
    if context.remote is not None:
        r_photpaths = [
            Path(leg(remote=True)['photonfile'])
            for leg in context.explode_legs()
        ]
        if all(path.exists() for path in r_photpaths):
            print(
                f"making temp local copies of photon file(s) from remote: "
                f"{[p.name for p in r_photpaths]}"
            )
            photonpaths = [
                Path(shutil.copy(path, context.temp_path()))
                for path in r_photpaths
            ]
    if len(photonpaths) == 0:
        photonpaths = [
            Path(leg()["photonfile"]) for leg in context.explode_legs()
        ]
    return photonpaths


def execute_photometry_only(ctx: PipeContext):
    errors = []
    for leg_step in ctx.explode_legs():
        loaded_arrays = load_moviemaker_results(leg_step, ctx.lil)
        # this is an error code
        if isinstance(loaded_arrays, str):
            errors.append(loaded_arrays)
            continue
        from gPhoton.lightcurve import make_lightcurves

        result = make_lightcurves(loaded_arrays, leg_step)
        if result != "successful":
            errors.append(result)
    print(
        f"{round(time() - ctx.watch.start_time, 2)} seconds for execution"
    )
    if len(errors) > 0:
        return "return code: " + ";".join(errors)
    return "return code: successful"


# TODO, maybe: add a check somewhere for: do we have information regarding
#  this eclipse at all?
def execute_pipeline(
    eclipse: int,
    band: GalexBand,
    depth: int | None = None,
    threads: int | None = None,
    local_root: str = "test_data",
    remote_root: str | None = None,
    download: bool = True,
    recreate: bool = False,
    verbose: int = 2,
    source_catalog_file: str | None = None,
    write: Mapping = MappingProxyType({"image": True, "movie": True}),
    aperture_sizes: Sequence[float] = (12.8,),
    lil: bool = True,
    coregister_lightcurves: bool = False,
    stop_after: Literal["photonpipe", "moviemaker"] | None = None,
    compression: Literal["none", "gzip", "rice"] = "gzip",
    hdu_constructor_kwargs: Mapping = MappingProxyType({}),
    min_exptime: float | None = None,
    photometry_only: bool = False,
    burst: bool = False,
    chunksz: int = 1000000,
    share_memory: bool | None = None,
    extended_photonlist: bool = False,
    extended_flagging: bool = False,
    aspect: Literal['aspect', 'aspect2'] = 'aspect',
    override_eclipse_limits: bool = False,
    suffix: str | None = None,
    aspect_dir: None | str | Path = None,
    ftype: str = "csv",
) -> str:
    """
    Args:
        eclipse: GALEX eclipse number to process
        band: GALEX band to process: 'FUV' or 'NUV'?
        depth: how many seconds of events to use when integrating each
            movie frame. in a sense: inverse FPS. None means "make a
            full-depth image only".
        threads: how many threads to use for parallel processing. Passing None
            turns off parallel processing entirely. Passing 1 will process in a
            single parallel thread (not recommended except for test
            purposes). Increasing thread count increases execution speed but
            also increases memory pressure, particularly for movie creation.
            Multithreading currently works best on Linux and is not
            guaranteed to be at all performant on other operating systems.
        local_root: Root of directory tree (on the local system, or at least
            somewhere with write permissions), to write and/or look for files
        remote_root: Root of another directory tree (perhaps a mounted S3
            bucket or something) to check for preexisting raw6 and
            photonlist files. Unlike local_root, there is no assumption that
            write access is available to these paths.
        download: if raw6 files aren't available, should we download them
            from MAST, or quit?
        recreate: if photonlist file is already present, should we recreate
            (and possibly overwrite) it?
        verbose: how many messages do you want to see? 0 turns almost all
            output off; levels up to 4 are meaningful.
        source_catalog_file: by default, the pipeline performs photometry on
            automatically-detected sources. passing the path to a CSV file as
            source_catalog_file specifies positions, preempting automated
            source detection.
        write: save images and/or movies to disk, or discard them after using
            them for photometry?
        aperture_sizes: what aperture size(s) (in arcseconds) should we
            use to compute photometry. passing multiple sizes may be useful
            for background estimation or related processes.
        lil: should we use matrix sparsification techniques whe movies?
            introduces some CPU overhead but can significantly reduce memory
            usage, especially for large numbers of frames, and especially
            during the frame integration and photometry steps.
        coregister_lightcurves: should we pin the start time of the first
            movie frame / lightcurve bin to the start time of the other
            band's first movie frame / lightcurve bin (if it exists)?
        stop_after: should we bail out after a particular phase? options are
            "photonpipe" (make photonlist only), "moviemaker" (make and write
            images and movies but don't perform photometry on them)
        compression: what sort of compression should we apply to movies and
            images? "gzip" is monolithic gzip; "rice" is RICE_1 (for the
            cntmap, lossy) tile compression; "none" is no compression at all.
        hdu_constructor_kwargs: optional mapping of kwargs to pass to
            `fitsio.FITS.write` (for instance, tile compression parameters)
        min_exptime: minimum effective exposure time to run image/movie
            and lightcurve generation. None means no lower bound.
        photometry_only: attempt to perform photometry on already-existing
            images/movies, doing nothing else
        burst: write movie frames to individual fits files? default is False.
        chunksz: max photons per chunk in photonpipe. default 1000000
        share_memory: use shared memory in photonpipe? default None, meaning
            do if running multithreaded, don't if not. True and False are
            also valid, and force use or non-use respectively.
        extended_photonlist: write extended variables to photonlists?
            these are not used in standard moviemaker/lightcurve pipelines.
            they are principally useful for diagnostics and ancillary products.
        extended_flagging: to run extended source finding. Includes flagging
        for non-catalog runs.
        aspect: default is standard aspect table, aspect.parquet ('aspect') but
            can designate to use alt aspect table, 'aspect2', which should be in the
            aspect directory and be named 'aspect2.parquet'
        override_eclipse_limits: attempt to execute pipeline even if metadata
            and/or support for this eclipse appear to be limited or absent?
            note that the pipeline will most likely still fail in these cases.
        suffix: optional string to append to the end of the output filenames
        aspect_dir: specifies the location of aspect tables
        ftype: file type desired for output files; can be either
            "csv" or "parquet", currently only affects photometry files
    Returns:
        str: `"return code: successful"` for fully successful execution;
            `"return code: {other_thing}"` for various known failure states
            (many of which produce a subset of valid output products)
    """
    e_warn, e_error = check_eclipse(eclipse, aspect_dir=aspect_dir)
    if (verbose > 0) and len(e_warn) > 0:
        print("\n".join(e_warn))
    if len(e_error) > 0:
        print("\n".join(e_error))
        if override_eclipse_limits is False:
            print("Bailing out.")
            return f"return code: {';'.join(e_error)}"
        print("override_eclipse_limits=True, continuing anyway")
    if not lil:
        warnings.warn(
            "lil=False no longer has any effect."
            " The `lil` argument will be removed in a future release."
        )
    if aspect not in ("aspect", "aspect2"):
        print(f"Invalid aspect argument {aspect}, bailing out.")
        return f"return code: invalid aspect argument {aspect}"
    if source_catalog_file is not None and not Path(source_catalog_file).exists():
        print(f"source_catalog_file {source_catalog_file} not found, bailing out.")
        return("return code: source catalog file not found")
    if not depth: # movie-writing has no meaning here
        write = dict(write)
        write['movie']=False
    ctx = PipeContext(
        eclipse,
        band,
        depth,
        compression,
        local_root,
        remote_root,
        aperture_sizes,
        threads=threads,
        download=download,
        recreate=recreate,
        verbose=verbose,
        source_catalog_file=source_catalog_file,
        write=write,
        lil=True,
        coregister_lightcurves=coregister_lightcurves,
        stop_after=stop_after,
        hdu_constructor_kwargs=hdu_constructor_kwargs,
        min_exptime=min_exptime,
        burst=burst,
        chunksz=chunksz,
        share_memory=share_memory,
        extended_photonlist=extended_photonlist,
        extended_flagging=extended_flagging,
        aspect=aspect,
        start_time=1000, # this is a no-op
        suffix=suffix,
        aspect_dir=aspect_dir,
        ftype=ftype,
    )
    ctx.watch.start()
    if photometry_only:
        return execute_photometry_only(ctx)
    return execute_full_pipeline(ctx)


def execute_full_pipeline(ctx):
    # SETUP AND FILE RETRIEVAL
    if ctx.verbose > 1:
        from gPhoton.aspect import aspect_tables

        metadata = aspect_tables(
            eclipse=ctx.eclipse,
            tables="metadata",
            aspect_dir=ctx.aspect_dir
        )[0]
        actual, nominal = gPhoton.reference.titular_legs(
            ctx.eclipse, aspect_dir=ctx.aspect_dir
        )
        headline = (
            f"eclipse {ctx.eclipse} {ctx.band}  -- "
            f"{metadata['obstype'][0].as_py()}; "
            f"{actual} leg(s)"
        )
        if actual != nominal:
            headline = headline + f" ({nominal} specified)"
        print(headline)
    if ctx.photometry_only is True:
        return execute_photometry_only(ctx)

    # fetch, locate, or create photonlist, as requested
    get_photonlist_result = get_photonlists(ctx, raise_errors=False)
    # we received an error return code
    if isinstance(get_photonlist_result, str):
        return get_photonlist_result  # this is an error return code
    else:
        photonpaths = get_photonlist_result  # strictly explanatory renaming
    if ctx.stop_after == "photonpipe":
        print(
            f"stop_after='photonpipe' passed, halting; "
            f"{round(time() - ctx.watch.start_time, 2)} "
            f"seconds for execution"
        )
        return "return code: successful (planned stop after photonpipe)"
    ctx.watch.click()
    from gPhoton.parquet_utils import get_parquet_stats

    ctx.start_time = get_parquet_stats(photonpaths[0], ['t'])['t']['min']
    leg_paths = []
    for path in photonpaths:
        p_stats = get_parquet_stats(str(path), ["flags", "ra"])
        if (p_stats["flags"]["min"] > 6) or (p_stats["ra"]["max"] is None):
            print(f"no unflagged data in {path}, not processing")
            leg_paths.append(False)
        leg_paths.append(path)
    if not any(leg_paths):
        print("no usable legs, bailing out.")
        return "return code: no unflagged data (stopped after photon list)"
    # MOVIE-RENDERING SECTION
    from gPhoton.moviemaker import (
        create_images_and_movies, write_moviemaker_results,
    )
    errors = []
    for leg_step, path in zip(ctx.explode_legs(), leg_paths):
        if path is False:
            print(f"skipping bad leg {leg_step.leg + 1}")
            continue
        # for brevity:
        # check to see if we're pinning our frame / lightcurve time series to
        # the time series of existing analysis for the other band
        fixed_start_time = check_fixed_start_time(leg_step)
        if len(photonpaths) > 1:
            print(f"processing leg {leg_step.leg + 1} of {len(photonpaths)}")
        try:
            results = create_images_and_movies(
                ctx, path, fixed_start_time=fixed_start_time
            )
        except ValueError as e:
            print(f"failed to create images and movies for leg {leg_step.leg}")
            print(f"Error: {e}")
            continue
        ctx.watch.click()
        if not (results["status"].startswith("successful")):
            message = (
                f"Moviemaker pipeline unsuccessful on leg {leg_step.leg} "
                f"{(results['status'])}"
            )
            print(message)
            errors.append(message)
            continue
        if ctx.stop_after != "moviemaker":
            from gPhoton.lightcurve import make_lightcurves

            photometry_result = make_lightcurves(results, leg_step)
        else:
            photometry_result = 'successful'
        write_result = write_moviemaker_results(results, leg_step)
        if photometry_result != 'successful':
            errors.append(photometry_result)
        if write_result != 'successful':
            errors.append(write_result)
    print(
        f"{round(time() - ctx.watch.start_time, 2)} seconds for execution"
    )
    if len(errors) > 0:
        return "return code: " + ";".join(errors)
    return "return code: successful"


def _look_for_raw6(ctx) -> Path:
    """
    :param ctx: pipeline context object
    :return: tuple of primary filename dict, path to photon list we'll be
    using, remote filename dict, name of temp/scratch directory
    :return: path to raw6 file
    """
    if (raw6path := Path(ctx["raw6"])).exists():
        return raw6path
    if (
        ctx.remote is not None
        and (remoteraw6 := Path(ctx(remote=True)['raw6'])).exists()
    ):
        print("making temp local copy of raw6 file from remote:", remoteraw6)
        raw6path = Path(shutil.copy(remoteraw6, ctx.temp_path()))
    if not raw6path.exists() and (ctx.download is True):
        from gPhoton.io.mast import retrieve_raw6

        print("downloading raw6file")
        raw6file = retrieve_raw6(ctx.eclipse, ctx.band, raw6path)
        if raw6file is not None:
            raw6path = Path(raw6file)
    return raw6path


def load_moviemaker_results(context, lil):
    create_local_paths(context)
    image = pick_and_copy_array(context, "image")
    if image is None:
        print("Photometry-only run, but image not found. Skipping.")
        return f"leg {context.leg}: image not found"
    if context.depth is not None:
        movie = pick_and_copy_array(context, "movie")
        if movie is None:
            print("Photometry-only run, but movie not found. Skipping.")
            return f"leg {context.leg}: movie not found"
    image_result = unpack_image(image, context.compression)
    results = {'wcs': image_result['wcs'], 'image_dict': image_result}
    if context.depth is not None:
        # noinspection PyUnboundLocalVariable
        results |= {
            'movie_dict': unpack_movie(movie, context.compression, lil)
        }
    else:
        results['movie_dict'] = {}
    return results


def pick_and_copy_array(context, which="image"):
    if (array_path := Path(context[which])).exists():
        return array_path
    if context.remote is None:
        return None
    if not Path(context(remote=True)[which]).exists():
        return None
    print(f"making temp copy of {which} from remote")
    shutil.copy(context(remote=True)[which], context.temp_path())
    return Path(context.temp_path(), Path(context[which]).name)


def create_local_paths(context: PipeContext) -> None:
    """
    path initialization step for execute_pipeline.
    :param context: PipeContext object containing pipeline options
    """
    if not context.eclipse_path().exists():
        context.eclipse_path().mkdir(parents=True)
    if not context.temp_path().exists():
        context.temp_path().mkdir(parents=True)


def parse_photonpipe_error(value_error: ValueError) -> str:
    if str(value_error).startswith("bad distortion correction"):
        print(str(value_error))
        return "return code: bad distortion correction solution"
    if "probably not a valid FUV observation" in str(value_error):
        print(str(value_error))
        return "return code: not a valid FUV observation"
    if "FUV temperature out of range" in str(value_error):
        print(str(value_error))
        return "return code: FUV temperature value out of range"
    raise value_error


def check_fixed_start_time(ctx: PipeContext) -> str | None:
    if (ctx.coregister_lightcurves is not True) or (ctx.depth is None):
        return None
    other = "NUV" if ctx.band == "FUV" else "FUV"
    expfile = None
    for root in filter(None, (ctx.remote, ctx.local)):
        exp_fn = ctx(root=root, band=other)["expfile"]
        if Path(exp_fn).exists():
            expfile = exp_fn
            break
    if expfile is None:
        print(
            f"Cross-band frame coregistration requested, but exposure "
            f"time table at this depth for {other} was not found."
        )
        return None
    import pandas as pd

    print(f"pinning first bin to first bin from {expfile}")
    # these files are small enough that we do not need to bother scratching
    # them to disk, even from a remote / fake filesystem
    coreg_exptime = pd.read_csv(expfile)
    return coreg_exptime["t0"].iloc[0]


def load_array_file(array_file, compression):
    import astropy.wcs
    import fitsio
    from gPhoton.io.fits_utils import AgnosticHDUL, pyfits_open_igzip
    if compression == 'gzip':
        hdul = AgnosticHDUL(pyfits_open_igzip(array_file))
    else:
        hdul = AgnosticHDUL(fitsio.FITS(array_file))
    cnt_hdu, flag_hdu = (hdul[i + 1] for i in range(2))
    headerdict = dict(cnt_hdu.header)
    tranges = keyfilter(lambda k: re.match(r"T[01]", k), headerdict)
    tranges = tuple(chunked(tranges.values(), 2))
    exptimes = tuple(
        keyfilter(lambda k: re.match(r"EXPT_", k), headerdict).values()
    )
    wcs = astropy.wcs.WCS(cnt_hdu.header)
    results = {"exptimes": exptimes, "tranges": tranges, "wcs": wcs}
    return (cnt_hdu, flag_hdu), results


def unpack_movie(movie_file, compression, lil):
    hdus, results = load_array_file(movie_file, compression)
    planes = ([], [])
    if lil is True:
        import scipy.sparse

        # noinspection PyUnresolvedReferences
        constructor = scipy.sparse.coo_matrix
    else:
        constructor = identity

    for hdu, plane in zip(hdus, planes):
        array = hdu.data
        for frame_ix in range(len(results['exptimes'])):
            cut = array[frame_ix]
            plane.append(constructor(cut))
    return results | {"cnt": planes[0], "flag": planes[1]}


def unpack_image(image_file, compression):
    hdus, results = load_array_file(image_file, compression)
    planes = {
        "cnt": hdus[0].data, "flag": hdus[1].data
    }
    return results | planes
