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

from pathlib import Path
import shutil
from time import time
from types import MappingProxyType
from typing import Optional, Sequence, Mapping
import warnings

from gPhoton.reference import eclipse_to_paths, Stopwatch
from gPhoton.types import GalexBand, Pathlike

# oh no! divide by zero! i am very distracting!

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def execute_pipeline(
    eclipse: int,
    band: GalexBand,
    depth: int,
    threads: Optional[int] = None,
    local_root: str = "test_data",
    remote_root: Optional[str] = None,
    download: bool = True,
    recreate: bool = False,
    verbose: int = 2,
    maxsize: Optional[int] = None,
    source_catalog_file: Optional[str] = None,
    write: Mapping = MappingProxyType({"image": True, "movie": True}),
    aperture_sizes: Sequence[float] = tuple([12.8]),
    lil: bool = True,
    coregister_lightcurves: bool = False,
) -> str:
    """
    :param eclipse: GALEX eclipse number to run
    :param band: "NUV" or "FUV" -- near or far ultraviolet
    :param depth: how many seconds of events to use when integrating each
    movie frame. in a sense: inverse FPS.
    :param threads: how many threads to use for parallel processing. "None"
    turns off parallel processing entirely. "1" will perform parallelism in a
    single thread (not recommended except for test purposes). Increasing
    thread count increases execution speed but also increases memory pressure,
    particularly for movie creation.
    :param local_root: path to write pipeline results to
    :param remote_root: additional path to check for preexisting raw6 and
    photonlist files (intended primarily for multiple servers referencing a
    shared set of remote resources)
    :param download: if raw6 files aren't available, should we download them
    from MAST, or quit?
    :param recreate: if photonlist file is already present,
    should we recreate (and possibly overwrite) it?
    :param verbose: how many messages do you want to see? 0 turns almost all
    output off; levels up to 4 are meaningful.
    TODO: make verbosity more consistent across the codebase
    :param maxsize: maximum working memory size in bytes. if estimated memory
    cost of generating image or movie frames exceeds this threshold, the
    pipeline will stop. Note that estimates are not 100% reliable!
    :param source_catalog_file: CSV file specifying source positions to use
    for photometry; preempts default automatic source detection
    :param write: should we save the images and/or movies we make, or discard
    them after using them for photometry?
    :param aperture_sizes: what aperture size(s) should we compute photometry
    for? multiple sizes may be useful for background estimation, etc.
    :param lil: should we use matrix sparsification techniques on movies?
    introduces some CPU overhead but can significantly reduce memory usage,
    especially for large numbers of frames, and especially during the frame
    integration and photometry steps.
    :param coregister_lightcurves: should we pin the start time of the first
    movie frame / lightcurve bin to the start time of the other band's first
    movie frame / lightcurve bin (if it exists)?
    :returns: "return code: successful" for fully successful
    execution; "return code: {other_thing}" for various known failure states
    (many of which produce a subset of valid output products)
    """

    # SETUP AND FILE RETRIEVAL
    stopwatch = Stopwatch()
    stopwatch.click()
    if eclipse > 47000:
        print("CAUSE data w/eclipse>47000 are not yet supported.")
        return "return code: CAUSE data w/eclipse>47000 are not yet supported."
    local_files, photonpath, remote_files, temp_directory = _set_up_paths(
        eclipse, band, local_root, remote_root, depth, recreate
    )

    # PHOTONLIST CREATION
    if recreate or not photonpath.exists():
        raw6path = _look_for_raw6(
            eclipse, band, download, local_files, remote_files, temp_directory
        )
        if not raw6path.exists():
            print("couldn't find raw6 file.")
            return "return code: couldn't find raw6 file."
        from gPhoton.photonpipe import execute_photonpipe

        try:
            execute_photonpipe(
                photonpath,
                band,
                raw6file=str(raw6path),
                verbose=verbose,
                chunksz=1000000,
                threads=threads,
            )
        except ValueError as value_error:
            return parse_photonpipe_error(value_error)
    else:
        print(f"using existing photon list {photonpath}")
    stopwatch.click()
    from gPhoton.parquet_utils import get_parquet_stats

    file_stats = get_parquet_stats(str(photonpath), ["flags", "ra"])
    if (file_stats["flags"]["min"] > 6) or (file_stats["ra"]["max"] is None):
        print(f"no unflagged data in {photonpath}. bailing out.")
        return "return code: no unflagged data (stopped after photon list)"

    # MOVIE-RENDERING SECTION
    from gPhoton.moviemaker import (
        create_images_and_movies,
        write_moviemaker_results,
    )

    # check to see if we're pinning our frame / lightcurve time series to
    # the time series of existing analysis for the other band
    fixed_start_time = check_fixed_start_time(
        eclipse, depth, local_root, remote_root, band, coregister_lightcurves
    )
    results = create_images_and_movies(
        str(photonpath), depth, band, lil, threads, maxsize, fixed_start_time
    )
    stopwatch.click()
    if results["movie_dict"] == {}:
        print("No movies available, halting before photometry.")
        write_moviemaker_results(
            results, local_files, depth, band, write, maxsize, stopwatch
        )
        return "return code: " + results["status"]

    # PHOTOMETRY SECTION
    from gPhoton.lightcurve import make_lightcurves

    photometry_result = make_lightcurves(
        results,
        eclipse,
        band,
        aperture_sizes,
        photonpath,
        source_catalog_file,
        threads,
        local_files,
        stopwatch,
    )

    # CLEANUP & CLOSEOUT SECTION
    write_result = write_moviemaker_results(
        results, local_files, depth, band, write, maxsize, stopwatch
    )
    print(f"{round(time() - stopwatch.start_time, 2)} seconds for execution")
    failures = [
        result
        for result in (photometry_result, write_result)
        if result != "successful"
    ]
    if len(failures) > 0:
        return "return code: " + ";".join(failures)
    return "return code: successful"


def _look_for_raw6(
    eclipse: int,
    band: GalexBand,
    download: bool,
    names: Mapping,
    remote_files: Optional[Mapping],
    temp_directory: Path,
) -> Path:
    """
    :param eclipse: GALEX eclipse number to run
    :param band: "NUV" or "FUV" -- near or far ultraviolet
    :param remote_files: mapping of paths to remote_files generated in
    _set_up_paths(); none if there is no remote_root
    :param temp_directory: path to scratch directory for temp copy of raw6
    from remote_root
    :return: tuple of primary filename dict, path to photon list we'll be
    using, remote filename dict, name of temp/scratch directory
    :return: path to raw6 file
    """

    raw6path = Path(names["raw6"])
    if not raw6path.exists() and (remote_files is not None):
        if Path(remote_files["raw6"]).exists():
            print(
                f"making temp local copy of raw6 file from remote: "
                f"{remote_files['raw6']}"
            )
        raw6path = Path(shutil.copy(remote_files["raw6"], temp_directory))
    if not raw6path.exists() and (download is True):
        from gPhoton.io.fetch import retrieve_raw6

        raw6file = retrieve_raw6(eclipse, band, raw6path)
        if raw6file is not None:
            raw6path = Path(raw6file)
    return raw6path


def _set_up_paths(
    eclipse: int,
    band: GalexBand,
    data_root: str,
    remote_root: Optional[str],
    depth: int,
    recreate: bool,
) -> tuple[dict, Path, Optional[dict], Path]:
    """
    initial path setup & file retrieval step for execute_pipeline.
    :param eclipse: GALEX eclipse number to run
    :param band: "NUV" or "FUV" -- near or far ultraviolet
    :param depth: how many seconds of events to use when integrating each
    movie frame. in a sense: inverse FPS.
    :param data_root: path to write pipeline results to
    :param remote_root: additional path to check for preexisting raw6 and
    photonlist files (intended primarily for multiple servers referencing a
    shared set of remote resources)
    :param recreate: if photonlist file is already present,
    should we recreate (and possibly overwrite) it?
    :return: tuple of: primary filename dict, path to photon list we'll be
    using, remote filename dict, name of temp/scratch directory
    """
    local_files = eclipse_to_paths(eclipse, data_root, depth)[band]
    if remote_root is not None:
        remote_files = eclipse_to_paths(eclipse, remote_root)[band]
    else:
        remote_files = None
    temp_directory = Path(data_root, "temp", str(eclipse).zfill(5))
    if not temp_directory.exists():
        temp_directory.mkdir(parents=True)
    photonpath = Path(local_files["photonfile"])
    if not photonpath.parent.exists():
        photonpath.parent.mkdir(parents=True)
    if (remote_root is not None) and (recreate is False):
        if Path(remote_files["photonfile"]).exists():
            print(
                f"making temp local copy of photon file from remote: "
                f"{remote_files['photonfile']}"
            )
            photonpath = Path(
                shutil.copy(Path(remote_files["photonfile"]), temp_directory)
            )
    return local_files, photonpath, remote_files, temp_directory


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


def check_fixed_start_time(
    eclipse,
    depth,
    local: Pathlike,
    remote: Optional[Pathlike],
    band: GalexBand,
    coregister: bool,
) -> Optional[str]:
    if coregister is not True:
        return None
    other = "NUV" if band == "FUV" else "FUV"
    expfile = None
    for location in (remote, local):
        if location is None:
            continue
        exp_fn = eclipse_to_paths(eclipse, location, depth)[other]["expfile"]
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
