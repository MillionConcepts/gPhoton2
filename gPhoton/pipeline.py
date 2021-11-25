"""
primary handler module for the 'full' gPhoton execute_pipeline. can:
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

# oh no! divide by zero! i am very distracting!
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def execute_pipeline(
    eclipse,
    band,
    depth,
    threads=None,
    data_root="test_data",
    remote_root=None,
    download=True,
    recreate=False,
    verbose=2,
    maxsize=None,
    source_catalog_file: Optional[str] = None,
    write: Mapping = MappingProxyType({"image": True, "movie": True}),
    # sizes of apertures in arcseconds
    aperture_sizes: Sequence[float] = tuple([12.8]),
    lil=True,
):
    stopwatch = Stopwatch()
    startt = time()
    stopwatch.click()
    if eclipse > 47000:
        print("CAUSE data w/eclipse>47000 are not yet supported.")
        return "return code: CAUSE data w/eclipse>47000 are not yet supported."
    names = eclipse_to_paths(eclipse, data_root, depth)[band]
    remote_files = eclipse_to_paths(eclipse, remote_root)[band]
    temp_directory = Path(data_root, "temp", str(eclipse).zfill(5))
    if not temp_directory.exists():
        temp_directory.mkdir(parents=True)
    photonpath = Path(names["photonfile"])
    if not photonpath.parent.exists():
        photonpath.parent.mkdir(parents=True)
    stopwatch.click()
    if (remote_root is not None) and (recreate is False):
        # check for remote photon list.
        if Path(remote_files["photonfile"]).exists():
            print(
                f"making temp local copy of photon file from remote: "
                f"{remote_files['photonfile']}"
            )
            photonpath = Path(
                shutil.copy(Path(remote_files["photonfile"]), temp_directory)
            )
    if recreate or not photonpath.exists():
        # find raw6 file.
        # first look locally. then look in remote (like s3) if provided.
        # if it's still not found, download if download was set True.
        raw6path = Path(names["raw6"])
        if not raw6path.exists() and (remote_root is not None):
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
            if str(value_error).startswith("bad distortion correction"):
                print(str(value_error))
                return "return code: bad distortion correction solution"
            if "probably not a valid FUV observation" in str(value_error):
                print(str(value_error))
                return "return code: not a valid FUV observation"
            if "FUV temperature out of range" in str(value_error):
                print(str(value_error))
                return "return code: FUV temperature value out of range"
            raise
    else:
        print(f"using existing photon list {photonpath}")
    stopwatch.click()
    from gPhoton.parquet_utils import get_parquet_stats

    file_stats = get_parquet_stats(str(photonpath), ["flags", "ra"])
    if (file_stats["flags"]["min"] > 6) or (file_stats["ra"]["max"] is None):
        print(f"no unflagged data in {photonpath}. bailing out.")
        return "return code: no unflagged data (stopped after photon list)"
    from gPhoton.moviemaker import (
        create_images_and_movies,
        write_moviemaker_results,
    )

    results = create_images_and_movies(
        str(photonpath), depth, band, lil, threads, maxsize
    )
    stopwatch.click()
    if results["movie_dict"] == {}:
        print("No movies available, halting before photometry.")
        write_moviemaker_results(
            results, names, depth, band, write, maxsize, stopwatch
        )
        return "return code: " + results["status"]
    from gPhoton.lightcurve import make_lightcurves

    photometry_result = make_lightcurves(
        results,
        eclipse,
        band,
        aperture_sizes,
        photonpath,
        source_catalog_file,
        threads,
        names,
        stopwatch,
    )
    write_result = write_moviemaker_results(
        results, names, depth, band, write, maxsize, stopwatch
    )
    print(
        f"{(time() - startt).__round__(2)} seconds for pipeline execution"
    )
    failures = [
        result
        for result in (photometry_result, write_result)
        if result != "successful"
    ]
    if len(failures) > 0:
        return "return code: " + ";".join(failures)
    return "return code: successful"
