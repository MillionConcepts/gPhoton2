from pathlib import Path
import shutil
from time import time
from types import MappingProxyType
from typing import Optional, Sequence, Mapping
import warnings

import gPhoton.constants as c
from gPhoton._pipe_components import retrieve_raw6
from gPhoton.pipeline_start import eclipse_to_files, Stopwatch

warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def pipeline(
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
    names = eclipse_to_files(eclipse, data_root, depth)[band]
    remote_files = eclipse_to_files(eclipse, remote_root)[band]
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
        raw6path = Path(eclipse_to_files(eclipse, data_root)[band]["raw6"])
        if not raw6path.exists() and (remote_root is not None):
            if Path(remote_files["raw6"]).exists():
                print(
                    f"making temp local copy of raw6 file from remote: "
                    f"{remote_files['raw6']}"
                )
            raw6path = Path(shutil.copy(remote_files["raw6"], temp_directory))
        if not raw6path.exists() and (download is True):
            raw6file = retrieve_raw6(
                eclipse, band, str(Path(data_root, str(eclipse).zfill(5)))
            )
            if raw6file is not None:
                raw6path = Path(raw6file)
        if not raw6path.exists():
            print("couldn't find raw6 file.")
            return "return code: couldn't find raw6 file."
        from gPhoton import PhotonPipe

        try:
            PhotonPipe.photonpipe(
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
    from gPhoton.pipeline_utils import get_parquet_stats

    file_stats = get_parquet_stats(str(photonpath), ["flags", "ra"])
    if (file_stats["flags"]["min"] > 6) or (file_stats["ra"]["max"] is None):
        print(f"no unflagged data in {photonpath}. bailing out.")
        return "return code: no unflagged data (stopped after photon list)"
    from gPhoton.moviemaker import handle_movie_and_image_creation, write_fits

    results = handle_movie_and_image_creation(
        str(photonpath), depth, band, lil, threads, maxsize
    )
    stopwatch.click()
    if results["movie_dict"] == {}:
        print("No movies available, halting pipeline before photometry.")
        write_fits(results, names, depth, band, write, maxsize, stopwatch)
        return "return code: " + results["status"]
    from gPhoton.photometry import (
        find_sources,
        extract_photometry,
        count_full_depth_image,
        write_exptime_file,
    )

    if source_catalog_file is not None:
        import pandas as pd
        sources = pd.read_csv(source_catalog_file)
        sources = sources.loc[sources["eclipse"] == eclipse]
        sources = sources[~sources.duplicated()].reset_index(drop=True)
    else:
        sources = None
    source_table = find_sources(
        eclipse,
        band,
        str(photonpath.parent),
        results["image_dict"],
        results["wcs"],
        source_table=sources,
    )
    # failure messages due to low exptime or no data
    if isinstance(source_table, str):
        return f"return code: {source_table}"
    stopwatch.click()
    # if source_table is None at this point, it should mean that DAOStarFinder
    # didn't find anything
    if source_table is not None:
        for aperture_size in aperture_sizes:
            aperture_size_px = aperture_size / c.ARCSECPERPIXEL
            source_table, apertures = count_full_depth_image(
                source_table,
                aperture_size_px,
                results["image_dict"],
                results["wcs"],
            )
            stopwatch.click()
            source_table = extract_photometry(
                results["movie_dict"], source_table, apertures, threads
            )
            photomfile = (
                names["photomfile"]
                + str(aperture_size).replace(".", "_")
                + ".csv"
            )
            print(f"writing source table to {photomfile}")
            source_table.to_csv(photomfile, index=False)
            stopwatch.click()
        write_exptime_file(names["expfile"], results["movie_dict"])
    write_result = write_fits(
        results, names, depth, band, write, maxsize, stopwatch
    )
    print(f"{(time() - startt).__round__(2)} seconds for pipeline execution")
    failures = []
    if sources is None:
        failures.append("skipped photometry due to low exptime or other issue")
    if write_result != "successful":
        failures.append(write_result)
    if len(failures) > 0:
        return "return code: " + ";".join(failures)
    return "return code: successful"
