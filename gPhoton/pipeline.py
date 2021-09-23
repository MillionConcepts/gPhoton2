import shutil
from time import time
from pathlib import Path

from gPhoton import PhotonPipe
from gfcat import gfcat_utils as gfu
from gPhoton.gphoton_utils import get_parquet_stats, eclipse_to_files
from gPhoton.moviemaker import handle_movie_and_image_creation, write_movie
from gPhoton.photometry import (
    find_sources,
    extract_photometry,
    write_photometry_tables
)
from gPhoton.devtools import Stopwatch

stopwatch = Stopwatch()


def pipeline(
    eclipse,
    band,
    depth,
    threads=None,
    recreate=False,
    data_root="test_data",
    distinct_raw6_root=None,
    download=True
):
    startt = time()
    stopwatch.click()
    # TODO: this unnecessarily moves raw6 files around in some
    #  unlikely use cases
    if eclipse > 47000:
        print('CAUSE data w/ eclipse>47000 are not yet supported.')
        return 'CAUSE data w/ eclipse>47000 are not yet supported.'
    clean_raw6 = lambda _: None
    if download is True:
        raw6file = gfu.download_raw6(eclipse, band, data_directory=data_root)
    elif distinct_raw6_root is None:
        raw6file = eclipse_to_files(eclipse, data_root)[band]["raw6"]
    else:
        remote_raw6file = eclipse_to_files(
            eclipse, distinct_raw6_root
        )[band]["raw6"]
        print(f"making temp local copy of {remote_raw6file}")
        raw6file = shutil.copy(remote_raw6file, data_root)
        clean_raw6 = Path(raw6file).unlink
    if (raw6file is None) or not Path(str(raw6file)).exists():
        print("couldn't find raw6 file.")
        return "couldn't find raw6 file."
    filenames = eclipse_to_files(eclipse, data_root, depth)[band]
    photonpath = Path(filenames["photonfile"])
    if not photonpath.parent.exists():
        photonpath.parent.mkdir(parents=True)
    stopwatch.click()
    if recreate or not photonpath.exists():
        PhotonPipe.photonpipe(
            photonpath,
            band,
            raw6file=raw6file,
            verbose=2,
            chunksz=1000000,
            threads=threads
        )
    else:
        print(f"using existing photon list {photonpath}")
    stopwatch.click()
    file_stats = get_parquet_stats(str(photonpath), ["flags", "ra"])
    if (file_stats["flags"]["min"] > 6) or (file_stats["ra"]["max"] is None):
        print(f"no unflagged data in {photonpath}. bailing out.")
        clean_raw6()
        return f"no unflagged data (stopped after photon list)"
    results = handle_movie_and_image_creation(
        str(photonpath),
        depth,
        band,
        lil=True,
        threads=threads
    )
    stopwatch.click()
    source_table, apertures = find_sources(
        eclipse,
        band,
        str(photonpath.parent),
        results["image_dict"],
        results["wcs"],
    )
    if source_table is not None:
        stopwatch.click()
        source_table = extract_photometry(
            results["movie_dict"], source_table, apertures, threads
        )
        stopwatch.click()
        write_photometry_tables(
            filenames["photomfile"],
            filenames["expfile"],
            source_table,
            results["movie_dict"]
        )
        stopwatch.click()
    write_movie(
        band,
        None,
        filenames["image"].replace(".gz", ""),
        results["image_dict"],
        clean_up=True,
        wcs=results["wcs"]
    )
    del results["image_dict"]
    stopwatch.click()
    write_movie(
        band,
        depth,
        filenames["movie"].replace(".gz", ""),
        results["movie_dict"],
        clean_up=True,
        wcs=results["wcs"]
    )
    stopwatch.click()
    clean_raw6()
    print(f"{(time() - startt).__round__(2)} seconds for pipeline execution")
    if source_table is None:
        return "skipped photometry due to low exptime or other issue"
    return "successful"

