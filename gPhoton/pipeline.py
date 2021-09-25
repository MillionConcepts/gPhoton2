from pathlib import Path
import shutil
from time import time

from gPhoton.pipeline_start import eclipse_to_files, Stopwatch


def pipeline(
    eclipse,
    band,
    depth,
    threads=None,
    data_root="test_data",
    remote_root=None,
    download=True,
    recreate = False,
):
    stopwatch = Stopwatch()
    startt = time()
    stopwatch.click()
    if eclipse > 47000:
        print('CAUSE data w/ eclipse>47000 are not yet supported.')
        return 'CAUSE data w/ eclipse>47000 are not yet supported.'
    filenames = eclipse_to_files(eclipse, data_root, depth)[band]
    remote_files = eclipse_to_files(eclipse, remote_root)[band]
    temp_directory = Path(data_root, "temp", str(eclipse).zfill(5))
    if not temp_directory.exists():
        temp_directory.mkdir(parents=True)
    photonpath = Path(filenames["photonfile"])
    if not photonpath.parent.exists():
        photonpath.parent.mkdir(parents=True)
    stopwatch.click()
    if remote_root is not None:
        # check for remote photon list.
        if Path(remote_files["photonfile"]).exists():
            print(
                f"making temp local copy of photon file from remote: "
                f"{remote_files['photonfile']}"
            )
            photonpath = shutil.copy(
                Path(remote_files["photonfile"]), temp_directory
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
            raw6path = shutil.copy(remote_files['raw6'], temp_directory)
        if not raw6path.exists() and (download is True):
            from gfcat.gfcat_utils import download_raw6
            raw6file = download_raw6(eclipse, band, data_directory=data_root)
            if raw6file is not None:
                raw6path = Path(raw6file)
        if not raw6path.exists():
            print("couldn't find raw6 file.")
            return "couldn't find raw6 file."
        from gPhoton import PhotonPipe
        PhotonPipe.photonpipe(
            photonpath,
            band,
            raw6file=str(raw6path),
            verbose=2,
            chunksz=1000000,
            threads=threads
        )
    else:
        print(f"using existing photon list {photonpath}")
    stopwatch.click()
    from gPhoton.pipeline_utils import get_parquet_stats
    file_stats = get_parquet_stats(str(photonpath), ["flags", "ra"])
    if (file_stats["flags"]["min"] > 6) or (file_stats["ra"]["max"] is None):
        print(f"no unflagged data in {photonpath}. bailing out.")
        return f"no unflagged data (stopped after photon list)"
    from gPhoton.moviemaker import handle_movie_and_image_creation, write_movie
    results = handle_movie_and_image_creation(
        str(photonpath),
        depth,
        band,
        lil=True,
        threads=threads
    )
    stopwatch.click()
    from gPhoton.photometry import find_sources, extract_photometry, write_photometry_tables
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
    print(f"{(time() - startt).__round__(2)} seconds for pipeline execution")
    if source_table is None:
        return "skipped photometry due to low exptime or other issue"
    return "successful"

