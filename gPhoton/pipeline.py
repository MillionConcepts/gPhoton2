from time import time
from pathlib import Path

from gPhoton.gphoton_utils import get_parquet_stats
from gPhoton.moviemaker import write_movie
from gPhoton.photometry import (
    find_sources,
    extract_photometry,
    write_photometry_tables,
)
from gPhoton.moviemaker import handle_movie_and_image_creation

# def load_full_depth_image(eclipse, datapath):
#     prefix = f"e{eclipse}-full-"
#     full_depth = read_image(Path(datapath, f"{prefix}cnt.fits.zstd"))
#     flag = read_image(Path(datapath, f"{prefix}flag.fits.zstd"))["image"]
#     edge = read_image(Path(datapath, f"{prefix}edge.fits.zstd"))["image"]
#     image_dict = {
#         "cnt": full_depth["image"],
#         "flag": flag,
#         "edge": edge,
#         "exptime": full_depth["exptimes"][0],
#     }
#     wcs = full_depth["wcs"]
#     return image_dict, wcs

from gPhoton.devtools import Stopwatch
from run_photonpipe import run_photonpipe

stopwatch = Stopwatch()


def pipeline(
    eclipse=None,
    band=None,
    depth=None,
    threads=None,
    recreate=False,
    data_root = "test_data"
):
    now = time()
    stopwatch.click()
    data_path = Path(data_root, f"e{eclipse}")
    photonfile = Path(
        data_path, f"e{eclipse}-{'n' if band == 'NUV' else 'f'}d.parquet"
    )
    if photonfile.exists():
        if recreate is True:
            print(f"overwriting {photonfile}")
            # TODO: run_photonpipe needs to be able to accept an eclipse
            run_photonpipe(eclipse)
        else:
            print(f"using existing photon list {photonfile}")
    else:
        run_photonpipe(eclipse)
    file_stats = get_parquet_stats(photonfile, ["flags", "ra"])
    if (file_stats["flags"]["min"] > 6) or (file_stats["ra"]["max"] is None):
        print(f"no unflagged data in {photonfile}. bailing out.")
        return
    stopwatch.click()

    results = handle_movie_and_image_creation(
        eclipse,
        depth,
        band,
        lil=True,
        threads=threads
    )
    stopwatch.click()
    source_table, apertures = find_sources(
        eclipse, data_path, results["image_dict"], results["wcs"], band
    )
    stopwatch.click()
    source_table = extract_photometry(
        results["movie_dict"], source_table, apertures, threads
    )
    stopwatch.click()
    write_photometry_tables(
        data_path, eclipse, depth, source_table, results["movie_dict"]
    )
    stopwatch.click()
    writer_kwargs = {
        "band": band,
        "eclipse": eclipse,
        "wcs": results["wcs"],
        "outpath": data_path,
    }
    write_movie(
        movie_dict=results["image_dict"],
        depth=None,
        clean_up=True,
        **writer_kwargs,
    )
    del results["image_dict"]
    stopwatch.click()
    write_movie(
        movie_dict=results["movie_dict"],
        depth=depth,
        clean_up=True,
        **writer_kwargs,
    )
    stopwatch.click()
    print(f"{(time() - now).__round__(2)} seconds for pipeline execution")
