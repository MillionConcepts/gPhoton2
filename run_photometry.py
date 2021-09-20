import pickle
from pathlib import Path

from gPhoton.moviemaker import write_movie
from gPhoton.photometry import (
    find_sources,
    extract_photometry,
    write_photometry_tables,
)
from run_moviemaker import run_moviemaker

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


if __name__ == "__main__":
    eclipse = 22650
    depth = 30
    band = "NUV"
    datapath = Path("test_data", f"e{eclipse}")
    compression = "zstd"
    # results = run_moviemaker(eclipse, depth, band, lil=True, save_images=False)
    # source_table, apertures = find_sources(
    #     eclipse, datapath, results["image_dict"], results["wcs"], band
    # )
    source_table = pickle.load(open("source_table.pkl", "rb"))
    apertures = pickle.load(open("apertures.pkl", "rb"))
    results = pickle.load(open("results.pkl", "rb"))
    source_table = extract_photometry(
        results["movie_dict"]["cnt"], source_table, apertures, threads=None
    )
    write_photometry_tables(
        datapath, eclipse, depth, source_table, results["movie_dict"]
    )
    writer_kwargs = {
        "band": band,
        "eclipse": eclipse,
        "compression": compression,
        "wcs": results["wcs"],
        "outpath": datapath,
    }
    write_movie(
        movie_dict=results["image_dict"],
        depth=None,
        clean_up=True,
        **writer_kwargs,
    )
    del results["image_dict"]
    write_movie(
        movie_dict=results["movie_dict"],
        depth=depth,
        clean_up=True,
        **writer_kwargs,
    )
