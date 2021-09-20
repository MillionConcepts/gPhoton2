import time
from pathlib import Path


from gPhoton.gphoton_utils import get_parquet_stats
from gPhoton.moviemaker import handle_movie_and_image_creation, write_movie
from run_photonpipe import run_photonpipe


def run_moviemaker(
    eclipse,
    depth,
    band="NUV",
    make_photon_list=True,
    lil=False,
    make_full_frame=True,
    compression="zstd",
    save_images=True,
    save_movies=False,
):
    data_path = Path("test_data", f"e{eclipse}")
    photonfile = Path(
        data_path, f"e{eclipse}-{'n' if band == 'NUV' else 'f'}d.parquet"
    )
    if not photonfile.exists():
        if make_photon_list is True:
            run_photonpipe(eclipse)
        else:
            raise OSError(
                f"{photonfile} does not exist and make_photon_list is set "
                f"to False. bailing out."
            )
    file_stats = get_parquet_stats(photonfile, ["flags", "ra"])
    if (file_stats["flags"]["min"] > 6) or (file_stats["ra"]["max"] is None):
        print(f"no unflagged data in {photonfile}. bailing out.")
        return
    results = handle_movie_and_image_creation(
        eclipse,
        depth,
        band,
        lil,
        make_full=make_full_frame,
    )
    writer_kwargs = {
        "band": band,
        "eclipse": eclipse,
        "compression": compression,
        "wcs": results["wcs"],
        "outpath": data_path,
    }
    if save_images:
        write_movie(
            movie_dict=results["image_dict"], depth=None, **writer_kwargs
        )
    if save_movies:
        write_movie(
            movie_dict=results["movie_dict"], depth=depth, **writer_kwargs
        )
    return results


if __name__ == "__main__":
    for ecl in [22650]:
        start = time.time()
        run_moviemaker(
            eclipse=ecl,
            depth=120,
            band="NUV",
            make_photon_list=False,
            lil=True,
            compression="zstd",
        )
        print(time.time() - start)
