import time
from pathlib import Path


from gPhoton.gphoton_utils import get_parquet_stats
from gPhoton.moviemaker import handle_movie_and_image_creation
from run_photonpipe import run_photonpipe


def run_moviemaker(
    eclipse,
    depth,
    band="NUV",
    make_photon_list=True,
    lil=False,
    compression="zstd",
    make_full_frame=True,
    write_to_file=True
):
    data_directory = "test_data"
    photonfile = Path(
        data_directory,
        f"e{eclipse}",
        f"e{eclipse}-{'n' if band == 'NUV' else 'f'}d.parquet",
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
    handle_movie_and_image_creation(
        eclipse,
        depth,
        band,
        lil,
        compression=compression,
        make_full=make_full_frame,
        write_to_file=write_to_file
    )


for ecl in [22650]:
    start = time.time()
    run_moviemaker(
        eclipse=ecl,
        depth=30,
        band="NUV",
        make_photon_list=False,
        lil=True,
        compression="zstd",
        write_to_file=False
    )
    print(time.time() - start)

