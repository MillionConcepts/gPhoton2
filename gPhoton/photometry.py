from multiprocessing import Pool
from pathlib import Path
from typing import Union

import pandas as pd
import scipy.sparse
from photutils import DAOStarFinder, CircularAperture, aperture_photometry


def find_sources(
    eclipse: int,
    datapath: Union[str, Path],
    image_dict,
    wcs,
    band: str = "NUV",
):
    # TODO, maybe: pop these into a handler function
    if not image_dict["cnt"].max():
        print(f"{eclipse} appears to contain nothing in {band}.")
        Path(datapath, f"No{band}").touch()
        return
    exptime = image_dict["exptimes"][0]
    if exptime < 600:
        print("Skipping low exposure time visit.")
        Path(datapath, "LowExpt").touch()
        return
    daofind = DAOStarFinder(fwhm=5, threshold=0.01)
    sources = daofind(image_dict["cnt"] / exptime)
    try:
        print(f"Located {len(sources)} sources.")
    except TypeError:
        print(f"{eclipse} {band} contains no sources.")
        Path(datapath, f"No{band}").touch()
        return
    positions = sources[["xcentroid", "ycentroid"]].to_pandas().values
    apertures = CircularAperture(positions, r=8.533333333333326)
    phot_table = aperture_photometry(image_dict["cnt"], apertures).to_pandas()
    flag_table = aperture_photometry(image_dict["flag"], apertures).to_pandas()
    edge_table = aperture_photometry(image_dict["edge"], apertures).to_pandas()
    source_table = pd.concat(
        [
            sources.to_pandas(),
            phot_table[["xcenter", "ycenter", "aperture_sum"]],
        ],
        axis=1,
    )
    source_table["aperture_sum_mask"] = flag_table["aperture_sum"]
    source_table["aperture_sum_edge"] = edge_table["aperture_sum"]
    world = [
        wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()
        for pos in apertures.positions
    ]
    source_table["ra"] = [coord[0] for coord in world]
    source_table["dec"] = [coord[1] for coord in world]
    return source_table, apertures


def extract_frame(frame, apertures):
    if isinstance(frame, scipy.sparse.spmatrix):
        frame = frame.toarray()
    return aperture_photometry(frame, apertures)["aperture_sum"].data


def _extract_photometry_unthreaded(movie, source_table, apertures):
    photometry = {}
    for ix, frame in enumerate(movie):
        photometry[ix] = extract_frame(frame, apertures)
    photometry = {
        f"aperture_sum_{ix}": photometry[ix] for ix in photometry.keys()
    }
    photometry = pd.DataFrame.from_dict(photometry)
    return pd.concat([source_table, photometry], axis=1)


# it's foolish to run this multithreaded unless you _are_ unpacking sparse
# matrices, but I won't stop you.
def _extract_photometry_threaded(movie, source_table, apertures, threads):
    pool = Pool(threads)
    photometry = {}
    for ix, frame in enumerate(movie):
        photometry[ix] = pool.apply_async(extract_frame, (frame, apertures))
    frame_indices = sorted(photometry.keys())
    pool.close()
    pool.join()
    photometry = {
        f"aperture_sum_{ix}": photometry[ix].get() for ix in frame_indices
    }
    photometry = pd.DataFrame.from_dict(photometry)
    return pd.concat([source_table, photometry], axis=1)


def extract_photometry(movie, source_table, apertures, threads):
    if threads is None:
        return _extract_photometry_unthreaded(movie, source_table, apertures)
    return _extract_photometry_threaded(
        movie, source_table, apertures, threads
    )


def write_photometry_tables(
    datapath, eclipse, depth, source_table, movie_dict
):
    photomfile = Path(datapath, f"e{eclipse}-{depth}s-photom.csv")
    source_table.to_csv(photomfile, index=False)
    exptime_table = pd.DataFrame(
        {
            "expt": movie_dict["exptimes"],
            "t0": [trange[0] for trange in movie_dict["tranges"]],
            "t1": [trange[1] for trange in movie_dict["tranges"]],
        }
    )
    exptimefile = Path(datapath, f"e{eclipse}-{depth}s-exptime.csv")
    # noinspection PyTypeChecker
    exptime_table.to_csv(exptimefile, index=False)
