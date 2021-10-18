from multiprocessing import Pool
from pathlib import Path
from typing import Union, Optional
import warnings

import numpy as np
import pandas as pd
from photutils import DAOStarFinder, CircularAperture, aperture_photometry
import scipy.sparse

from gPhoton import MCUtils as mc


def find_sources(
    eclipse: int,
    band,
    datapath: Union[str, Path],
    image_dict,
    wcs,
    sources: pd.DataFrame = Optional[None],
):
    # TODO, maybe: pop these into a handler function
    if not image_dict["cnt"].max():
        print(f"{eclipse} appears to contain nothing in {band}.")
        Path(datapath, f"No{band}").touch()
        return None, None
    exptime = image_dict["exptimes"][0]
    if exptime < 300:
        print("Skipping low exposure time visit.")
        Path(datapath, "LowExpt").touch()
        return None, None
    if sources is None:
        print("Extracting sources.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            daofind = DAOStarFinder(fwhm=5, threshold=0.01)
            sources = daofind(image_dict["cnt"] / exptime).to_pandas()
        try:
            print(f"Located {len(sources)} sources.")
        except TypeError:
            print(f"{eclipse} {band} contains no sources.")
            Path(datapath, f"No{band}").touch()
            return None, None
        positions = sources[["xcentroid", "ycentroid"]].to_pandas().values
    else:
        print(f"Using specified catalog of {len(sources)} sources.")
        positions = np.vstack(
            [
                wcs.wcs_pix2world([position], 1, ra_dec_order=True)
                for position in sources[["ra", "dec"]].values
            ]
        )
        sources[["xcentroid", "ycentroid"]] = positions
    apertures = CircularAperture(positions, r=8.533333333333326)
    print("Performing aperture photometry on primary image.")
    phot_table = aperture_photometry(image_dict["cnt"], apertures).to_pandas()
    print("Performing aperture photometry on flag maps.")
    flag_table = aperture_photometry(image_dict["flag"], apertures).to_pandas()
    print("Performing aperture photometry on edge maps.")
    edge_table = aperture_photometry(image_dict["edge"], apertures).to_pandas()
    source_table = pd.concat(
        [sources, phot_table[["xcenter", "ycenter", "aperture_sum"]]],
        axis=1,
    )
    source_table["aperture_sum_mask"] = flag_table["aperture_sum"]
    source_table["aperture_sum_edge"] = edge_table["aperture_sum"]
    # TODO: this isn't necessary for specified catalog positions. but
    #  should we do a sanity check?
    if "ra" not in source_table.columns:
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


def _extract_photometry_unthreaded(movie, apertures):
    photometry = {}
    for ix, frame in enumerate(movie):
        mc.print_inline(f"extracting frame {ix}")
        photometry[ix] = extract_frame(frame, apertures)
    return photometry


# it's foolish to run this multithreaded unless you _are_ unpacking sparse
# matrices, but I won't stop you.
def _extract_photometry_threaded(movie, apertures, threads):
    pool = Pool(threads)
    photometry = {}
    for ix, frame in enumerate(movie):
        photometry[ix] = pool.apply_async(extract_frame, (frame, apertures))
    pool.close()
    pool.join()
    return {ix: result.get() for ix, result in photometry.items()}


def extract_photometry(movie_dict, source_table, apertures, threads):
    photometry_tables = []
    for key in ["cnt", "flag", "edge"]:
        title = "primary movie" if key == "cnt" else f"{key} map"
        print(f"extracting photometry from {title}")
        if threads is None:
            photometry = _extract_photometry_unthreaded(
                movie_dict[key], apertures
            )
        else:
            photometry = _extract_photometry_threaded(
                movie_dict[key], apertures, threads
            )
        frame_indices = sorted(photometry.keys())
        if key in ("edge", "flag"):
            column_prefix = f"aperture_sum_{key}"
        else:
            column_prefix = "aperture_sum"
        photometry = {
            f"{column_prefix}_{ix}": photometry[ix] for ix in frame_indices
        }
        photometry_tables.append(pd.DataFrame.from_dict(photometry))
    return pd.concat([source_table, *photometry_tables], axis=1)


def write_photometry_tables(photomfile, expfile, source_table, movie_dict):
    print(f"writing source table to {photomfile}")
    source_table.to_csv(photomfile, index=False)
    exptime_table = pd.DataFrame(
        {
            "expt": movie_dict["exptimes"],
            "t0": [trange[0] for trange in movie_dict["tranges"]],
            "t1": [trange[1] for trange in movie_dict["tranges"]],
        }
    )
    print(f"writing exposure time table to {expfile}")
    # noinspection PyTypeChecker
    exptime_table.to_csv(expfile, index=False)
