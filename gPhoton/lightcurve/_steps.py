"""
methods for generating lightcurves from FITS images/movies, especially those
produced by the gPhoton.moviemaker pipeline. They are principally intended
for use as components of the primary gPhoton.lightcurve pipeline, called as
part of the course of running gPhoton.lightcurve.core.make_lightcurves(), and
may not suitable for independent use.
"""

from multiprocessing import Pool
from pathlib import Path
from typing import Union, Optional, Mapping
import warnings

import astropy.wcs
import gc
import numpy as np
import pandas as pd
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
import scipy.sparse

from gPhoton.pretty import print_inline
from gPhoton.types import Pathlike, GalexBand
from gPhoton.lightcurve.photometry_utils import (mask_for_extended_sources,
                                                 check_point_in_extended)

def count_full_depth_image(
    source_table: pd.DataFrame,
    aperture_size: float,
    image_dict: Mapping[str, np.ndarray],
    system: astropy.wcs.WCS,
    ctx
):
    source_table = source_table.reset_index(drop=True)
    positions = source_table[["xcentroid", "ycentroid"]].to_numpy()
    apertures = CircularAperture(positions, r=aperture_size)
    phot_table = aperture_photometry(image_dict["cnt"], apertures, method='exact').to_pandas()
    source_table = pd.concat(
        [source_table, phot_table[["xcenter", "ycenter", "aperture_sum"]]],
        axis=1,
    )
    if ctx.source_catalog_file is None:
        # we don't want to run this for forced photometry
        # aperture photometry of YA value, primarily for ghosts in post-CSP
        ya_apers = apertures.area_overlap(
            image_dict["ya"],
            mask=np.isnan(image_dict["ya"]),
            method="exact",
        )
        ya_phot_table = aperture_photometry(
            image_dict["ya"],
            apertures,
            mask=np.isnan(image_dict["ya"]),
            method="exact",
        ).to_pandas()
        ya_phot_table = ya_phot_table.rename(
            columns={"aperture_sum": "ya_aperture_sum"}
        )
        source_table = pd.concat(
            [source_table, ya_phot_table[["ya_aperture_sum"]]],
            axis=1,
        )
        source_table["ya_aperture_sum"] /= ya_apers
        del ya_apers
        gc.collect()

        # col / row dispersion aperture photometry
        colrow_sum = image_dict["col"] + image_dict["row"]
        disp_apers = apertures.area_overlap(
            colrow_sum,
            mask=np.isnan(colrow_sum),
            method="exact",
        )
        stdcolrow_phot_table = aperture_photometry(
            colrow_sum,
            apertures,
            mask=np.isnan(colrow_sum),
            method="exact",
        ).to_pandas()
        stdcolrow_phot_table = stdcolrow_phot_table.rename(
            columns={"aperture_sum": "stdcolrow_aperture_sum"}
        )
        source_table = pd.concat(
            [source_table, stdcolrow_phot_table[["stdcolrow_aperture_sum"]]],
            axis=1,
        )
        source_table["stdcolrow_aperture_sum"] /= disp_apers
        del disp_apers
        gc.collect()

        # Q (pulse height) aperture photometry
        q_apers = apertures.area_overlap(
            image_dict["q"],
            mask=np.isnan(image_dict["q"]),
            method="exact",
        )
        q_phot_table = aperture_photometry(
            image_dict["q"],
            apertures,
            mask=np.isnan(image_dict["q"]),
            method="exact",
        ).to_pandas()
        q_phot_table = q_phot_table.rename(
            columns={"aperture_sum": "q_aperture_sum"}
        )

        source_table = pd.concat(
            [source_table, q_phot_table[["q_aperture_sum"]]],
            axis=1,
        )
        source_table["q_aperture_sum"] /= q_apers
        del q_apers
        gc.collect()

    source_table["artifact_flag"] = bitwise_aperture_photometry(
        image_dict["flag"],
        apertures)

    # TODO: this isn't necessary for specified catalog positions. but
    #  should we do a sanity check?
    if "ra" not in source_table.columns:
        world = system.wcs_pix2world(apertures.positions, 1, ra_dec_order=True)
        source_table["ra"] = world[:, 0]
        source_table["dec"] = world[:, 1]
    return source_table, apertures


def bitwise_aperture_photometry(artifact_map: np.ndarray, apertures):
    """ Run aperture photometry on each bit in artifact flag backplane.
    Non-zero results mean that bit is flagged. Only 4 bits set rn. """
    bitmask_column= np.zeros(len(apertures), dtype=np.uint8)
    # add check to see if there are any bits in the map?
    for bit in range(4):
        if not np.any(artifact_map & (1 << bit)):
            # most useful for skipping ghost flag in eclipses pre-CSP
            continue
        bit_image = (artifact_map & (1 << bit)) > 0
        bit_flag_table = aperture_photometry(bit_image.astype(int), apertures).to_pandas()
        bitmask_column |= (bit_flag_table["aperture_sum"].to_numpy() > 0).astype(np.uint8) << bit
    return bitmask_column


def check_empty_image(eclipse:int, band:GalexBand, image_dict):
    if not image_dict["cnt"].max():
        print(f"{eclipse} appears to contain nothing in {band}.")
        # do we want to return a file when the image is empty?
        # realistically it probably means there's something wrong
        # with the observation
        return f"{eclipse} appears to contain nothing in {band}."


def extract_frame(frame, apertures, key):
    if isinstance(frame, scipy.sparse.spmatrix):
        frame = frame.toarray()
    if key == "flag":
        return bitwise_aperture_photometry(frame, apertures)
    return aperture_photometry(frame, apertures)["aperture_sum"].data


def _extract_photometry_unthreaded(movie, apertures, key):
    photometry = {}
    for ix, frame in enumerate(movie):
        print_inline(f"extracting frame {ix}")
        photometry[ix] = extract_frame(frame, apertures, key)
    return photometry


# it's foolish to run this multithreaded unless you _are_ unpacking sparse
# matrices, but I won't stop you.
def _extract_photometry_threaded(movie, apertures, threads, key):
    pool = Pool(threads)
    photometry = {}
    for ix, frame in enumerate(movie):
        photometry[ix] = pool.apply_async(extract_frame, (frame, apertures, key))
    pool.close()
    pool.join()
    return {ix: result.get() for ix, result in photometry.items()}


def extract_photometry(movie_dict, source_table, apertures, threads):
    photometry_tables = []
    for key in ["cnt", "flag"]:
        title = "primary movie" if key == "cnt" else f"{key} map"
        print(f"extracting photometry from {title}")
        if threads is None:
            photometry = _extract_photometry_unthreaded(
                movie_dict[key], apertures, key
            )
        else:
            photometry = _extract_photometry_threaded(
                movie_dict[key], apertures, threads, key
            )
        frame_indices = sorted(photometry.keys())
        if key in ("flag"):
            column_prefix = f"artifact_flag"
        else:
            column_prefix = "aperture_sum"
        photometry = {
            f"{column_prefix}_{ix}": photometry[ix] for ix in frame_indices
        }
        if key == "cnt":
            # add exposure time info
            # only need to do this once though
            expt_dict = {
                "expt": movie_dict["exptimes"],
                "t0": [trange[0] for trange in movie_dict["tranges"]],
                "t1": [trange[1] for trange in movie_dict["tranges"]],
            }
            for name, values in expt_dict.items():
                for i, val in enumerate(values):
                    photometry[f"{name}_{i}"] = round(val,3)

        photometry_tables.append(pd.DataFrame.from_dict(photometry))
    return pd.concat([source_table, *photometry_tables], axis=1)


def write_exptime_file(expfile: Pathlike, movie_dict) -> None:
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


def _load_csv_catalog(
    source_catalog_file: Pathlike, eclipse: int
) -> pd.DataFrame:
    sources = pd.read_csv(source_catalog_file)
    try:
        return sources.loc[sources["eclipse"] == eclipse]
    except KeyError:
        # If there is not an eclipse column, then just use everything
        # This lets you feed gPhoton2's outputs back into itself
        return sources


def _load_parquet_catalog(
    source_catalog_file: Pathlike, eclipse: int
) -> pd.DataFrame:
    from pyarrow import parquet
    ds = parquet.read_schema(source_catalog_file)
    if "eclipse" in ds.names:
        filters = [("eclipse", "=", eclipse)],
    else:
        # If there is not an eclipse column, then just use everything
        # This lets you feed gPhoton2's outputs back into itself
        filters = None

    return parquet.read_table(
        source_catalog_file,
        filters=filters,
        columns=['ra', 'dec']
    ).to_pandas()


def load_source_catalog(
    source_catalog_file: Pathlike, eclipse: int
) -> pd.DataFrame:
    source_catalog_file = Path(source_catalog_file)
    if source_catalog_file.suffix == ".csv":
        format_ = "csv"
    elif source_catalog_file.suffix == ".parquet":
        format_ = "parquet"
    else:
        raise ValueError(
            "Couldn't automatically determine source catalog format from the "
            "extension {source_catalog_file.suffix}. Please pass a .csv or "
            ".parquet file with at least the columns 'eclipse', 'ra', 'dec'."
        )
    try:
        if format_ == "csv":
            sources = _load_csv_catalog(source_catalog_file, eclipse)
        else:
            sources = _load_parquet_catalog(source_catalog_file, eclipse)
        sources = sources[['ra', 'dec']]
    except KeyError:
        raise ValueError(
            "The source catalog file must specify source positions in "
            "columns named 'ra' and 'dec' with a reference column named "
            "'eclipse'."
        )
    return sources[~sources.duplicated()].reset_index(drop=True)


def format_source_catalog(source_table, wcs):
    print(f"Using specified catalog of {len(source_table)} sources.")
    positions = np.vstack(
        [
            wcs.wcs_world2pix([position], 1, ra_dec_order=True)
            for position in source_table[["ra", "dec"]].values
        ]
    )
    source_table[["xcentroid", "ycentroid"]] = positions
    return source_table
