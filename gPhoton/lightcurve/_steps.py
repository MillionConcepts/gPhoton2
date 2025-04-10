"""
methods for generating lightcurves from FITS images/movies, especially those
produced by the gPhoton.moviemaker pipeline. They are principally intended
for use as components of the primary gPhoton.lightcurve pipeline, called as
part of the course of running gPhoton.lightcurve.core.make_lightcurves(), and
may not suitable for independent use.
"""

from multiprocessing import Pool
from pathlib import Path
from collections.abc import Callable, Sequence, Mapping
from typing import Any, TypeVar, cast

import astropy.wcs
import numpy as np
import pandas as pd
from photutils.aperture import CircularAperture, aperture_photometry
import scipy.sparse

from gPhoton.pretty import print_inline
from gPhoton.types import Pathlike, GalexBand, NDArray


FramePM = TypeVar("FramePM", NDArray[np.float64], NDArray[np.uint8])


def count_full_depth_image(
    source_table: pd.DataFrame,
    aperture_size: float,
    image_dict: Mapping[str, NDArray[Any]],
    system: astropy.wcs.WCS
) -> tuple[pd.DataFrame, CircularAperture]:
    source_table = source_table.reset_index(drop=True)
    positions = source_table[["xcentroid", "ycentroid"]].to_numpy()
    apertures = CircularAperture(positions, r=aperture_size)
    phot_table = aperture_photometry(image_dict["cnt"], apertures).to_pandas()
    source_table = pd.concat(
        [source_table, phot_table[["xcenter", "ycenter", "aperture_sum"]]],
        axis=1,
    )
    source_table["artifact_flag"] = bitwise_aperture_photometry(
        image_dict["flag"],
        apertures)

    # TODO: this isn't necessary for specified catalog positions. but
    #  should we do a sanity check?
    if "ra" not in source_table.columns:
        world = [
            system.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()
            for pos in apertures.positions
        ]
        source_table["ra"] = [coord[0] for coord in world]
        source_table["dec"] = [coord[1] for coord in world]
    return source_table, apertures


def bitwise_aperture_photometry(
    artifact_map: NDArray[Any],
    apertures: CircularAperture,
) -> NDArray[np.uint8]:
    """ Run aperture photometry on each bit in artifact flag backplane.
    Non-zero results mean that bit is flagged. Only 4 bits set rn. """
    bitmask_column= np.zeros(len(apertures), dtype=np.uint8)
    # add check to see if there are any bits in the map?
    for bit in range(4):
        bit_image = (artifact_map & (1 << bit)) > 0
        bit_flag_table = aperture_photometry(bit_image.astype(int), apertures).to_pandas()
        bitmask_column |= (bit_flag_table["aperture_sum"].to_numpy() > 0).astype(np.uint8) << bit
    return bitmask_column


def check_empty_image(
    eclipse: int,
    band: GalexBand,
    image_dict: Mapping[str, NDArray[Any]]
) -> str | None:
    if not image_dict["cnt"].max():
        print(f"{eclipse} appears to contain nothing in {band}.")
        # do we want to return a file when the image is empty?
        # realistically it probably means there's something wrong
        # with the observation
        return f"{eclipse} appears to contain nothing in {band}."
    return None


def count_frame(
    frame: NDArray[Any] | scipy.sparse.spmatrix,
    apertures: CircularAperture,
) -> NDArray[np.float64]:
    if isinstance(frame, scipy.sparse.spmatrix):
        frame = frame.toarray()
    return cast(
        NDArray[np.float64],
        aperture_photometry(frame, apertures)["aperture_sum"].data
    )


def flag_frame(
    frame: NDArray[Any] | scipy.sparse.spmatrix,
    apertures: CircularAperture,
) -> NDArray[np.uint8]:
    if isinstance(frame, scipy.sparse.spmatrix):
        frame = frame.toarray()
    return bitwise_aperture_photometry(frame, apertures)


def _extract_photometry_unthreaded(
    movie: Sequence[NDArray[Any] | scipy.sparse.spmatrix],
    apertures: CircularAperture,
    extractor: Callable[[NDArray[Any] | scipy.sparse.spmatrix, CircularAperture], FramePM]
) -> list[FramePM]:
    photometry: list[FramePM] = []
    for frame in movie:
        print_inline(f"extracting frame {len(photometry)}")
        photometry.append(extractor(frame, apertures))
    return photometry


# it's foolish to run this multithreaded unless you _are_ unpacking sparse
# matrices, but I won't stop you.
def _extract_photometry_threaded(
    movie: Sequence[NDArray[Any] | scipy.sparse.spmatrix],
    apertures: CircularAperture,
    extractor: Callable[[NDArray[Any] | scipy.sparse.spmatrix, CircularAperture], FramePM],
    threads: int,
) -> list[FramePM]:
    with Pool(threads) as pool:
        photo_futures = [
            pool.apply_async(extractor, (frame, apertures))
            for frame in movie
        ]
        pool.close()
        pool.join()
        return [result.get() for result in photo_futures]


def extract_photometry(
    movie_dict: Mapping[str, Sequence[NDArray[Any] | scipy.sparse.spmatrix]],
    source_table: pd.DataFrame,
    apertures: CircularAperture,
    threads: int | None
) -> pd.DataFrame:
    if threads is None:
        print("extracting photometry from primary movie")
        counts = _extract_photometry_unthreaded(
            movie_dict["cnt"], apertures, count_frame
        )
        print("extracting photometry from flag map")
        flags = _extract_photometry_unthreaded(
            movie_dict["flag"], apertures, flag_frame
        )
    else:
        print("extracting photometry from primary movie")
        counts = _extract_photometry_threaded(
            movie_dict["cnt"], apertures, count_frame, threads
        )
        print("extracting photometry from flag map")
        flags = _extract_photometry_threaded(
            movie_dict["flag"], apertures, flag_frame, threads
        )

    counts_table = pd.DataFrame.from_dict({
        f"aperture_sum_{ix}": count
        for ix, count in enumerate(counts)
    })
    flags_table = pd.DataFrame.from_dict({
        f"artifact_flag_{ix}": flag
        for ix, flag in enumerate(flags)
    })
    return pd.concat([source_table, counts_table, flags_table], axis=1)


def write_exptime_file(
    expfile: Pathlike,
    movie_dict: Mapping[str, Any]
) -> None:
    exptime_table = pd.DataFrame(
        {
            "expt": movie_dict["exptimes"],
            "t0": [trange[0] for trange in movie_dict["tranges"]],
            "t1": [trange[1] for trange in movie_dict["tranges"]],
        }
    )
    print(f"writing exposure time table to {expfile}")
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

    return cast(pd.DataFrame, parquet.read_table(
        source_catalog_file,
        filters=filters,
        columns=['ra', 'dec']
    ).to_pandas())


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
    except KeyError as e:
        raise ValueError(
            "The source catalog file must specify source positions in "
            "columns named 'ra' and 'dec' with a reference column named "
            "'eclipse'."
        ) from e
    return sources[~sources.duplicated()].reset_index(drop=True)


def format_source_catalog(
    source_table: pd.DataFrame,
    wcs: astropy.wcs.WCS
) -> pd.DataFrame:
    print(f"Using specified catalog of {len(source_table)} sources.")
    positions = np.vstack(
        [
            wcs.wcs_world2pix([position], 1, ra_dec_order=True)
            for position in source_table[["ra", "dec"]].to_numpy()
        ]
    )
    source_table[["xcentroid", "ycentroid"]] = positions
    return source_table
