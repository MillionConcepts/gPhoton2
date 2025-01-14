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
import numpy as np
import pandas as pd
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
import scipy.sparse

from gPhoton.pretty import print_inline
from gPhoton.types import Pathlike, GalexBand

from gPhoton.lightcurve.photometry_utils import (mask_for_extended_sources,
                                                 image_segmentation,
                                                 check_point_in_extended)

def count_full_depth_image(
    source_table: pd.DataFrame,
    aperture_size: float,
    image_dict: Mapping[str, np.ndarray],
    system: astropy.wcs.WCS
):
    source_table = source_table.reset_index(drop=True)
    positions = source_table[["xcentroid", "ycentroid"]].to_numpy()
    apertures = CircularAperture(positions, r=aperture_size)
    phot_table = aperture_photometry(image_dict["cnt"], apertures).to_pandas()
    flag_table = aperture_photometry(image_dict["flag"], apertures).to_pandas()
    edge_table = aperture_photometry(image_dict["edge"], apertures).to_pandas()
    source_table = pd.concat(
        [source_table, phot_table[["xcenter", "ycenter", "aperture_sum"]]],
        axis=1,
    )
    source_table["aperture_sum_mask"] = flag_table["aperture_sum"]
    source_table["aperture_sum_edge"] = edge_table["aperture_sum"]
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


def find_sources(
    eclipse: int,
    band: GalexBand,
    datapath: Union[str, Path],
    image_dict,
    wcs,
    source_table: Optional[pd.DataFrame] = None
):
    from gPhoton.coadd import zero_flag_and_edge, flag_and_edge_mask
    # TODO, maybe: pop these into a handler function
    if not image_dict["cnt"].max():
        print(f"{eclipse} appears to contain nothing in {band}.")
        Path(datapath, f"No{band}").touch()
        return f"{eclipse} appears to contain nothing in {band}."
    exptime = image_dict["exptimes"][0]
    if source_table is None:
        print("Extracting sources with DAOFIND.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # for some reason image segmentation still finds sources in the
            # masked cnt image? so we mask twice: once by zeroing and then
            # again by making a bool mask and feeding it to image segmentation
            # TODO: reinvestigate the cause of this
            masked_cnt_image = zero_flag_and_edge(
                image_dict["cnt"],
                image_dict["flag"],
                image_dict["edge"],
                copy=True
            ) / exptime
            masked_cnt_image = masked_cnt_image.astype(np.float32)
            flag_edge_mask = flag_and_edge_mask(
                image_dict["cnt"],
                image_dict["flag"],
                image_dict["edge"])
            image_dict["cnt"] = scipy.sparse.coo_matrix(image_dict["cnt"]) # why is it doing this?
            source_table, segment_map, extended_source_paths, extended_source_cat = \
                get_point_and_extended_sources(masked_cnt_image, band, flag_edge_mask, exptime)
            del masked_cnt_image
            #gc.collect()
            image_dict["cnt"] = image_dict["cnt"].toarray()
        try:
            print(f"Located {len(source_table)} sources.")
        except TypeError:
            print(f"{eclipse} {band} contains no sources.")
            Path(datapath, f"No{band}").touch()
            return None, None
    else:
        print(f"Using specified catalog of {len(source_table)} sources.")
        positions = np.vstack(
            [
                wcs.wcs_world2pix([position], 1, ra_dec_order=True)
                for position in source_table[["ra", "dec"]].values
            ]
        )
        source_table[["xcentroid", "ycentroid"]] = positions
    #return source_table
    # TODO: This breaks if you pass a catalog to `execute_pipeline` because it
    # doesn't know anything about the segment maps.
    try:
        return source_table, segment_map, extended_source_paths, extended_source_cat
    except UnboundLocalError:
        # This is the fallback when a catalog is passed to `execute_pipeline`
        # and therefore no segment map is generated. This is sort of messy and
        # possibly a future TODO.
        return source_table, None, None, None


def get_point_and_extended_sources(
        cnt_image: np.ndarray,
        band: str,
        f_e_mask,
        exposure_time
        ):

    """
    Main function for extracting point and extended sources from an eclipse.
    Image segmentation and point source extraction occurs in a separate function.
    The threshold for being a source in NUV is set at 1.5 times the background
    rms values (2d array) for eclipses over 800 sec, while for NUV under 800s and
    for all FUV it is 3 times bkg rms. Then there is a minimum threshold for FUV
    of the upper quartile of all threshold values over 0.0005.
    Extended source extraction occurs in helper functions.
    """

    print("Masking for extended sources.")
    masks, extended_source_cat = mask_for_extended_sources(cnt_image, band, exposure_time)

    deblended_seg_map, outline_seg_map, seg_sources = image_segmentation(cnt_image, band, f_e_mask, exposure_time)
    print("Checking for extended source overlap with point sources.")
    seg_sources = check_point_in_extended(outline_seg_map, masks, seg_sources)

    #seg_sources.dropna() was old setting
    return seg_sources, deblended_seg_map, masks, extended_source_cat


def extract_frame(frame, apertures):
    if isinstance(frame, scipy.sparse.spmatrix):
        frame = frame.toarray()
    return aperture_photometry(frame, apertures)["aperture_sum"].data


def _extract_photometry_unthreaded(movie, apertures):
    photometry = {}
    for ix, frame in enumerate(movie):
        print_inline(f"extracting frame {ix}")
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

    return parquet.read_table(
        source_catalog_file,
        filters=[('eclipse', '=', eclipse)],
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
