"""
methods for retrieving aspect solution (meta)data from gPhoton 2's combined
aspect solution tables
"""

from pathlib import Path
from typing import Mapping, Collection, Literal, Dict, Hashable

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet

from gPhoton import ASPECT_DIR
from gPhoton.coords.gnomonic import gnomfwd_simple
from gPhoton.parquet_utils import parquet_to_ndarrays
from gPhoton.pretty import print_inline

# fully-qualified paths to aspect table files
TABLE_PATHS = {
    "aspect": Path(ASPECT_DIR, "aspect.parquet"),
    "boresight": Path(ASPECT_DIR, "boresight.parquet"),
    "metadata": Path(ASPECT_DIR, "metadata.parquet"),
}


def aspect_tables(
    eclipse: int,
    tables: Collection[
        Literal["aspect", "boresight", "metadata"]
    ] = ("aspect", "boresight", "metadata"),
) -> list[pa.Table]:
    """
    fetch full-resolution aspect, per-leg boresight, and/or general metadata
    for a particular eclipse.
    """
    if tables is None:
        paths = TABLE_PATHS.values()
    else:
        paths = [TABLE_PATHS[table_name] for table_name in tables]
    filters = [("eclipse", "=", eclipse)]
    return [parquet.read_table(path, filters=filters) for path in paths]


def distribute_legs(
    aspect: pd.DataFrame, boresight: pd.DataFrame
) -> pd.DataFrame:
    """
    assign per-leg boresight positions to full-resolution aspect data.
    boresight positions are given in the original GALEX corpus separately
    from full-resolution aspect data. One nominal boresight position exists
    for each "leg" of a visit. For observations with more than one "leg",
    it is important to accurately assign a boresight position to each
    full-resolution aspect data point.
    """
    aspect = pd.DataFrame(aspect)
    boresight = pd.DataFrame(boresight)
    legframes = []
    for leg_ix in boresight.index:
        leg = boresight.loc[leg_ix]
        time_pred = aspect["time"] >= leg["time"]
        if leg_ix != len(boresight) - 1:
            time_pred = time_pred & (
                aspect["time"] < boresight.loc[leg_ix + 1]["time"]
            )
        leg_indices = aspect.loc[time_pred].index
        leg_arrays = {
            'ra0': np.full(len(leg_indices), leg['ra0']),
            'dec0': np.full(len(leg_indices), leg['dec0']),
            'leg': np.full(len(leg_indices), leg_ix, dtype=np.uint8)
        }
        legframes.append(pd.DataFrame(leg_arrays))
    distributed = pd.concat(legframes).reset_index(drop=True)
    return pd.concat([aspect, distributed], axis=1)


def load_aspect_solution(
    eclipse: int, verbose: int = 0
) -> pd.DataFrame:
    """
    loads full-resolution aspect solution + per-leg boresight solution for
    a given eclipse and projects aspect solution to detector coordinates.

    :param eclipse: eclipse for which to load aspect solution
    :param verbose: higher values return more feedback about solution
    :return: dataframe of aspect solution + sky coordinates
    """
    if verbose > 0:
        print_inline("Loading aspect solution from disk...")
    aspect, boresight = [
        tab.to_pandas()
        for tab in aspect_tables(eclipse, ("aspect", "boresight"))
    ]
    if verbose > 1:
        trange = [aspect["time"].min(), aspect["time"].max()]
        print(f"                        trange= ( {trange[0]} , {trange[1]} )")
    if verbose > 1:
        print(
            f"RA AVG: {aspect['ra'].mean()}, DEC AVG: {aspect['dec'].mean()}, "
            f"ROLL AVG: {aspect['roll'].mean()}"
        )
    # This projects the aspect_data solutions onto the MPS field centers.
    if verbose > 0:
        print_inline("Computing aspect_data vectors...")
    aspect = distribute_legs(aspect, boresight)
    aspect['xi'], aspect['eta'] = gnomfwd_simple(
        aspect["ra"].to_numpy(),
        aspect["dec"].to_numpy(),
        aspect["ra0"].to_numpy(),
        aspect["dec0"].to_numpy(),
        -aspect["roll"].to_numpy(),
        1.0 / 36000.0,
        0.0,
    )
    return aspect
