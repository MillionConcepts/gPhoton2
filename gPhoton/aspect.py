"""
methods for retrieving aspect solution (meta)data from gPhoton 2's combined
aspect solution tables
"""

from pathlib import Path
from typing import Mapping, Collection, Literal

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
    ] = ("aspect", "boresight", "metadata")
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
    aspect: Mapping[str, np.ndarray], boresight: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    assign per-leg boresight positions to full-resolution aspect data.
    boresight positions are given in the original GALEX corpus separately
    from full-resolution aspect data. One nominal boresight position exists
    for each "leg" of a visit. For observations with more than one "leg",
    it is important to accurately assign a boresight position to each
    full-resolution aspect data point.
    """
    distributed = pd.DataFrame()
    distributed["time"] = aspect["time"]
    distributed["ra0"] = np.nan
    distributed["dec0"] = np.nan
    boresight = pd.DataFrame(boresight)
    for _, leg in boresight.iterrows():
        leg_indices = distributed.loc[distributed["time"] >= leg["time"]].index
        for axis in ("ra0", "dec0"):
            distributed.loc[leg_indices, axis] = leg[axis]
    return {
        "ra0": distributed["ra0"].values,
        "dec0": distributed["dec0"].values,
    }


def load_aspect_solution(
    eclipse: int, verbose: int = 0
) -> dict[str, np.ndarray]:
    """
    loads full-resolution aspect solution + per-leg boresight solution for
    a given eclipse and projects aspect solution to detector coordinates.
    The dict returned by this function is gPhoton's canonical "prepared
    aspect data" structure, used as input to primary pipeline components
    like photonpipe.

    :param eclipse: eclipse for which to load aspect solution
    :param verbose: higher values return more feedback about solution
    :return: dictionary of aspect solution + relevant sky coordinates
    """
    if verbose > 0:
        print_inline("Loading aspect solution from disk...")
    aspect, boresight = [
        parquet_to_ndarrays(tab, tab.column_names)
        for tab in aspect_tables(eclipse, ("aspect", "boresight"))
    ]
    if verbose > 1:
        trange = [aspect["time"].min(), aspect["time"].max()]
        print(f"			trange= ( {trange[0]} , {trange[1]} )")
    if verbose > 1:
        print(
            f"RA AVG: {aspect['ra'].mean()}, DEC AVG: {aspect['dec'].mean()}, "
            f"ROLL AVG: {aspect['roll'].mean()}"
        )
    # This projects the aspect_data solutions onto the MPS field centers.
    if verbose > 0:
        print_inline("Computing aspect_data vectors...")
    if len(boresight) > 1:
        # legs > 0; must distribute boresight positions correctly
        boresight = distribute_legs(aspect, boresight)
    xi, eta = gnomfwd_simple(
        aspect["ra"],
        aspect["dec"],
        boresight["ra0"],
        boresight["dec0"],
        -aspect["roll"],
        1.0 / 36000.0,
        0.0,
    )
    return aspect | {"xi": xi, "eta": eta}
