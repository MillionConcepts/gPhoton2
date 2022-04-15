from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from pyarrow import parquet

from gPhoton import ASPECT_DIR
from gPhoton.coords.gnomonic import gnomfwd_simple
from gPhoton.parquet_utils import parquet_to_ndarrays
from gPhoton.pretty import print_inline

TABLE_PATHS = {
    "aspect": Path(ASPECT_DIR, "aspect.parquet"),
    "boresight": Path(ASPECT_DIR, "boresight.parquet"),
    "metadata": Path(ASPECT_DIR, "metadata.parquet"),
}


def aspect_tables(eclipse, tables=None):
    if tables is None:
        paths = TABLE_PATHS.values()
    else:
        paths = [TABLE_PATHS[table_name] for table_name in tables]
    filters = [("eclipse", "=", eclipse)]
    return [parquet.read_table(path, filters=filters) for path in paths]


def distribute_legs(
    aspect: Mapping[str, np.ndarray], boresight: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
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
    loads full-resolution aspect_data solution + per-leg boresight solution,
    projects aspect_data solution to detector coordinates
    :param eclipse: eclipse to load aspect_data solution for
    :param verbose: higher values return more feedback about solution
    :return: dictionary of aspect_data solution + relevant sky coordinates
    """
    if verbose > 0:
        print_inline("Loading aspect_data data from disk...")
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
