"""
methods for retrieving aspect solution (meta)data from gPhoton 2's combined
aspect solution tables
"""

from pathlib import Path
from typing import Any, Iterable, Literal, cast, get_args

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import parquet

from gPhoton import DEFAULT_ASPECT_DIR
from gPhoton.coords.gnomonic import gnomfwd_simple
from gPhoton.pretty import print_inline

ASPECT_TABLE_TYPE = Literal["aspect", "aspect2", "boresight", "metadata"]
# This odd-looking construct is needed because get_args returns
# tuple[Any, ...]; we're essentially saying 'trust me, in this case
# the elements of the tuple are all appropriate strings'.
# Literal guarantees to deduplicate its args but not necessarily to
# give them back in any particular order
ALL_ASPECT_TABLES = \
    cast(list[ASPECT_TABLE_TYPE], sorted(get_args(ASPECT_TABLE_TYPE)))

def aspect_tables(
    eclipse: None | int,
    tables: None | ASPECT_TABLE_TYPE | Iterable[ASPECT_TABLE_TYPE] =
        ["aspect", "boresight", "metadata"],
    # The type specification for pyarrow filter expressions is too
    # complicated (and probably variable depending on pyarrow version)
    # to replicate.  It might be worth revisiting this after pyarrow
    # itself grows type annotations, if it ever does.
    filters: Any = None,
    aspect_dir: None | str | Path = None,
    **kwargs: Any # additional arguments passed to parquet.read_table
) -> list[pa.Table]:
    """
    fetch full-resolution aspect, per-leg boresight, and/or general metadata
    for a particular eclipse.
    """
    if aspect_dir is None:
        aspect_dir = DEFAULT_ASPECT_DIR
    if not isinstance(aspect_dir, Path):
        aspect_dir = Path(aspect_dir)

    if isinstance(tables, str):
        tables = [tables]
    elif tables is None:
        tables = ALL_ASPECT_TABLES
    else:
        tables = sorted(set(tables))
        if not tables:
            tables = ALL_ASPECT_TABLES

    assert "filters" not in kwargs
    if filters is not None:
        if eclipse is not None:
            # adding a conjunctive clause to filters is too difficult,
            # given the variety of things filters could be, and the
            # lack of any "convert this filter argument to an Expression"
            # utility in pyarrow (that I can find); it'll be easier
            # for the caller
            raise NotImplementedError(
                "sorry, not implemented: automatically combining filters="
                " with eclipse= (note: eclipse=N is shorthand for"
                " filters=[('eclipse', '=', N)])"
            )
        kwargs["filters"] = filters
    elif eclipse is not None:
        kwargs["filters"] = [("eclipse", "=", eclipse)]

    return [
        parquet.read_table(aspect_dir / (table + ".parquet"), **kwargs)
        for table in tables
    ]


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
    eclipse: int,
    aspect: Literal["aspect", "aspect2"],
    verbose: int = 0,
    aspect_dir: None | str | Path = None,
) -> pd.DataFrame:
    """
    loads full-resolution aspect solution + per-leg boresight solution for
    a given eclipse and projects aspect solution to detector coordinates.

    :param eclipse: eclipse for which to load aspect solution
    :param aspect: can designate 2nd aspect table to use, default is aspect
    :param verbose: higher values return more feedback about solution
    :return: dataframe of aspect solution + sky coordinates
    """
    if verbose > 0:
        print_inline("Loading aspect solution from disk...")
    aspect, boresight = [
        tab.to_pandas()
        for tab in aspect_tables(
            eclipse=eclipse,
            tables=(aspect, "boresight"),
            aspect_dir=aspect_dir
        )
    ]
    if "ra" not in aspect.columns:
        print("Using aspect2.parquet")
        aspect = aspect[(aspect['hvnom_nuv'] == 1) | (aspect['hvnom_nuv'] == 1)]\
            .reset_index(drop=True)
        aspect = aspect.rename(columns={"pktime": "time", "ra_acs": "ra",
                                        "dec_acs": "dec", "roll_acs": "roll"})


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
    if verbose > 1:
        trange = [aspect["time"].min(), aspect["time"].max()]
        print(f"trange= ( {trange[0]} , {trange[1]} )")
        print(
            f"RA AVG: {aspect['ra'].mean()}, DEC AVG: {aspect['dec'].mean()}, "
            f"ROLL AVG: {aspect['roll'].mean()}"
        )
    return aspect
