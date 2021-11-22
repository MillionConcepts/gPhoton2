"""
temporary home for pipeline utilities to reduce need for expensive and maybe
incompatible imports.
"""
from collections import defaultdict, Collection
from itertools import product
from statistics import mode

import numpy as np
from pyarrow import parquet


def get_parquet_stats(fn, columns, row_group=0):
    group = parquet.read_metadata(fn).row_group(row_group)
    statistics = {}
    for column in group.to_dict()["columns"]:
        if column["path_in_schema"] in columns:
            statistics[column["path_in_schema"]] = column["statistics"]
    return statistics


def table_values(table, columns):
    return np.array([table[column].to_numpy() for column in columns]).T


def where_between(whatever, t0, t1):
    return np.where((whatever >= t0) & (whatever < t1))[0]


def unequally_stepped(array, rtol=1e-5, atol=1e-8):
    diff = array[1:] - array[:-1]
    unequal = np.where(~np.isclose(diff, mode(diff), rtol=rtol, atol=atol))
    return unequal, diff[unequal]


def get_fits_radec(header, endpoints_only=True):
    ranges = {}
    for coord, ix in zip(("ra", "dec"), (1, 2)):
        # explanatory variables
        steps = header[f"NAXIS{ix}"]
        stepsize = header[f"CDELT{ix}"]
        extent = steps * stepsize / 2
        center = header[f"CRVAL{ix}"]
        coord_endpoints = (center - extent, center + extent)
        if endpoints_only is True:
            ranges[coord] = coord_endpoints
        else:
            ranges[coord] = np.arange(*coord_endpoints, stepsize)
    return ranges


class NestingDict(defaultdict):
    """
    shorthand for automatically-nesting dictionary -- i.e.,
    insert a series of keys at any depth into a NestingDict
    and it automatically creates all needed levels above.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = NestingDict

    __repr__ = dict.__repr__


def diff_photonlist_and_movie_coords(movie_radec, photon_radec):
    diffs = {}
    minmax_funcs = {"min": min, "max": max}
    for coord, op in product(("ra", "dec"), ("min", "max")):
        diffs[f"{coord}_{op}"] = abs(
            minmax_funcs[op](movie_radec[coord]) - photon_radec[coord][op]
        )
    return diffs


def listify(thing):
    """Always a list, for things that want lists"""
    if isinstance(thing, Collection):
        if not isinstance(thing, str):
            return list(thing)
    return [thing]
