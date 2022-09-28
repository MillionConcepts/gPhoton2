"""
metadata and sky-coordinate search functions. find eclipses/visits that
meet specific criteria.
"""
from inspect import getmembers
import sys
import warnings

from pyarrow import parquet

from gPhoton.aspect import TABLE_PATHS


# parquet filters implementing canonical definitions for specific eclipse types
MISLIKE_FILTER = (
    ("legs", "=", 0), ("obstype", "in", ["MIS", "DIS", "GII"])
)
# 'big' eclipses, useful category for testing
COMPLEX_FILTER = (("legs", ">", 0),)


def boundaries_of_a_square(x: float, y: float, size: float):
    """boundaries of a square centered at (x, y) with side length `size`"""
    return x - size / 2, x + size / 2, y - size / 2, y + size / 2


def galex_sky_box(ra: float, dec: float, arcseconds: float):
    """
    return eclipse numbers of all GALEX visits such that ra, dec is within
    `arcseconds` of a box bounding their recorded boresight positions
    """
    deg = arcseconds / 3600
    metadata = parquet.read_table(TABLE_PATHS["metadata"]).to_pandas()
    ra_dmin = metadata['ra_min'] - ra
    ra_dmax = metadata['ra_max'] - ra
    ra_dmin.loc[ra_dmin.abs() > 180] = (360 - ra_dmin.loc[ra_dmin.abs() > 180])
    ra_dmax.loc[ra_dmax.abs() > 180] = (360 - ra_dmax.loc[ra_dmax.abs() > 180])
    dec_dmin = metadata['dec_min'] - dec
    dec_dmax = metadata['dec_max'] - dec
    return metadata.loc[
        ((abs(ra_dmin) < deg) | (abs(ra_dmax) < deg))
        & ((abs(dec_dmin) < deg) | (abs(dec_dmax) < deg))
    ]


def eclipses_near_object(object_name: str, arcseconds: float, verbose=True):
    """
    query SIMBAD for the position of `object`. return eclipse numbers of
    all GALEX visits whose nominal boresight bounds are within `arcseconds`
    of that object.

    this function requires astroquery.
    """
    from astroquery import simbad

    query = simbad.Simbad()
    # simbad does not return position in decimal degrees by default
    query.add_votable_fields('ra(d)', 'dec(d)')
    with warnings.catch_warnings():
        # we would rather handle our own object-not-found reporting
        warnings.simplefilter("ignore")
        result = query.query_objects([object_name])
    if result is None:
        raise ValueError(f"{object_name} not found in SIMBAD database.")
    ra_d, dec_d = float(result['RA_d'][0]), float(result['DEC_d'][0])
    if verbose:
        print(f"{object_name} position: RA {ra_d}, DEC {dec_d}")
    matches = galex_sky_box(ra_d, dec_d, arcseconds)
    if len(matches) == 0:
        raise ValueError(
            f"No eclipses found within {arcseconds} asec of {object_name}."
        )
    if verbose:
        print(
            f"{len(matches)} eclipses found within {arcseconds} asec of "
            f"{object_name}."
        )
    return matches


def filter_galex_eclipses(eclipse_type=None, filters=None, as_pandas=True):
    """
    randomly select a set of GALEX eclipses matching predefined criteria.
    if no arguments are passed, returns all eclipses.
    """
    available_filters = {
        name.lower(): obj
        for (name, obj) in
        getmembers(sys.modules[__name__], lambda obj: isinstance(obj, tuple))
        if name.lower().endswith("filter")
     }
    if (filters is not None) and (eclipse_type is not None):
        warnings.warn(
            "Both eclipse_type and filters passed; ignoring eclipse_type"
        )
    if filters is not None:
        eclipse_filters = filters
    elif eclipse_type is not None:
        if f"{eclipse_type.lower()}_filter" not in available_filters:
            raise ValueError(f"I don't know about {eclipse_type} eclipses.")
        eclipse_filters = available_filters[f"{eclipse_type.lower()}_filter"]
    else:
        eclipse_filters = []
    result = parquet.read_table(
        TABLE_PATHS["metadata"], filters=eclipse_filters
    )
    if as_pandas is True:
        return result.to_pandas()
    return result
