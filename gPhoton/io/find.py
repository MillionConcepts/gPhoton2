from gPhoton.io.query import mast_url, BASE_DB, truncate


def exposure_range(band, ra0, dec0, t0=1, t1=10000000000000):
    """
    Find time ranges for which data exists at a given position.

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :param ra0: The right ascension, in degrees, around which to search.

    :type ra0: float

    :param dec0: The declination, in degrees, around which to search.

    :type dec0: float

    :param t0: The minimum time stamp to search for exposure ranges.

    :type t0: long

    :param t1: The maximum time stamp to search for exposure ranges.

    :type t1: long

    :returns: str -- The query to submit to the database.
    """

    return mast_url(
        f"select startTimeRange, endTimeRange from {BASE_DB}.fGetTimeRanges("
        f"{int(t0)},{int(t1)},{repr(float(ra0))},{repr(float(dec0))}) where "
        f"band='{band}'"
    )


def aspect(t0, t1):
    """
    Return aspect information based on a time range.

    :param t0: The minimum time stamp to search.

    :type t0: long

    :param t1: The maximum time stamp to search.

    :type t1: long

    :returns: str -- The query to submit to the database.
    """
    return mast_url(
        f"select eclipse, filename, time, ra, dec, twist, flag, ra0, dec0,"
        f" twist0 from aspect where time >= {truncate(t0)} and time < "
        f"{truncate(t1)} order by time"
    )
