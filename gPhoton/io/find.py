from gPhoton.io.query import mast_url, BASE_DB, truncate


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


def obstype_from_t(t: float):
    """
    Get active dither pattern type from a time stamp.
    """
    return mast_url(f"SELECT * from {BASE_DB}.fGetLegObsType({truncate(t)})")


def obstype_from_eclipse(eclipse):
    try:
        t = fu.web_query_aspect(eclipse,quiet=True)[3]
        obsdata = gq.getArray(gq.obstype_from_t(t[int(len(t)/2)]))[0]
        obstype = obsdata[0]
        nlegs = obsdata[4]
        print(
            "e{eclipse} is an {obstype} mode observation w/ {n} legs.".format(
                eclipse=eclipse, obstype=obstype, n=nlegs
            )
        )
        return obstype, len(t), nlegs
    except (IndexError, TypeError, ValueError):
        return "NoData", 0, 0