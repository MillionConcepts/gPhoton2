"""
holding module for mostly-deprecated MAST query functions, still desired by
some external legacy code. We do not recommend using them for anything at all.
"""
import numpy as np

from gPhoton.io.mast import (
    manage_networked_sql_request, mast_url, download_data, MCAT_DB
)
from gPhoton.types import Pathlike


def has_nan(query):
    """
    Check if there is NaN in a query (or any string) and, if so, raise an
        exception -- NaNs probably indicate that something has gone wrong.

    :param query: The query string to check.

    :type query: str
    """

    if "NaN" in query:
        raise RuntimeError("Malformed query: contains NaN values.")

    return


# not presently in use
def get_array(query, verbose=0, retries=100):
    """
    Manage a database call which returns an array of values.

    :param query: The query to run.

    :type query: str

    :param verbose: Verbosity level, a value of 0 is minimum verbosity.

    :type verbose: int

    :param retries: Number of query retries to attempt before giving up.

    :type retries: int

    :returns: requests.Response or None -- The response from the server. If the
        query does not receive a response, returns None.
    """

    has_nan(query)

    out = manage_networked_sql_request(query, maxcnt=retries, verbose=verbose)

    if out is not None:
        try:
            out = out.json()["data"]["Tables"][0]["Rows"]
        except Exception as ex:
            print(f"Failed: {query}: {type(ex)}: {ex}")
            raise
        return out
    else:
        print(f"Failed: {query}")
        raise ValueError(
            "Query never finished on server, run with verbose > 0 for more "
            "info."
        )


# not presently in use
def obstype(objid):
    """
    Get the dither pattern type based on the object id. Not currently in use.

    :param objid: The MCAT Object ID to return the observation type data from.

    :type objid: long

    :returns: str -- The query to submit to the database.
    """
    return mast_url(
        f"select distinct vpe.mpstype as survey, vpe.tilename,"
        f" vpe.photoextractid, vpe.petal, vpe.nlegs, vpe.leg, vpe.eclipse,"
        f" vpe.img, vpe.subvis from {MCAT_DB}.visitPhotoextract as vpe inner"
        f" join {MCAT_DB}.imgrun as iv on vpe.photoextractid=iv.imgrunid"
        f" inner join {MCAT_DB}.visitphotoobjall as p on vpe.photoextractid"
        f"=p.photoextractid where p.objid={objid}"
    )


# ------------------------------------------------------------------------------
def retrieve_aspect(eclipse, retries=20, quiet=False):
    """
    Grabs and organizes aspect data from MAST databases based on eclipse.
    Use in pipeline is deprecated by consolidated aspect tables, but may be
    necessary to validate or regenerate those tables.

    :param eclipse: Eclipse number of aspect data.

    :type eclipse: int

    :param retries: The number of times to retry a query before giving up.

    :type retries: int

    :returns: tuple -- Returns a six-element tuple containing the RA, DEC,
        twist (roll), time, header, and aspect_data flags. Each of these EXCEPT for
        header, are returned as numpy.ndarrays. The header is returned as a dict
        containing the RA, DEC, and roll from the headers of the aspec files in
        numpy.ndarrays.
    """

    if not quiet:
        print("Attempting to query MAST database for aspect_data records.")
    entries = get_array(aspect_ecl(eclipse), retries=retries)
    n = len(entries)
    if not quiet:
        print("		Located " + str(n) + " aspect_data entries.")
        if not n:
            print("No aspect_data entries for eclipse " + str(eclipse))
            return
    ra, dec, twist, time, flags = [], [], [], [], []
    ra0, dec0, twist0 = [], [], []
    for i in range(n):
        # The times are *1000 in the database to integerify
        time.append(float(entries[i][2]) / 1000.0)
        ra.append(float(entries[i][3]))
        dec.append(float(entries[i][4]))
        twist.append(float(entries[i][5]))
        flags.append(float(entries[i][6]))
        ra0.append(float(entries[i][7]))
        dec0.append(float(entries[i][8]))
        twist0.append(float(entries[i][9]))

    # Need to sort the output so that it is time ordered before returning.
    # Although it should already be ordered by time because that is requested
    #  in the SQL query above. If this is time consuming, remove it.
    ix = np.argsort(np.array(time))
    header = {
        "RA": np.array(ra0)[ix],
        "DEC": np.array(dec0)[ix],
        "ROLL": np.array(twist0)[ix],
    }

    return (
        np.array(ra)[ix],
        np.array(dec)[ix],
        np.array(twist)[ix],
        np.array(time)[ix],
        header,
        np.array(flags)[ix],
    )


def aspect_ecl(eclipse):
    """
    Return aspect information for a given eclipse.

    :param eclipse: Eclipse number of desired aspect data.

    :type eclipse: int

    :returns: str -- url containing query to the database.
    """
    return mast_url(
        f"select eclipse, filename, time, ra, dec, twist, flag, ra0, dec0, "
        f"twist0 from aspect_data where eclipse={eclipse} order by time"
    )


def retrieve_scstfile(
    eclipse: int, outbase: Pathlike = ".", verbose: int = 0
) -> str:
    """retrieve SCST (aspect) file from MAST. Not currently in use."""
    scstfile = download_data(eclipse, "scst", datadir=outbase, verbose=verbose)
    if scstfile is None:
        raise ValueError("Unable to retrieve SCST file for this eclipse.")
    return scstfile