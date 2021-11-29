"""
.. module:: query
   :synopsis: Methods for communicating with MAST
"""
# NOTE: contains some functions previously -- and perhaps one day again! --
# housed in FileUtils & GQuery
import time

import requests

from gPhoton import TIME_ID
from gPhoton.pretty import print_inline


# The following global variables are used to construct properly-formatted
# queries to the MAST database. Don't change them unless you know what you're
# doing!
BASE_URL = (
    "https://mastcomp.stsci.edu/portal/Mashup/MashupQuery.asmx/Galex"
    "PhotonListQueryTest?query="
)
BASE_DB = "GPFCore.dbo"
MCAT_DB = "GR6Plus7.dbo"
# TIME_ID is a per-execution identifier to aid serverside troubleshooting.
FORMAT_URL = f" -- {TIME_ID}&format=extjs"


def mast_url(sql_string: str) -> str:
    return BASE_URL + sql_string + FORMAT_URL


def has_nan(query):
    """
    Check if there is NaN in a query (or any string) and, if so, raise an
        exception because that probably indicates that something has gone wrong.

    :param query: The query string to check.

    :type query: str
    """

    if "NaN" in query:
        raise RuntimeError("Malformed query: contains NaN values.")

    return


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


def obstype(objid):
    """
    Get the dither pattern type based on the object id.

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


def manage_networked_sql_request(
    query, maxcnt=100, wait=1, timeout=60, verbose=0
):
    """
    Manage calls via `requests` to SQL servers behind HTTP endpoints,
    providing better feedback and making them more robust against network
    errors.

    :param query: The URL containing the query.

    :type query: str

    :param maxcnt: The maximum number of attempts to make before failure.

    :type maxcnt: int

    :param wait: The length of time to wait before attempting the query again.
        Currently a placeholder.

    :type wait: int

    :param timeout: The length of time to wait for the server to send data
        before giving up, specified in seconds.

    :type timeout: float

    :param verbose: If > 0, print additional messages to STDOUT. Higher value
        represents more verbosity.

    :type verbose: int

    :returns: requests.Response or None -- The response from the server. If the
        query does not receive a response, returns None.
    """

    # Keep track of the number of failures.
    cnt = 0

    # This will keep track of whether we've gotten at least one
    # successful response.
    successful_response = False

    while cnt < maxcnt:
        if cnt > 1:
            time.sleep(wait)
        try:
            response = requests.get(query, timeout=timeout)
            successful_response = True
        except KeyboardInterrupt:
            raise
        except requests.exceptions.ConnectTimeout:
            if verbose:
                print("Connection timed out.")
            cnt += 1
            continue
        except requests.exceptions.ConnectionError:
            if verbose:
                print("Domain did not resolve.")
            cnt += 1
            continue
        except Exception as ex:
            if verbose:
                print(f"bad query? {query}: {type(ex)}: {ex}")
            cnt += 1
            continue
        if response.json()["status"] == "EXECUTING":
            if verbose > 1:
                print_inline("EXECUTING")
            cnt = 0
            continue
        elif response.json()["status"] == "COMPLETE":
            if verbose > 1:
                print_inline("COMPLETE")
            break
        elif response.json()["status"] == "ERROR":
            print("ERROR")
            print(f"Unsuccessful query: {query}")
            raise ValueError(response.json()["msg"])
        else:
            print(f"Unknown return: {response.json()['status']}")
            cnt += 1
            continue

    if successful_response is not True:
        response = None

    # noinspection PyUnboundLocalVariable
    return response


# The photon event timestamps are stored in the database at the precision
# level of SQL's BIGINT in order to save space. This is accomplished by
# multiplying the raw timestamps by 1000. This truncates (not rounds) some
# timestamps at the level of 1ms. Most timestamps have a resolution of only
# 5ms except for rare high resolution visits, and even in that case the
# extra precision does not matter for science. For consistency with the
# database, we truncate times at 1ms for queries.
TSCALE = 1000


def truncate(n):
    return str(n * TSCALE).split(".")[0]
