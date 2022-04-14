"""
.. module:: mast
   :synopsis: Methods for retrieving GALEX files from MAST. Aspect solution
       retrieval is largely deprecated by consolidated aspect tables, but raw6
       (L0 telemetry) retrieval is not.
"""
# NOTE: contains some functions previously -- and perhaps one day again! --
# housed in FileUtils & GQuery. Some related but conditionally-deprecated
# functions are housed in gPhoton.io._query.mast_query.

import os
import time
from typing import Optional, Literal

import requests

from gPhoton import TIME_ID
from gPhoton.io.netutils import chunked_download
from gPhoton.pretty import print_inline
from gPhoton.types import GalexBand


# The photon event timestamps are stored in MAST's database at the level of
# level of SQL's BIGINT in order to save space. This is accomplished by
# multiplying the raw timestamps by 1000. This truncates (not rounds) some
# timestamps at the level of 1ms. Most timestamps have a resolution of only
# 5ms except for rare high resolution visits, and even in that case the
# extra precision does not matter for science. For consistency with the
# database, we truncate times at 1ms for queries.
TSCALE = 1000

BASE_DB = "GPFCore.dbo"
MCAT_DB = "GR6Plus7.dbo"
# TIME_ID is a per-execution identifier to aid serverside troubleshooting.
FORMAT_URL = f" -- {TIME_ID}&format=extjs"


def truncate(n):
    return str(n * TSCALE).split(".")[0]


def get_raw_paths(eclipse, verbose=0):
    url = raw_data_paths(eclipse)
    if verbose > 1:
        print(url)
    response = manage_networked_sql_request(url)
    out = {"NUV": None, "FUV": None, "scst": None}
    for f in response.json()["data"]["Tables"][0]["Rows"]:
        if (f[1].strip() == "NUV") or (f[1].strip() == "FUV"):
            out[f[1].strip()] = f[2]
        elif f[1].strip() == "BOTH":  # misnamed scst path
            out["scst"] = f[2]
    return out


def download_data(
    eclipse,
    ftype: Literal["raw6", "scst"],
    band: Optional[GalexBand] = None,
    datadir="./",
    verbose=0,
):
    urls = get_raw_paths(eclipse, verbose=verbose)
    if ftype not in ["raw6", "scst"]:
        raise ValueError("ftype must be either raw6 or scst")
    if ftype == "raw6" and band not in ["NUV", "FUV"]:
        raise ValueError("band must be either NUV or FUV")
    url = urls[band] if ftype == "raw6" else urls["scst"]
    if not url:
        print(f"Unable to locate {ftype} file on MAST server.")
        return None
    if url[-6:] == ".gz.gz":  # Handling a very rare mislabeling of the URL.
        url = url[:-3]
    if verbose > 1:
        print(url)
    if not datadir:
        datadir = "."
    if datadir and datadir[-1] != "/":
        datadir += "/"
    filename = url.split("/")[-1]
    opath = f"{datadir}{filename}"
    if os.path.isfile(opath):
        print(f"Using {ftype} file already at {os.path.abspath(opath)}")
    else:
        # TODO: investigate handling options for different error cases
        try:
            chunked_download(url, opath, render_bar=verbose > 0)
        except Exception as ex:
            print(f"Unable to download data from {url}: {type(ex)}: {ex}")
            raise
    return opath


# -----------------------------------------------------------------------------
def raw_data_paths(eclipse):
    """
    Construct a query that returns a data structure containing MAST
    download paths

    :param eclipse: GALEX eclipse number.

    :type eclipse: int

    :returns: str -- The query to submit to the database.
    """
    return mast_url(f"spGetRawUrls {eclipse}")


def retrieve_raw6(eclipse, band, outbase):
    """retrieve raw6 (L0 telemetry) file from MAST."""
    raw6file = download_data(
        eclipse, "raw6", band, datadir=os.path.dirname(outbase)
    )
    if raw6file is None:
        raise ValueError("Unable to retrieve raw6 file for this eclipse.")
    return raw6file


# The following global variables are used to construct properly-formatted
# queries to the MAST database. Don't change them unless you know what you're
# doing!
BASE_URL = (
    "https://mastcomp.stsci.edu/portal/Mashup/MashupQuery.asmx/Galex"
    "PhotonListQueryTest?query="
)


def mast_url(sql_string: str) -> str:
    return BASE_URL + sql_string + FORMAT_URL


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
