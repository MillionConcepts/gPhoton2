"""
.. module:: mast
   :synopsis: Methods for retrieving GALEX files from MAST. Aspect solution
       retrieval is largely deprecated by consolidated aspect tables, but raw6
       (L0 telemetry) retrieval is not. Most methods of this module operate
       via live queries to MAST resources and therefore require internet
       access.
"""
# NOTE: contains some functions previously -- and perhaps one day again! --
# housed in FileUtils & GQuery. Some related but conditionally-deprecated
# functions are housed in gPhoton.io._query.mast_query.

import os
import time
from pathlib import Path
from typing import Optional, Literal

import requests

from gPhoton import TIME_ID
from gPhoton.io.netutils import chunked_download
from gPhoton.pretty import print_inline
from gPhoton.types import GalexBand, Pathlike

# The following global variables are used to construct properly-formatted
# queries to the MAST database. Don't change them unless you know what you're
# doing!

BASE_URL = (
    "https://mast.stsci.edu/portal/Mashup/MashupQuery.asmx/Galex"
    "PhotonListQueryTest?query="
)
TSCALE = 1000
BASE_DB = "GPFCore.dbo"
MCAT_DB = "GR6Plus7.dbo"
FORMAT_URL = f" -- {TIME_ID}&format=extjs"


def truncate(n: float):
    """
    The photon event timestamps are stored in MAST's database at the level of
    SQL's BIGINT in order to save space. This is accomplished by
    multiplying the raw timestamps by 1000. This truncates (not rounds) some
    timestamps at the level of 1ms. Most timestamps have a resolution of only
    5ms except for rare high resolution visits, and even in that case the
    extra precision does not matter for science. For consistency with the
    database, we truncate times at 1ms for queries.
    """
    return str(n * TSCALE).split(".")[0]


def get_raw_paths(eclipse: int, verbose: int = 0) -> dict[str, Optional[str]]:
    """
    query MAST for URLs to the NUV and FUV raw6 (L0 telemetry) and scst
    (aspect solution) files associated with a particular eclipse.
    """
    url = raw_data_paths(eclipse)
    if verbose > 1:
        print(url)
    response = manage_networked_sql_request(url)
    if response is None:
        raise RuntimeError(f"gave up trying to retrieve {url}")
    out: dict[str, Optional[str]] = {"NUV": None, "FUV": None, "scst": None}
    for f in response.json()["data"]["Tables"][0]["Rows"]:
        band = f[1].strip()
        if band in ("NUV", "FUV", "scst"):
            out[band] = f[2]
        elif band == "BOTH":  # misnamed scst path
            out["scst"] = f[2]
        else:
            if verbose > 1:
                print(f"ignoring unrecognized band {band}")
    return out


def download_data(
    eclipse: int,
    ftype: Literal["raw6", "scst"],
    band: Optional[GalexBand] = None,
    datadir: Pathlike = ".",
    verbose: int = 0,
) -> Path | None:
    """
    download a raw6 (L0 telemetry) or scst (aspect solution) file for a given
    eclipse from MAST to datadir.
    """
    urls = get_raw_paths(eclipse, verbose=verbose)
    if ftype == "raw6":
        if band not in ("NUV", "FUV"):
            raise ValueError("band must be either NUV or FUV")
        url = urls[band]
    elif ftype == "scst":
        url = urls["scst"]
    else:
        raise ValueError("ftype must be either raw6 or scst")
    if not url:
        print(f"Unable to locate {ftype} file on MAST server.")
        return None
    if url[-6:] == ".gz.gz":  # Handling a very rare mislabeling of the URL.
        url = url[:-3]
    if verbose > 1:
        print(url)
    if not datadir:
        datadir = "."
    if not isinstance(datadir, Path):
        datadir = Path(datadir)
    filename = url.split("/")[-1]
    opath = datadir / filename
    if os.path.isfile(opath):
        print(f"Using {ftype} file already at {opath.resolve()}")
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
    Construct a URL that will retrieve a data structure containing MAST
    download paths for raw6 and scst files.

    :param eclipse: GALEX eclipse number.

    :type eclipse: int

    :returns: str -- URL to submit this query to the database.
    """
    return mast_url(f"spGetRawUrls {eclipse}")


def retrieve_raw6(eclipse: int, band: GalexBand, outbase: Pathlike) -> Path:
    """retrieve raw6 (L0 telemetry) file from MAST and save it to outbase."""
    raw6file = download_data(
        eclipse, "raw6", band, datadir=os.path.dirname(outbase),
    )
    if raw6file is None:
        raise ValueError("Unable to retrieve raw6 file for this eclipse.")
    return raw6file


def mast_url(sql_string: str) -> str:
    """
    given a string containing a SQL query, construct a URL that will feed
    that query to MAST's GALEX database.
    """
    return f"{BASE_URL}{sql_string}{FORMAT_URL}"


def manage_networked_sql_request(
    query: str,
    maxcnt: int = 100,
    wait: int = 1,
    timeout: int = 60,
    verbose: int = 0
) -> Optional[requests.Response]:
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

    while cnt < maxcnt:
        if cnt > 1:
            time.sleep(wait)

        try:
            response = requests.get(query, timeout=timeout)
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

        json = response.json()
        status = json.get("status", "<status missing>")

        if status == "COMPLETE":
            if verbose > 1:
                print_inline("COMPLETE")
            return response

        if status == "EXECUTING":
            if verbose > 1:
                print_inline("EXECUTING")
            cnt = 0
            continue

        if status == "ERROR":
            print("ERROR")
            print(f"Unsuccessful query: {query}")
            raise ValueError(json["msg"])

        print(f"Unknown response status {status}, retrying...")
        cnt += 1

    return None
