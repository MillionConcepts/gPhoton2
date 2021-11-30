"""
.. module:: fetch
   :synopsis: Methods for retrieving GALEX files from MAST
"""
# NOTE: contains some functions previously -- and perhaps one day again! --
# housed in FileUtils & GQuery

import os
from typing import Optional

import numpy as np

from gPhoton.io.netutils import download_with_progress_bar
from gPhoton.io.query import mast_url, get_array, manage_networked_sql_request
from gPhoton.types import GalexBand, Pathlike


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


def download_data(eclipse, band, ftype, datadir="./", verbose=0):
    urls = get_raw_paths(eclipse, verbose=verbose)
    if ftype not in ["raw6", "scst"]:
        raise ValueError("ftype must be either raw6 or scst")
    if band not in ["NUV", "FUV"]:
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
            download_with_progress_bar(url, opath)
        except Exception as ex:
            print(f"Unable to download data from {url}: {type(ex)}: {ex}")
            raise
    return opath


# ------------------------------------------------------------------------------
def retrieve_aspect(eclipse, retries=20, quiet=False):
    """
    Grabs the aspect data from MAST databases based on eclipse.

    :param eclipse: The number of the eclipse to retrieve aspect files for.

    :type eclipse: int

    :param retries: The number of times to retry a query before giving up.

    :type retries: int

    :returns: tuple -- Returns a six-element tuple containing the RA, DEC,
        twist (roll), time, header, and aspect flags. Each of these EXCEPT for
        header, are returned as numpy.ndarrays. The header is returned as a dict
        containing the RA, DEC, and roll from the headers of the aspec files in
        numpy.ndarrays.
    """

    if not quiet:
        print("Attempting to query MAST database for aspect records.")
    entries = get_array(aspect_ecl(eclipse), retries=retries)
    n = len(entries)
    if not quiet:
        print("		Located " + str(n) + " aspect entries.")
        if not n:
            print("No aspect entries for eclipse " + str(eclipse))
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


# ------------------------------------------------------------------------------


def raw_data_paths(eclipse):
    """
    Construct a query that returns a data structure containing the download
    paths

    :param eclipse: GALEX eclipse number.

    :type eclipse: int

    :returns: str -- The query to submit to the database.
    """
    return mast_url(f"spGetRawUrls {eclipse}")


def aspect_ecl(eclipse):
    """
    Return the aspect information based upon an eclipse number.

    :param eclipse: The eclipse to return aspect information for.

    :type eclipse: int

    :returns: str -- url containing query to the database.
    """
    return mast_url(
        f"select eclipse, filename, time, ra, dec, twist, flag, ra0, dec0, "
        f"twist0 from aspect where eclipse={eclipse} order by time"
    )


def retrieve_scstfile(
    band: GalexBand,
    eclipse: Optional[int],
    outbase: Pathlike,
    scstfile: Optional[Pathlike],
) -> str:
    if not scstfile:
        if not eclipse:
            raise ValueError("Must specify eclipse if no scstfile.")
        else:
            scstfile = download_data(
                eclipse, band, "scst", datadir=os.path.dirname(outbase)
            )
        if scstfile is None:
            raise ValueError("Unable to retrieve SCST file for this eclipse.")
    return scstfile


def retrieve_raw6(eclipse, band, outbase):
    if not eclipse:
        raise ValueError("Must specify eclipse if no raw6file.")
    else:
        raw6file = download_data(
            eclipse, band, "raw6", datadir=os.path.dirname(outbase)
        )
        if raw6file is None:
            raise ValueError("Unable to retrieve raw6 file for this eclipse.")
    return raw6file
