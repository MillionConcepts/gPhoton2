"""
.. module:: io
   :synopsis: Methods for retrieving and reading GALEX aspect ("scst") and raw
   data ("raw6") files, including communication w/MAST.
"""
# NOTE: contains some functions previously -- and perhaps one day again! --
# housed in FileUtils and GQuery.


import os
from typing import Sequence

from astropy.io import fits as pyfits
import numpy as np

# gPhoton imports.
from gPhoton import time_id
from gPhoton.netutils import (
    download_with_progress_bar,
    manage_networked_sql_request,
)


# The following global variables are used in constructing a properly
# formatted query to the MAST database. Don't change them unless you know what
# you're doing!

# All queries from the same _run_ of the photon tools should have identical
# time_id, providing a quick way to troubleshoot issues on the server side.
baseURL = (
    "https://mastcomp.stsci.edu/portal/Mashup/MashupQuery.asmx/Galex"
    "PhotonListQueryTest?query="
)
MCATDB = "GR6Plus7.dbo"
formatURL = " -- " + str(time_id) + "&format=extjs"


def get_raw_paths(eclipse, verbose=0):
    url = raw_data_paths(eclipse)
    if verbose > 1:
        print(url)
    r = manage_networked_sql_request(url)
    out = {"NUV": None, "FUV": None, "scst": None}
    for f in r.json()["data"]["Tables"][0]["Rows"]:
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
        print("Unable to locate {f} file on MAST server.".format(f=ftype))
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
    opath = "{d}{f}".format(d=datadir, f=filename)
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
# TODO: check if this is actually working as it should in current pipeline
def load_aspect(aspfiles: Sequence[str]):
    """
    Loads a set of aspect files into a bunch of arrays.

    :param aspfiles: List of aspect files (+paths) to read.

    :type aspfiles: list

    :returns: tuple -- Returns a six-element tuple containing the RA, DEC,
        twist (roll), time, header, and aspect flags. Each of these EXCEPT for
        header are returned as np.ndarrays. The header is returned as a dict
        containing the RA, DEC, and roll from the headers of the files.
    """
    ra, dec = np.array([]), np.array([])
    twist, time, aspflags = np.array([]), np.array([]), np.array([])

    header = {"RA": [], "DEC": [], "ROLL": []}
    for aspfile in aspfiles:
        print("         ", aspfile)
        hdulist = pyfits.open(aspfile, memmap=1)
        ra = np.append(ra, np.array(hdulist[1].data.field("ra")))
        dec = np.append(dec, np.array(hdulist[1].data.field("dec")))
        twist = np.append(twist, np.array(hdulist[1].data.field("roll")))
        time = np.append(time, np.array(hdulist[1].data.field("t")))
        aspflags = np.append(
            aspflags, np.array(hdulist[1].data.field("status_flag"))
        )
        header["RA"] = np.append(
            header["RA"],
            np.zeros(len(hdulist[1].data.field("ra")))
            + hdulist[0].header["RA_CENT"],
        )
        header["DEC"] = np.append(
            header["DEC"],
            np.zeros(len(hdulist[1].data.field("dec")))
            + hdulist[0].header["DEC_CENT"],
        )
        header["ROLL"] = np.append(
            header["ROLL"],
            np.zeros(len(hdulist[1].data.field("roll")))
            + hdulist[0].header["ROLL"],
        )
        hdulist.close()

    # return arrays sorted by time
    ix = np.argsort(time)
    return ra[ix], dec[ix], twist[ix], time[ix], header, aspflags[ix]


# ------------------------------------------------------------------------------
def web_query_aspect(eclipse, retries=20, quiet=False):
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
    entries = getArray(aspect_ecl(eclipse), retries=retries)
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


def hasNaN(query):
    """
    Check if there is NaN in a query (or any string) and, if so, raise an
        exception because that probably indicates that something has gone wrong.

    :param query: The query string to check.

    :type query: str
    """

    if "NaN" in query:
        raise RuntimeError("Malformed query: contains NaN values.")

    return


def getArray(query, verbose=0, retries=100):
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

    hasNaN(query)

    out = manage_networked_sql_request(query, maxcnt=retries, verbose=verbose)

    if out is not None:
        try:
            out = out.json()["data"]["Tables"][0]["Rows"]
        except:
            print("Failed: {q}".format(q=query))
            raise
        return out
    else:
        print("Failed: {q}".format(q=query))
        raise ValueError(
            "Query never finished on server, run with verbose"
            " turned on for more info."
        )


def obstype(objid):
    """
    Get the dither pattern type based on the object id.

    :param objid: The MCAT Object ID to return the observation type data from.

    :type objid: long

    :returns: str -- The query to submit to the database.
    """

    return (
        "{baseURL}select distinct vpe.mpstype as survey, vpe.tilename,"
        " vpe.photoextractid, vpe.petal, vpe.nlegs, vpe.leg, vpe.eclipse,"
        " vpe.img, vpe.subvis from {MCATDB}.visitPhotoextract as vpe inner"
        " join {MCATDB}.imgrun as iv on vpe.photoextractid=iv.imgrunid"
        " inner join {MCATDB}.visitphotoobjall as p on vpe.photoextractid"
        "=p.photoextractid where p.objid={objid}{formatURL}".format(
            baseURL=baseURL, MCATDB=MCATDB, objid=objid, formatURL=formatURL
        )
    )


def raw_data_paths(eclipse):
    """
    Construct a query that returns a data structure containing the download
    paths

    :param eclipse: GALEX eclipse number.

    :type flag: int

    :returns: str -- The query to submit to the database.
    """
    return "https://mastcomp.stsci.edu/portal/Mashup/MashupQuery.asmx/GalexPhotonListQueryTest?query=spGetRawUrls {ecl}&format=extjs".format(
        ecl=int(eclipse)
    )


def aspect_ecl(eclipse):
    """
    Return the aspect information based upon an eclipse number.

    :param eclipse: The eclipse to return aspect information for.

    :type eclipse: int

    :returns: str -- The query to submit to the database.
    """

    return str(
        baseURL
    ) + "select eclipse, filename, time, ra, dec, twist, flag, ra0, dec0," " twist0 from aspect where eclipse=" + str(
        eclipse
    ) + " order by time" + str(
        formatURL
    )
