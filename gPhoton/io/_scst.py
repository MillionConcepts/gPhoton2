"""
methods for reading GALEX .scst (raw aspect data) files. These are no longer
core inputs to gPhoton's pipelines; their use has been replaced by the
consolidated tables in gPhoton/aspect. However, they remain useful for
regression tests on and modifications to those tables.
"""

from typing import Sequence

from astropy.io import fits as pyfits
import numpy as np


def load_aspect_files(aspfiles: Sequence[str]):
    """
    Loads a set of aspect_data files into a bunch of arrays.

    :param aspfiles: List of aspect_data files (+paths) to read.

    :type aspfiles: list

    :returns: tuple -- Returns a six-element tuple containing the RA, DEC,
        twist (roll), time, header, and aspect_data flags. Each of these EXCEPT for
        header are returned as np.ndarrays. The header is returned as a dict
        containing the RA, DEC, and roll from the headers of the files.
    """
    ra, dec = np.array([]), np.array([])
    twist, time, aspflags = np.array([]), np.array([]), np.array([])

    header = {"RA": np.array([]), "DEC": np.array([]), "ROLL": np.array([])}
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
