"""
generic wrappers for manipulating FITS files and associated objects.

TODO: these are somewhat repetitive/redundant, tossed around inconsistently,
 and should be consolidated.
"""

import warnings
from typing import Callable, Sequence, Literal

import astropy.io.fits
import astropy.wcs
import numpy as np
from dustgoggles.scrape import head_file

from gPhoton.coords.wcs import extract_wcs_keywords
from gPhoton.pretty import make_monitors
from gPhoton.reference import crudely_find_library
from gPhoton.types import Pathlike


def get_fits_data(filename, dim=0, verbose=0):
    """
    Reads FITS data. A wrapper for common astropy.io.fits commands.

    :param filename: The name of the FITS file to retrieve the data from.

    :type filename: str

    :param dim: The extension to retrieve the data from, 0=Primary, 1=First
        Extension, etc.

    :type dim: int

    :param verbose: If > 0, print messages to STDOUT.

    :type verbose: int

    :returns: Data instance -- The data from the 'dim' HDU.
    """

    if verbose:
        print("         ", filename)

    hdulist = astropy.io.fits.open(filename, memmap=1)

    data = hdulist[dim].data

    hdulist.close()

    return data
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def get_fits_header(filename):
    """
    Reads a FITS header. A wrapper for common astropy.io.fits commands.

    :param filename: The name of the FITS file to retrieve the header from.

    :type filename: str

    :returns: Header instance -- The header from the primary HDU.
    """

    hdulist = astropy.io.fits.open(filename, memmap=1)

    htab = hdulist[0].header

    hdulist.close()

    return htab
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def get_tbl_data(filename, comment='|'):
    """
    Reads data from a table into a numpy array.

    :param filename: The name of the FITS file to read.

    :type filename: str

    :param comment: The symbol that represents a comment.

    :type comment: str

    :returns: numpy.ndarray -- The table data.
    """

    f = open(filename)
    lines = f.readlines()
    tbl = []

    for line in lines:
        if line[0] != comment:
            strarr = str.split(str(line))
            if len(strarr) > 0:
                tbl.append(strarr)

    return np.array(tbl, dtype='float64')


# ------------------------------------------------------------------------------
def pyfits_open_igzip(fn: str) -> astropy.io.fits.hdu.HDUList:
    """
    open a gzipped FITS file using astropy.io.fits and the ISA-L igzip
    algorithm rather than the slower libdeflate gzip implementation found
    in the python standard library
    """
    from isal import igzip

    # TODO: does this leak the igzip stream handle?
    if fn.endswith("gz"):
        stream = igzip.open(fn)
        return astropy.io.fits.open(stream)
    else:
        return astropy.io.fits.open(fn)


def first_fits_header(path: Pathlike, header_records: int = 1):
    """
    return the first header_records header cards from a FITS file. used for
    skimming metadata from large groups of FITS files
    """
    from isal import igzip

    if str(path).endswith("gz"):
        stream = igzip.open(path)
    else:
        stream = open(path, "rb")
    if "rice" in str(path):
        stream.seek(2880)
    head = head_file(stream, 2880 * header_records)
    stream.close()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # we know we truncated it, thank you
        return astropy.io.fits.open(head)[0].header


def read_wcs_from_fits(*fits_paths: Pathlike) -> tuple[
    Sequence[astropy.io.fits.header.Header], Sequence[astropy.wcs.WCS]
]:
    """
    Construct a WCS object for each FITS file in fits_paths.

    TODO: This function is obsolete and must be rewritten to handle RICE
     compression.
    """
    headers = [first_fits_header(path) for path in fits_paths]
    systems = [astropy.wcs.WCS(header) for header in headers]
    return headers, systems


def get_header(hdul: Sequence, hdu_ix: int, library: str):
    """
    fetch header from either an astropy or fitsio HDU list object
    """
    if library == "fitsio":
        return hdul[hdu_ix].read_header()
    elif library == "astropy":
        return hdul[hdu_ix].header
    raise ValueError(f"don't know {library}")


