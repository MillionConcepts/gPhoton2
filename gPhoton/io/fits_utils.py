"""generic wrappers for astropy.io.fits (pyfits) methods."""
import warnings

from astropy.io import fits as pyfits
import numpy as np

# ------------------------------------------------------------------------------

from astropy.wcs import wcs
from dustgoggles.scrape import head_file
from isal import igzip


def get_fits_data(filename, dim=0, verbose=0):
    """
    Reads FITS data. A wrapper for common pyfits commands.

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

    hdulist = pyfits.open(filename, memmap=1)

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

    hdulist = pyfits.open(filename, memmap=1)

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
def pyfits_open_igzip(fn):
    # TODO: does this leak the igzip stream handle?
    if fn.endswith("gz"):
        stream = igzip.open(fn)
        return pyfits.open(stream)
    else:
        return pyfits.open(fn)


def first_fits_header(path, header_records=1):
    if str(path).endswith("gz"):
        stream = igzip.open(path)
    else:
        stream = open(path, "rb")
    head = head_file(stream, 2880 * header_records)
    stream.close()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # we know we truncated it, thank you
        return pyfits.open(head)[0].header


def read_wcs_from_fits(*fits_paths):
    headers = [first_fits_header(path) for path in fits_paths]
    systems = [wcs.WCS(header) for header in headers]
    return headers, systems
