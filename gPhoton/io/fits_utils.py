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


def logged_fits_initializer(
    path: Pathlike,
    loader: Callable,
    hdu_indices: Sequence[int],
    get_wcs: bool = False,
    get_handles: bool = False,
    verbose: int = 0,
    logged: bool = True,
    astropy_handle_attribute: str = "data"
):
    """
    initialize a FITS object using a passed 'loader' -- probably
    astropy.io.fits.open, a constructor for fitsio.FITS, or a wrapped
    version of one of those. optionally also meticulously record time and
    network transfer involved at all stages of its initialization. At
    present, this function is primarily used for benchmarking.
    """
    # initialize fits HDU list object and read selected HDU's header
    stat, note = make_monitors(fake=not logged)
    hdul = loader(path)
    note(f"init fits object,{path},{stat()}", verbose > 0)
    library = crudely_find_library(loader)
    header = get_header(hdul, hdu_indices[0], library)
    note(f"got header,{path},{stat()}", verbose > 1)
    # TODO: this is a slightly weird hack to revert astropy's automatic
    #  translation of some FITS header values to astropy types. There might
    #  be a cleaner way to do this.
    if library == "astropy":
        output_header = {}
        for k, v in header.items():
            if isinstance(v, (str, float, int)):
                output_header[k] = v
            else:
                output_header[k] = str(v)
        header = output_header
    output = {'header': header, 'stat': stat}
    file_attr = next(filter(lambda attr: "filename" in attr, dir(hdul)))
    output['path'] = getattr(hdul, file_attr)
    if isinstance(output['path'], Callable):
        output['path'] = output['path']()
    if get_handles is True:
        # initialize selected HDU object and get its data 'handles'
        output["handles"] = [hdul[hdu_ix] for hdu_ix in hdu_indices]
        # fitsio exposes slices on HDU data by assigning a __getitem__ method
        # directly to its HDU objects. astropy instead assigns __getitem__
        # methods to attributes of HDU objects, so here we return an attribute
        # rather than the HDU itself as the "handle". by default this is
        # "data", but there are other attributes, notably "section", that also
        # offer data access
        if library == "astropy":
            output["handles"] = [
                getattr(h, astropy_handle_attribute) for h in output["handles"]
            ]
        # TODO: section case
        note(f"got data handles,{path},{stat()}", loud=verbose > 1)
    if get_wcs is True:
        output['wcs'] = astropy.wcs.WCS(extract_wcs_keywords(header))
        note(f"initialized wcs,{path},{stat()}", loud=verbose > 1)
    output['log'] = note(None, eject=True)
    return output
