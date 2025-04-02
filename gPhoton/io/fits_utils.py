"""
generic wrappers for manipulating FITS files and associated objects.

TODO: these are somewhat repetitive/redundant, tossed around inconsistently,
 and should be consolidated.
"""

import warnings
from collections.abc import Sequence
from io import IOBase
from typing import Any, cast

import astropy.io.fits
import astropy.wcs
import fitsio

import numpy as np
from dustgoggles.scrape import head_file

from gPhoton.coords.wcs import extract_wcs_keywords
from gPhoton.types import NDArray, Pathlike

class AgnosticHDU:
    """
    wrapper class to smooth over some of the differences between
    the astropy and fitsio APIs for HDU objects
    """

    def __init__(
        self,
        hdu: astropy.io.fits.hdu.base._BaseHDU | fitsio.hdu.base.HDUBase
    ):
        self._hdu = hdu

    @property
    def header(self) -> astropy.io.fits.header.Header | fitsio.FITSHDR:
        if hasattr(self._hdu, "header"):
            return self._hdu.header
        return self._hdu.read_header()

    @property
    def data(self) -> NDArray[Any]:
        if hasattr(self._hdu, "data"):
            blob = self._hdu.data
        else:
            blob = self._hdu.read()
        return cast(NDArray[Any], blob)

    @property
    def shape(self) -> tuple[int]:
        if hasattr(self._hdu, "shape"):
            # astropy returns a tuple, but doesn't have type annotations
            return cast(tuple[int], self._hdu.shape)
        else:
            # fitsio returns a list
            return tuple(self._hdu.get_dims())

    @property
    def wcs_(self) -> astropy.wcs.WCS:
        return astropy.wcs.WCS(extract_wcs_keywords(self.header))

    def __getitem__(self, item: int) -> Any:
        return self.data[item]

    def __getattribute__(self, attr: str) -> Any:
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            pass
        return self._hdu.__getattribute__(attr)

    # TODO, maybe: attempt to deal with weird array shape differences
    #  between astropy and fitsio in response to the same slices

    def __str__(self) -> str:
        return cast(str, self._hdu.__str__())

    def __repr__(self) -> str:
        return cast(str, self._hdu.__repr__())


class AgnosticHDUL:
    """
    wrapper class to smooth over some of the differences between
    the astropy and fitsio APIs for open FITS files
    """

    def __init__(self, hdul: astropy.io.fits.hdu.HDUList | fitsio.FITS):
        self._hdul = hdul

    def __getitem__(self, item: int) -> AgnosticHDU:
        return AgnosticHDU(self._hdul[item])

    def __str__(self) -> str:
        return cast(str, self._hdul.__str__())

    def __repr__(self) -> str:
        return cast(str, self._hdul.__repr__())

    def __len__(self) -> int:
        return cast(int, self._hdul.__len__())

    def __enter__(self) -> "AgnosticHDUL":
        self._hdul.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._hdul.__exit__(*args)


def get_fits_data(
    filename: Pathlike,
    dim: int = 0,
    verbose: int = 0,
) -> NDArray[Any]:
    """
    Reads FITS data. A wrapper for common astropy.io.fits commands.

    :param filename: The name of the FITS file to retrieve the data from.

    :type filename: Pathlike

    :param dim: The extension to retrieve the data from, 0=Primary, 1=First
        Extension, etc.

    :type dim: int

    :param verbose: If > 0, print messages to STDOUT.

    :type verbose: int

    :returns: Data instance -- The data from the 'dim' HDU.
    """

    if verbose:
        print("         ", filename)

    with pyfits_open_igzip(filename, memmap=1) as hdulist:
        return cast(NDArray[Any], hdulist[dim].data)


def get_fits_header(filename: Pathlike) -> astropy.io.fits.header.Header:
    """
    Reads a FITS header. A wrapper for common astropy.io.fits commands.

    :param filename: The name of the FITS file to retrieve the header from.

    :type filename: str

    :returns: Header instance -- The header from the primary HDU.
    """

    with pyfits_open_igzip(filename, memmap=1) as hdulist:
        return hdulist[0].header


def get_tbl_data(filename: Pathlike, comment: str = '|') -> NDArray[np.float64]:
    """
    Reads data from a table into a numpy array.

    :param filename: The name of the FITS file to read.

    :type filename: Pathlike

    :param comment: The symbol that represents a comment.

    :type comment: str

    :returns: numpy.ndarray -- The table data.
    """
    with open(filename, "rt") as fp:
        tbl = []
        for line in fp:
            if line and line[0] != comment:
                fields = line.split()
                if fields:
                    tbl.append(fields)
        return np.array(tbl, dtype='float64')


def pyfits_open_igzip(
    fn: Pathlike,
    *args: Any,
    **kwargs: Any,
) -> astropy.io.fits.hdu.HDUList:
    """
    open a gzipped FITS file using astropy.io.fits and the ISA-L igzip
    algorithm rather than the slower libdeflate gzip implementation found
    in the python standard library

    :param filename: The name of the FITS file to read.

    :type filename: Pathlike

    :param args: Additional positional arguments to pass to astropy.io.fits.open

    :param kwargs: Additional keyword arguments to pass to astropy.io.fits.open
    """
    from isal import igzip

    if str(fn).endswith("gz"):
        stream = igzip.open(fn) # type: ignore
        hdul = astropy.io.fits.open(stream, *args, **kwargs)
    else:
        hdul = astropy.io.fits.open(fn, *args, **kwargs)
    return cast(astropy.io.fits.hdu.HDUList, hdul)


def first_fits_header(
    path: Pathlike,
    header_records: int = 1
) -> astropy.io.fits.header.Header:
    """
    return the first header_records header cards from a FITS file. used for
    skimming metadata from large groups of FITS files
    """
    from isal import igzip
    def open_fits(path: Pathlike) -> IOBase:
        if str(path).endswith("gz"):
            return cast(IOBase, igzip.open(path)) # type: ignore
        else:
            return open(path, "rb")
    with open_fits(path) as stream:
        if "rice" in str(path):
            stream.seek(2880)
            head = head_file(stream, 2880 * header_records)

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
