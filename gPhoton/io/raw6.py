"""
.. module:: raw6
   :synopsis: Methods for reading GALEX L0 telemetry ("raw6") files
"""
from typing import Optional, Literal, Mapping, cast

import numpy as np
from astropy.io import fits as pyfits
import fitsio

from gPhoton.calibrate import center_and_scale
from gPhoton.pretty import print_inline
from gPhoton.types import Pathlike, GalexBand


def load_raw6(raw6file: Pathlike, verbose: int):
    """
    open and decode a GALEX raw telemetry (.raw6) file and return it as a
    DataFrame. This function replicates mission-standard L0 data processing.
    The dict returned by this function is gPhoton's canonical 'raw photon
    data' structure, used as an input to primary pipeline components
    (particularly photonpipe).

    :param raw6file: path/filename of raw6 file to load
    :param verbose: verbosity level -- higher is more messages
    :return:
    """
    if verbose > 0:
        print_inline("Loading raw6 file...      ")
    raw6hdulist = fitsio.FITS(raw6file)
    raw6header = raw6hdulist[0].read_header()
    raw6htab = raw6hdulist[1].read_header()
    band: GalexBand = "NUV" if raw6header["BAND"] == 1 else "FUV"
    eclipse = raw6header["ECLIPSE"]
    nphots = raw6htab["NAXIS2"]
    if verbose > 1:
        print(f"\n{nphots} events")
    data = decode_telemetry(band, eclipse, raw6hdulist)
    raw6hdulist.close()
    return data, nphots


def get_eclipse_from_header(
    raw6file: Pathlike, eclipse: Optional[int] = None
):
    # note that astropy is much faster than fitsio for the specific purpose of
    # skimming a FITS header from a compressed FITS file
    hdulist = pyfits.open(raw6file)
    hdr = hdulist[0].header
    hdulist.close()
    if eclipse and (eclipse != hdr["eclipse"]):  # just a consistency check
        print(
            f"Warning: eclipse mismatch {eclipse} vs. "
            f"{hdr['eclipse']} (header)"
        )
    eclipse = hdr["eclipse"]
    return eclipse


def decode_telemetry(
    band: GalexBand, eclipse: int, raw6hdulist: fitsio.FITS
):
    """"""
    data = bitwise_decode_photonbytes(unpack_raw6(raw6hdulist))
    data = center_and_scale(band, data, eclipse)
    data["t"] = data["t"].astype("f8")
    return data


def unpack_raw6(raw6hdulist: fitsio.FITS) -> dict[
    Literal["t", 1, 2, 3, 4, 5], np.ndarray
]:
    """
    GALEX raw6 files are stored in a complex binary format. This function
    performs the first step in converting them to physically meaningful
    values: slicing the time and relevant 'photonbyte' columns from the FITS
    table structure, converting them to data types suitable for further
    processing, and placing them in a convenient data structure.
    """
    print_inline("Unpacking raw6 data...")
    photonbyte_cols = [f"phb{byte + 1}" for byte in range(5)]
    table_data = raw6hdulist[1][:][["t"] + photonbyte_cols]
    photonbytes_as_short = table_data[photonbyte_cols].astype(
        [(col, np.int16) for col in photonbyte_cols]
    )
    # mypy can't see that `for byte in range(5)` returns values that
    # satisfy the Literal spec on the return value
    photonbytes: dict[str | int, np.ndarray] = {}
    for byte in range(5):
        photonbytes[byte + 1] = photonbytes_as_short[photonbyte_cols[byte]]
    photonbytes["t"] = table_data["t"]
    return cast(dict[Literal["t", 1, 2, 3, 4, 5], np.ndarray], photonbytes)


def bitwise_decode_photonbytes(
    photonbytes: Mapping[Literal['t', 1, 2, 3, 4, 5], np.ndarray]
) -> dict[Literal["t", "xb", "xamc", "yb", "yamc", "q", "xa"], np.ndarray]:
    """
    GALEX photon events are stored in a packed binary format in the
    'photonbyte' (phb1 - phb6, though phb6 is unused) columns of raw6 files.
    this function 'decodes' these packed bytes into physically meaningful
    values.
    """
    return {
        "t": photonbytes["t"],
        "xb": photonbytes[1] >> 5,
        "xamc": (
            np.array(((photonbytes[1] & 31) << 7), dtype="int16")
            + np.array(((photonbytes[2] & 254) >> 1), dtype="int16")
            - np.array(((photonbytes[1] & 16) << 8), dtype="int16")
        ),
        "yb": ((photonbytes[2] & 1) << 2) + ((photonbytes[3] & 192) >> 6),
        "yamc": (
            np.array(((photonbytes[3] & 63) << 6), dtype="int16")
            + np.array(((photonbytes[4] & 252) >> 2), dtype="int16")
            - np.array(((photonbytes[3] & 32) << 7), dtype="int16")
        ),
        "q": ((photonbytes[4] & 3) << 3) + ((photonbytes[5] & 224) >> 5),
        "xa": (
            ((photonbytes[5] & 16) >> 4)
            + ((photonbytes[5] & 3) << 3)
            + ((photonbytes[5] & 12) >> 1)
        )
    }
