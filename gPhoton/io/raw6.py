"""
.. module:: raw6
   :synopsis: Methods for reading raw GALEX data ("raw6") files
"""
from astropy.io import fits as pyfits
import fitsio
import numpy as np

from gPhoton.calibrate import center_and_scale
from gPhoton.pretty import print_inline


def load_raw6(band, eclipse, raw6file, verbose):
    if verbose > 0:
        print_inline("Loading raw6 file...")
    raw6hdulist = fitsio.FITS(raw6file)
    raw6htab = raw6hdulist[1].read_header()
    nphots = raw6htab["NAXIS2"]
    if verbose > 1:
        print("		" + str(nphots) + " events")
    data = decode_telemetry(band, 0, None, "", eclipse, raw6hdulist)
    raw6hdulist.close()
    return data, nphots


def get_eclipse_from_header(eclipse, raw6file):
    # note that astropy is much faster than fitsio for the specific purpose of
    # skimming a FITS header from a compressed FITS file
    hdulist = pyfits.open(raw6file)
    hdr = hdulist[0].header
    hdulist.close()
    if eclipse and (eclipse != hdr["eclipse"]):  # just a consistency check
        print(
            "Warning: eclipse mismatch {e0} vs. {e1} (header)".format(
                e0=eclipse, e1=hdr["eclipse"]
            )
        )
    eclipse = hdr["eclipse"]
    return eclipse


def decode_telemetry(band, chunkbeg, chunkend, chunkid, eclipse, raw6hdulist):
    data = bitwise_decode_photonbytes(
        band, unpack_raw6(chunkbeg, chunkend, chunkid, raw6hdulist)
    )
    data = center_and_scale(band, data, eclipse)
    data["t"] = data["t"].byteswap().newbyteorder()
    return data


def unpack_raw6(chunkbeg, chunkend, chunkid, raw6hdulist):
    print_inline(chunkid + "Unpacking raw6 data...")
    photonbyte_cols = [f"phb{byte + 1}" for byte in range(5)]
    table_chunk = raw6hdulist[1][chunkbeg:chunkend][["t"] + photonbyte_cols]
    photonbytes_as_short = table_chunk[photonbyte_cols].astype(
        [(col, np.int16) for col in photonbyte_cols]
    )
    photonbytes = {}
    for byte in range(5):
        photonbytes[byte + 1] = photonbytes_as_short[photonbyte_cols[byte]]
    photonbytes["t"] = table_chunk["t"]
    return photonbytes


def bitwise_decode_photonbytes(band, photonbytes):
    print_inline("Band is {band}.".format(band=band))
    data = {"t": photonbytes["t"]}
    # Bitwise "decoding" of the raw6 telemetry.
    data["xb"] = photonbytes[1] >> 5
    data["xamc"] = (
        np.array(((photonbytes[1] & 31) << 7), dtype="int16")
        + np.array(((photonbytes[2] & 254) >> 1), dtype="int16")
        - np.array(((photonbytes[1] & 16) << 8), dtype="int16")
    )
    data["yb"] = ((photonbytes[2] & 1) << 2) + ((photonbytes[3] & 192) >> 6)
    data["yamc"] = (
        np.array(((photonbytes[3] & 63) << 6), dtype="int16")
        + np.array(((photonbytes[4] & 252) >> 2), dtype="int16")
        - np.array(((photonbytes[3] & 32) << 7), dtype="int16")
    )
    data["q"] = ((photonbytes[4] & 3) << 3) + ((photonbytes[5] & 224) >> 5)
    data["xa"] = (
        ((photonbytes[5] & 16) >> 4)
        + ((photonbytes[5] & 3) << 3)
        + ((photonbytes[5] & 12) >> 1)
    )
    return data