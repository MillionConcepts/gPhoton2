from __future__ import absolute_import, division, print_function

# Core and Third Party imports.
import numpy as np
import pandas as pd

# gPhoton imports.
import gPhoton.cal as cal
import gPhoton.curvetools as ct
import gPhoton.galextools as gt
from gPhoton.MCUtils import print_inline


def calibrate_photons(data, band):
    # data = pd.read_csv(photon_file,names=['t','x','y','xa','ya','q','xi','eta','ra','dec','flags'])
    # data = pd.read_hdf(photon_file.replace('.csv','.h5'),'photons')

    print_inline("Applying detector-space calibrations.")
    image = pd.DataFrame()
    flat, _ = cal.flat(band)
    col, row = ct.xieta2colrow(
        np.array(data["xi"]), np.array(data["eta"]), band
    )

    # Use only data that is on the detector.
    ix = np.where((col > 0) & (col < 800) & (row > 0) & (row < 800))
    image["t"] = pd.Series(np.array(data.iloc[ix]["t"]) / 1000.0)
    image["ra"] = pd.Series(np.array(data.iloc[ix]["ra"]))
    image["dec"] = pd.Series(np.array(data.iloc[ix]["dec"]))
    image["flags"] = pd.Series(np.array(data.iloc[ix]["flags"]))
    image["col"] = pd.Series(col[ix])
    image["row"] = pd.Series(row[ix])
    flat = flat[
        np.array(image["col"], dtype="int16"),
        np.array(image["row"], dtype="int16"),
    ]
    image["flat"] = pd.Series(flat)
    scale = gt.compute_flat_scale(np.array(data.iloc[ix]["t"]) / 1000.0, band)
    image["scale"] = pd.Series(scale)
    response = np.array(image["flat"]) * np.array(image["scale"])
    image["response"] = pd.Series(response)
    image["flags"] = pd.Series(np.array(data.iloc[ix]["flags"]))

    # define the hotspot mask
    mask, maskinfo = cal.mask(band)
    npixx, npixxy = mask.shape
    pixsz, detsz = maskinfo["CDELT2"], 1.25
    maskfill = detsz / (npixx * pixsz)

    image["mask"] = pd.Series(
        (
            mask[
                np.array(col[ix], dtype="int64"),
                np.array(row[ix], dtype="int64"),
            ]
            == 0
        )
    )

    return image
