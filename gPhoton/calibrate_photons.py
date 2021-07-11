# Core and Third Party imports.
import numpy as np
import pyarrow

# gPhoton imports.
import gPhoton.cal as cal
import gPhoton.curvetools as ct
import gPhoton.galextools as gt
from gPhoton.MCUtils import print_inline


def calibrate_photons(data: pyarrow.Table, band: str):
    col, row, ix, on_detector = select_on_detector_data(band, data)
    col_ix, row_ix = col.astype(np.int16), row.astype(np.int16)
    # define the hotspot mask
    cal_mask, _ = cal.mask(band)
    mask = cal_mask[col_ix, row_ix] == 0
    flat, _ = cal.flat(band)
    flat = flat[col_ix, row_ix]
    del col_ix, row_ix
    # convert time from archival integer units back to floating point
    t = on_detector["t"].to_numpy() / 1000
    scale = gt.compute_flat_scale(t, band)
    response = flat * scale
    return pyarrow.Table.from_arrays(
        [
            t,
            on_detector["ra"],
            on_detector["dec"],
            on_detector["flags"],
            col,
            row,
            scale,
            response,
            flat,
            mask,
        ],
        names=[
            "t",
            "ra",
            "dec",
            "flags",
            "col",
            "row",
            "scale",
            "response",
            "flat",
            "mask",
        ],
    )


def select_on_detector_data(band, data: pyarrow.Table):
    col, row = ct.xieta2colrow(
        data["xi"].to_numpy(), data["eta"].to_numpy(), band
    )
    ix = np.where((col > 0) & (col < 800) & (row > 0) & (row < 800))
    detector_col = col[ix]
    detector_row = row[ix]
    data = pyarrow.compute.take(data, ix[0], boundscheck=False)
    return detector_col, detector_row, ix, data
