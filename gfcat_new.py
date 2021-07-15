import os

import fast_histogram as fh
import numpy as np
import pandas as pd
from pyarrow import parquet
from memory_profiler import profile

from gPhoton.MCUtils import NestingDict
from gfcat.gfcat_utils import (
    obstype_from_eclipse,
    make_photometry,
    download_raw6,
    make_wcs,
    compute_exptime,
)
import gPhoton.constants as c
from gPhoton.PhotonPipe import photonpipe
from gPhoton import cal
from gPhoton import MCUtils as mc


def get_parquet_stats(fn, columns, row_group=0):
    group = parquet.read_metadata(fn).row_group(row_group)
    statistics = {}
    for column in group.to_dict()["columns"]:
        if column["path_in_schema"] in columns:
            statistics[column["path_in_schema"]] = column["statistics"]
    return statistics


# @profile
def optimize_wcs(radec):
    real_ra = radec[:, 0][~np.isnan(radec[:, 0])]
    real_dec = radec[:, 1][~np.isnan(radec[:, 1])]
    ra_range = real_ra.min(), real_ra.max()
    dec_range = real_dec.min(), real_dec.max()
    center_skypos = (np.mean(ra_range), np.mean(dec_range))
    imsz = (
        int(np.ceil((ra_range[1] - ra_range[0]) / c.DEGPERPIXEL)),
        int(np.ceil((dec_range[1] - dec_range[0]) / c.DEGPERPIXEL)),
    )
    return make_wcs(center_skypos, imsz=imsz, pixsz=c.DEGPERPIXEL)


def make_frame(foc, weights, wcs):
    imsz = (
        int((wcs.wcs.crpix[0] - 0.5) * 2),
        int((wcs.wcs.crpix[1] - 0.5) * 2),
    )
    frame = fh.histogram2d(
        foc[:, 1] - 0.5,
        foc[:, 0] - 0.5,
        bins=imsz,
        range=([[0, imsz[0]], [0, imsz[1]]]),
        weights=weights,
    )
    return frame


def optimize_compute_shutter(events, trange, shutgap=0.05):
    t0 = events[:, 0]
    flags = events[:, 1]
    ix = np.where((t0 >= trange[0]) & (t0 < trange[1]) & (flags == 0))
    t = np.sort([trange[0]] + list(np.unique(t0[ix])) + [trange[1]])
    ix = np.where(t[1:] - t[:-1] >= shutgap)
    shutter = np.array(t[1:] - t[:-1])[ix].sum()
    return shutter


def optimize_compute_exptime(events, band, trange):
    rawexpt = trange[1] - trange[0]
    times = events[:, 0]
    tix = np.where((times >= trange[0]) & (times < trange[1]))
    shutter = optimize_compute_shutter(events, trange)

    # Calculate deadtime
    model = {
        "NUV": [-0.000434730599193, 77.217817988],
        "FUV": [-0.000408075976406, 76.3000943221],
    }

    # TODO: THIS IS A CORRECTION THAT NEEDS TO BE
    #  IMPLEMENTED IN gPhoton!!!
    rawexpt -= shutter

    if rawexpt == 0:
        return rawexpt
    gcr = len(times[tix]) / rawexpt
    feeclkratio = 0.966
    refrate = model[band][1] / feeclkratio
    scr = model[band][0] * gcr + model[band][1]
    deadtime = 1 - scr / feeclkratio / refrate
    return rawexpt * (1.0 - deadtime)


def select_on_detector(event_table, threshold=400):
    detrad = event_table["detrad"].to_numpy()
    return event_table.take(
        # TODO: is isfinite() necessary?
        np.where(np.isfinite(detrad) & (detrad < threshold))[0]
    )


def table_values(table, columns):
    return np.array([table[column].to_numpy() for column in columns]).T


def make_images(photonfile, depth=(None, 30), band="NUV"):
    exposure_arrays, indexed, trange, wcs = prep_image_inputs(photonfile)
    # TODO: is this supposed to write each framesize out mid-loop?
    for framesize in depth:
        cntmovie, edgemovie, flagmovie = make_frames_at_depth(
            framesize,
            exposure_arrays,
            indexed,
            trange,
            wcs,
            band,
        )
        mc.print_inline("")
    # noinspection PyUnboundLocalVariable
    return cntmovie, flagmovie, edgemovie

    # TODO: Write the images.


def prep_image_inputs(photonfile):
    event_table, exposure_arrays = load_image_tables(photonfile)
    foc, wcs = generate_wcs_components(event_table)
    weights = 1.0 / event_table["response"].to_numpy()
    mask_ix = np.where(event_table["mask"].to_numpy())
    edge_ix = np.where(event_table["detrad"].to_numpy() > 350)
    t = event_table["t"].to_numpy()
    indexed = generate_indexed_values(edge_ix, foc, mask_ix, t, weights)
    trange = (
        np.min(event_table["t"].to_numpy()),
        np.max(event_table["t"].to_numpy()),
    )
    return exposure_arrays, indexed, trange, wcs


def generate_indexed_values(edge_ix, foc, mask_ix, t, weights):
    indexed = NestingDict()
    for value, value_name in zip((t, foc, weights), ("t", "foc", "weights")):
        for index, index_name in zip(
            (edge_ix, mask_ix, slice(None)), ("edge", "mask", "det")
        ):
            indexed[value_name][index_name] = value[index]
    return indexed


def make_frames_at_depth(
    framesize,
    exposure_arrays,
    indexed,
    trange,
    wcs,
    band,
):
    t0s = np.arange(
        trange[0],
        trange[1],
        framesize if framesize else trange[1] - trange[0],
    )
    cntmovie, flagmovie, edgemovie = [], [], []
    exptimes, tranges = [], []
    for i, t0 in enumerate(t0s):  # NOTE: 15s per loop
        mc.print_inline(f"Integrating frame {i + 1} of {len(t0s)}")
        # TODO: what are we doing with tranges and exptimes?
        t1 = t0 + (framesize if framesize else trange[1] - trange[0])
        trange = (t0, t1)
        tranges += [trange]
        exptimes += [
            optimize_compute_exptime(exposure_arrays, band, tranges[-1])
        ]
        cntmap, edgemap, flagmap = make_maps(indexed, trange, wcs)
        if len(t0s) == 1:
            cntmovie = cntmap
            flagmovie = flagmap
            edgemovie = edgemap
        else:
            cntmovie += [cntmap]
            flagmovie += [flagmap]
            edgemovie += [edgemap]
    return cntmovie, edgemovie, flagmovie


def where_between(whatever, t0, t1):
    return np.where((whatever >= t0) & (whatever < t1))[0]


def make_maps(indexed, trange, wcs):
    t0, t1 = trange
    t, foc, weights = [indexed[n]["det"] for n in ("t", "foc", "weights")]
    cnt_tix = where_between(t, t0, t1)
    cntmap = make_frame(foc[cnt_tix], weights[cnt_tix], wcs)
    flag_tix = where_between(indexed["t"]["mask"], t0, t1)
    flagmap = make_frame(
        indexed["foc"]["mask"][flag_tix],
        indexed["weights"]["mask"][flag_tix],
        wcs,
    )
    edge_tix = where_between(indexed["t"]["edge"], t0, t1)
    edgemap = make_frame(
        indexed["foc"]["edge"][edge_tix],
        indexed["weights"]["edge"][edge_tix],
        wcs,
    )
    return cntmap, edgemap, flagmap


def generate_wcs_components(event_table):
    wcs = optimize_wcs(table_values(event_table, ["ra", "dec"]))
    # This is a bottleneck, so only do it once.
    foc = wcs.sip_pix2foc(
        wcs.wcs_world2pix(table_values(event_table, ["ra", "dec"]), 1), 1
    )
    return foc, wcs


def load_image_tables(photonfile):
    event_table = parquet.read_table(
        photonfile,
        columns=[
            "ra",
            "dec",
            "t",
            "response",
            "flags",
            "mask",
            "detrad",
        ],
    )
    # Only deal with data actually on the 800x800 detector grid
    exposure_arrays = table_values(event_table, ["t", "flags"])
    event_table = select_on_detector(event_table)
    return event_table, exposure_arrays


#
bucketname = "gfcat-test"
eclipse = 23456
band = "NUV"
data_directory = "../test_data"
rerun = False
retain = False
ext = "parquet"

eclipse_directory = f"{data_directory}/e{eclipse}"
try:
    os.makedirs(eclipse_directory)
except FileExistsError:
    pass

# Use `dryrun=True` to just get the raw6 filepath
raw6file = download_raw6(
    eclipse, band, data_directory=data_directory, dryrun=True
)

obstype, rawexpt, nlegs = obstype_from_eclipse(eclipse)

if not obstype in ["MIS", "DIS"]:
    print(f"Skipping {obstype} mode visit.")
if nlegs > 0:
    print(f"Skipping multi-leg visit.")
if rawexpt < 600:
    print(f"Skipping visit with {rawexpt}s depth.")

# Download the raw6file from MAST for real
raw6file = download_raw6(eclipse, band, data_directory=data_directory)
photonfile = raw6file.replace("-raw6.fits.gz", ".parquet")
if not os.path.exists(photonfile):
    photonpipe(
        raw6file.split(".")[0][:-5],
        band,
        raw6file=raw6file,
        verbose=2,
        overwrite=False,
    )
    print("Calibrating photon list...")
file_stats = get_parquet_stats(photonfile, ["flags", "ra"])
if (file_stats["flags"]["min"] != 0) or (file_stats["ra"]["max"] is None):
    print("There is no unflagged data in this visit.")
#     # TODO: Set a flag to halt processing at this point


cnt, flg, edg = make_images(photonfile, depth=[None])
