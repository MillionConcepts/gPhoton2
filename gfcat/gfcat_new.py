import os

import fast_histogram as fh
import numpy as np
import pandas as pd
from pyarrow import parquet

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


bucketname = "gfcat-test"
eclipse = 23456
band = "NUV"
data_directory = "data"
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


def get_parquet_stats(fn, columns, row_group=0):
    group = parquet.read_metadata(fn).row_group(row_group)
    statistics = {}
    for column in group.to_dict()["columns"]:
        if column["path_in_schema"] in columns:
            statistics[column["path_in_schema"]] = column["statistics"]
    return statistics


# Download the raw6file from MAST for real
raw6file = download_raw6(eclipse, band, data_directory=data_directory)
photonfile = raw6file.replace("-raw6.fits.gz", ".parquet")
if not os.path.exists(photonfile):
    photonpipe(raw6file.split(".")[0][:-5], band, raw6file=raw6file, verbose=2, overwrite=False)
    print("Calibrating photon list...")
file_stats = get_parquet_stats(photonfile, ["flags", "ra"])
if (file_stats["flags"]["min"] != 0) or (file_stats["ra"]["max"] is None):
    print("There is no unflagged data in this visit.")
    # TODO: Set a flag to halt processing at this point


def optimize_wcs(event_data):
    ra_range = event_data["ra"].min(), event_data["ra"].max()
    dec_range = event_data["dec"].min(), event_data["dec"].max()
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


def make_images(photonfile, depth=[None, 30]):
    event_data = parquet.read_table(
        photonfile, columns=["ra", "dec", "t", "response", "flags", "col", "row", "mask", "detrad"]
    ).to_pandas()
    # Only deal with data actually on the 800x800 detector grid
    # MICHAEL: this is probably easier as as something like: event_data.dropna(subset='response')
    #detector_radius = np.sqrt(
    #    (event_data.col.values - 400) ** 2 + (event_data.row.values - 400) ** 2
    #)
    ix = np.where(np.isfinite(event_data["detrad"]values) & (event_data["detrad"]values < 400))

    wcs = optimize_wcs(event_data[['ra','dec']].iloc[ix])

    # This is a bottleneck, so only do it once.
    foc = wcs.sip_pix2foc(
        wcs.wcs_world2pix(event_data[["ra", "dec"]].values, 1), 1
    )  # This is a bottleneck. Takes ~17s.
    weights = 1.0 / event_data["response"].values

    trange = (
        event_data["t"].iloc[ix].min(),
        event_data["t"].iloc[ix].max(),
    )

    mask_ix = np.where(event_data['mask'].values)

    edge_ix = np.where(event_data["detrad"]values > 350)
    # NOTE: 1m45s runtime up to this point

    for framesize in depth:
        t0s = np.arange(
            trange[0],
            trange[1],
            framesize if framesize else trange[1] - trange[0],
        )
        cntmovie, flagmovie, edgemovie = [], [], []
        exptimes, tranges = [], []
        for i, t0 in enumerate(t0s):  # NOTE: 15s per loop
            mc.print_inline(f"Integrating frame {i+1} of {len(t0s)}")
            t1 = t0 + (framesize if framesize else trange[1] - trange[0])
            tranges += [[t0, t1]]
            exptimes += [compute_exptime(event_data, band, tranges[-1])]

            tix = np.where(
                (event_data["t"].values[ix] >= t0)
                & (event_data["t"].values[ix] < t1)
            )
            cntmap = make_frame(foc[ix][tix], weights[ix][tix], wcs)
            tix = np.where(
                (event_data.t.values[ix][mask_ix] >= t0)
                & (event_data.t.values[ix][mask_ix] < t1)
            )
            flagmap = make_frame(
                foc[ix][mask_ix][tix], weights[ix][mask_ix][tix], wcs
            )
            tix = np.where(
                (event_data.t.values[ix][edge_ix] >= t0)
                & (event_data.t.values[ix][edge_ix] < t1)
            )
            edgemap = make_frame(
                foc[ix][edge_ix][tix], weights[ix][edge_ix][tix], wcs
            )

            if len(t0s) == 1:
                cntmovie = cntmap
                flagmovie = flagmap
                edgemovie = edgemap
            else:
                cntmovie += [cntmap]
                flagmovie += [flagmap]
                edgemovie += [edgemap]
        mc.print_inline("")
    return cntmovie, flagmovie, edgemovie

    # TODO: Write the images.


cnt,flg,edg=make_images(photonfile,depth=[None])

make_photometry(eclipse, band, rerun=rerun, data_directory=data_directory)
