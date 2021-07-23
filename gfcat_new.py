import re
import warnings
from collections.abc import Sequence
import os
from pathlib import Path
from typing import Optional, Union

import fast_histogram as fh
import astropy.io.fits as pyfits
import numpy as np
import astropy.wcs.wcs
from more_itertools import windowed
from pyarrow import parquet
from zstandard import ZstdCompressor

from gPhoton.gphoton_utils import (
    table_values,
    where_between,
    write_zstd,
    NestingDict,
    get_parquet_stats,
)
import gfcat.gfcat_utils as gfu
import gPhoton.constants as c
from gPhoton import __version__, PhotonPipe
from gPhoton import MCUtils as mc

import time


# bucketname = "gfcat-test"
# data_directory = "../test_data"
# rerun = False
# retain = False
# ext = "parquet"
#
# eclipse_directory = f"{data_directory}/e{eclipse}"
# try:
#     os.makedirs(eclipse_directory)
# except FileExistsError:
#     pass
#
# # Use `dryrun=True` to just get the raw6 filepath
# raw6file = download_raw6(
#     eclipse, band, data_directory=data_directory, dryrun=True
# )
#
# obstype, rawexpt, nlegs = obstype_from_eclipse(eclipse)
#
# if not obstype in ["MIS", "DIS"]:
#     print(f"Skipping {obstype} mode visit.")
# if nlegs > 0:
#     print(f"Skipping multi-leg visit.")
# if rawexpt < 600:
#     print(f"Skipping visit with {rawexpt}s depth.")
#
# # Download the raw6file from MAST for real
# raw6file = download_raw6(eclipse, band, data_directory=data_directory)
# photonfile = raw6file.replace("-raw6.fits.gz", ".parquet")
# if not os.path.exists(photonfile):
#     photonpipe(
#         raw6file.split(".")[0][:-5],
#         band,
#         raw6file=raw6file,
#         verbose=2,
#         overwrite=False,
#     )
#     print("Calibrating photon list...")
# file_stats = get_parquet_stats(photonfile, ["flags", "ra"])
# if (file_stats["flags"]["min"] != 0) or (file_stats["ra"]["max"] is None):
#     print("There is no unflagged data in this visit.")
# #     # TODO: Set a flag to halt processing at this point


# @profile
def optimize_wcs(radec):
    real_ra = radec[:, 0][np.isfinite(radec[:, 0])]
    real_dec = radec[:, 1][np.isfinite(radec[:, 1])]
    ra_range = real_ra.min(), real_ra.max()
    dec_range = real_dec.min(), real_dec.max()
    center_skypos = (np.mean(ra_range), np.mean(dec_range))
    imsz = (
        int(np.ceil((dec_range[1] - dec_range[0]) / c.DEGPERPIXEL)),
        int(np.ceil((ra_range[1] - ra_range[0]) / c.DEGPERPIXEL)),
    )
    # imsz = (3200, 3200)
    return gfu.make_wcs(center_skypos, imsz=imsz, pixsz=c.DEGPERPIXEL)


def make_frame(foc, weights, wcs):
    imsz = (
        int((wcs.wcs.crpix[1] - 0.5) * 2),
        int((wcs.wcs.crpix[0] - 0.5) * 2),
    )
    frame = fh.histogram2d(
        foc[:, 1] - 0.5,
        foc[:, 0] - 0.5,
        bins=imsz,
        range=([[0, imsz[0]], [0, imsz[1]]]),
        weights=weights,
    )
    return frame


def optimize_compute_shutter(timeslice, flagslice, trange, shutgap=0.05):
    ix = np.where(flagslice == 0)
    t = np.hstack([trange[0], np.unique(timeslice[ix]), +trange[1]])
    ix = np.where(t[1:] - t[:-1] >= shutgap)
    shutter = np.array(t[1:] - t[:-1])[ix].sum()
    return shutter


def optimize_compute_exptime(
    events: np.ndarray, band: str, trange: Sequence[int]
) -> float:
    rawexpt = trange[1] - trange[0]
    times = events[:, 0]
    tix = np.where((times >= trange[0]) & (times < trange[1]))
    timeslice = times[tix]
    shutter = optimize_compute_shutter(
        timeslice, flagslice=events[:, 1][tix], trange=trange
    )

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
    gcr = len(timeslice) / rawexpt
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


# @profile
def make_images(
    photonfile: Union[str, os.PathLike],
    depth: Sequence[Union[int, None]] = (None, 30),
    band: str = "NUV",
    edge_threshold: int = 350,
):
    exposure_array, indexed, trange, wcs = prep_image_inputs(
        photonfile, edge_threshold
    )
    # TODO: write each framesize out mid_loop
    movies, tranges, exptimes = {}, [], []
    for framesize in depth:
        movies, tranges, exptimes = make_frames_at_depth(
            framesize,
            exposure_array,
            indexed,
            trange,
            wcs,
            band,
        )
        mc.print_inline("")
    return movies, wcs, tranges, exptimes

    # TODO: Write the images.


def prep_image_inputs(photonfile, edge_threshold):
    event_table, exposure_array = load_image_tables(photonfile)
    foc, wcs = generate_wcs_components(event_table)
    with warnings.catch_warnings():
        # don't bother us about divide-by-zero errors
        warnings.simplefilter("ignore")
        weights = 1.0 / event_table["response"].to_numpy()
    mask_ix = np.where(event_table["mask"].to_numpy())
    edge_ix = np.where(event_table["detrad"].to_numpy() > edge_threshold)
    t = event_table["t"].to_numpy()
    indexed = generate_indexed_values(edge_ix, foc, mask_ix, t, weights)
    trange = (
        np.min(event_table["t"].to_numpy()),
        np.max(event_table["t"].to_numpy()),
    )
    return exposure_array, indexed, trange, wcs


def generate_indexed_values(edge_ix, foc, mask_ix, t, weights):
    indexed = NestingDict()
    for value, value_name in zip((t, foc, weights), ("t", "foc", "weights")):
        for index, index_name in zip(
            (edge_ix, mask_ix, slice(None)), ("edge", "mask", "det")
        ):
            indexed[value_name][index_name] = value[index]
    return indexed


def make_frames_at_depth(
    framesize: Optional[int],
    exposure_array: np.ndarray,
    indexed: dict,
    total_trange: tuple[int, int],
    wcs: astropy.wcs.wcs.WCS,
    band: str,
) -> tuple[dict[str, list[np.ndarray]], list, list]:
    """
    :param framesize: framesize in seconds; None for full range
    :param exposure_array: t and flags, _including_ off-detector, for exptime
    :param indexed: weights, t, and foc indexed against det, edge, and mask
    :param total_trange: (time minimum, time maximum) for on-detector events
    :param wcs: wcs object
    :param band: "FUV" or "NUV"
    """
    interval = total_trange[1] - total_trange[0]
    t0s = np.arange(
        total_trange[0],
        total_trange[1] + framesize
        if framesize is not None
        else total_trange[1] + interval,
        framesize
        if framesize is not None
        else total_trange[1] - total_trange[0],
    )
    tranges = list(windowed(t0s, 2))

    movies = {"cnt": [], "flag": [], "edge": []}
    exptimes = []
    for i, t0 in enumerate(tranges):
        mc.print_inline(f"Integrating frame {i + 1} of {len(t0s)}")
        trange = tranges[i]
        exptimes += [optimize_compute_exptime(exposure_array, band, trange)]
        # noinspection PyTypeChecker
        cntmap, edgemap, flagmap = make_maps(indexed, trange, wcs)
        movies["cnt"] += [cntmap]
        movies["flag"] += [flagmap]
        movies["edge"] += [edgemap]
    return movies, tranges, exptimes


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
    exposure_array = table_values(event_table, ["t", "flags"])
    event_table = select_on_detector(event_table)
    return event_table, exposure_array


def populate_fits_header(band, wcs, tranges, exptimes):
    header = pyfits.Header()
    header["CDELT1"], header["CDELT2"] = wcs.wcs.cdelt
    header["CTYPE1"], header["CTYPE2"] = wcs.wcs.ctype
    header["CRPIX1"], header["CRPIX2"] = wcs.wcs.crpix
    header["CRVAL1"], header["CRVAL2"] = wcs.wcs.crval
    header["EQUINOX"], header["EPOCH"] = 2000.0, 2000.0
    header["BAND"] = 1 if band == "NUV" else 2
    header["VERSION"] = "v{v}".format(v=__version__)
    header["EXPSTART"] = np.array(tranges).min()
    header["EXPEND"] = np.array(tranges).max()
    header["EXPTIME"] = sum(t1 - t0 for (t0, t1) in tranges)
    header["N_FRAME"] = len(tranges)
    for i, trange in enumerate(tranges):
        header["T0_{i}".format(i=i)] = trange[0]
        header["T1_{i}".format(i=i)] = trange[1]
        header["EXPT_{i}".format(i=i)] = exptimes[i]
    return header


from memory_profiler import profile


@profile
def write_fits_movie(band, depth, eclipse, exptimes, movies, tranges, wcs):
    # TODO: nicer depth filenaming in the inner loop, garbage hack
    if depth == [None]:
        frame_title = "-full"
    else:
        frame_title = f"-{depth[0]}s"
    movie_scratch_fn = f"test_data/e{eclipse}/e{eclipse}{frame_title}-cnt.fits"
    # TODO: rewrite this to have to not assemble the primary hdu in order to
    #  make the header
    if os.path.exists(movie_scratch_fn):
        os.unlink(movie_scratch_fn)
    header = populate_fits_header(band, wcs, tranges, exptimes)
    for key in ["cnt", "flag", "edge"]:
        pyfits.append(movie_scratch_fn, np.stack(movies[key]), header=header)
        del movies[key]
    ZstdCompressor().copy_stream(
        open(movie_scratch_fn, "rb"), open(movie_scratch_fn + ".zstd", "wb")
    )
    os.unlink(movie_scratch_fn)


def run_photonpipe(eclipse):
    # eclipse = 23456
    band = "NUV"
    data_directory = "test_data"
    raw6file = gfu.download_raw6(eclipse, band, data_directory=data_directory)
    photonfile = Path(
        data_directory,
        f"e{eclipse}",
        f"e{eclipse}-{'n' if band == 'NUV' else 'f'}d",
    )
    print(f"Photon data file: {photonfile}")
    PhotonPipe.photonpipe(
        photonfile,
        band,
        raw6file=raw6file,
        verbose=2,
        chunksz=1000000,
        threads=4,
    )


def make_movies(eclipse, band, depths):
    photonfile = f"test_data/e{eclipse}/e{eclipse}-nd.parquet"
    movies, wcs, tranges, exptimes = make_images(photonfile, depth=depths)
    write_fits_movie(
        band,
        depths,
        eclipse,
        exptimes,
        movies,
        tranges,
        wcs,
    )

    #
    # def optimize_make_photometry(listfile: str, cntfile: str):
    #     # TODO: specify depth in filename by default?
    #     photomfile = re.sub(r"-\w\w\.parquet", "-photom.csv", listfile)
    #     cntmap, flagmap, edgemap, wcs, tranges, exptimes = read_image(cntfilename)
    #     if not cntmap.max():
    #         print('Image contains nothing.')
    #         pathlib.Path(os.path.dirname(photonfile) + "/No{band}".format(band=band)
    #                                 ).touch()
    #         return
    #
    #     trange, exptime = tranges[0], exptimes[0]
    #     if exptime < 600:
    #         print("Skipping low exposure time visit.")
    #         pathlib.Path("{path}/LowExpt".format(path=os.path.dirname(photonfile))).touch()
    #         return
    #     movmap, _, _, _, tranges, exptimes = read_image(cntfilename.replace("-cnt", "-mov"))
    #     daofind = DAOStarFinder(fwhm=5, threshold=0.01)
    #     sources = daofind(cntmap / exptime)
    #     try:
    #         print(f'Located {len(sources)} sources.')
    #     except TypeError:
    #         print('Image contains no sources.')
    #         pathlib.Path(os.path.dirname(photonfile) + "/No{band}".format(band=band)
    #                                         ).touch()
    #         return
    #     positions = (sources["xcentroid"], sources["ycentroid"])
    #     apertures = CircularAperture(positions, r=8.533333333333326)
    #     phot_table = aperture_photometry(cntmap, apertures)
    #     flag_table = aperture_photometry(flagmap, apertures)
    #     edge_table = aperture_photometry(edgemap, apertures)
    #
    #     phot_visit = sources.to_pandas()
    #     phot_visit["xcenter"] = phot_table.to_pandas().xcenter.tolist()
    #     phot_visit["ycenter"] = phot_table.to_pandas().ycenter.tolist()
    #     phot_visit["aperture_sum"] = phot_table.to_pandas().aperture_sum.tolist()
    #     phot_visit["aperture_sum_mask"] = flag_table.to_pandas().aperture_sum.tolist()
    #     phot_visit["aperture_sum_edge"] = edge_table.to_pandas().aperture_sum.tolist()
    #     phot_visit["ra"] = [
    #         wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()[0]
    #         for pos in apertures.positions
    #     ]
    #     phot_visit["dec"] = [
    #         wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()[1]
    #         for pos in apertures.positions
    #     ]
    #
    #     for i, frame in enumerate(movmap):
    #         mc.print_inline("Extracting photometry from frame #{i}".format(i=i))
    #         phot_visit["aperture_sum_{i}".format(i=i)] = (
    #             aperture_photometry(frame, apertures).to_pandas()["aperture_sum"].tolist()
    #         )
    #     print("Writing data to {f}".format(f=photomfile))
    #     phot_visit.to_csv(photomfile, index=False)
    #     pd.DataFrame(
    #         {
    #             "expt": exptimes,
    #             "t0": np.array(tranges)[:, 0].tolist(),
    #             "t1": np.array(tranges)[:, 1].tolist(),
    #         }
    #     ).to_csv(cntfilename.replace("-cnt.fits.gz", "-exptime.csv"), index=False)

    return


if __name__ == "__main__":
    for eclipse in [
        4357,
        # 7666,
        # 9006,
        #     42645,
        #     28622,
        #     4357,
        #     46794,
        #     40200,
        #     34479,
        #     11943,
        #     22650,
    ]:
        # for eclipse in (43009, 43010, 43011, 43012, 43013):
        print(eclipse)
        start = time.time()
        data_directory = "test_data"
        band = "NUV"
        photonfile = Path(
            data_directory,
            f"e{eclipse}",
            f"e{eclipse}-{'n' if band == 'NUV' else 'f'}d.parquet",
        )
        if not photonfile.exists():
            run_photonpipe(eclipse)
        file_stats = get_parquet_stats(photonfile, ["flags", "ra"])
        if (file_stats["flags"]["min"] != 0) or (
            file_stats["ra"]["max"] is None
        ):
            print("no unflagged data in this visit, not making movies")
            continue
        make_movies(eclipse, "NUV", [30])
        print(time.time() - start)
