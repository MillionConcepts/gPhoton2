"""
.. module:: gphoton_utils
   :synopsis: Read, plot, time conversion, and other functionality useful when
       dealing with gPhoton data.
"""

from __future__ import absolute_import, division, print_function

# Core and Third Party imports.
import gzip
from collections import defaultdict, Collection
from io import BytesIO
from itertools import product
from pathlib import Path
from statistics import mode


import fitsio
import numpy as np
import pandas as pd
import zstandard
from astropy.time import Time
import scipy.stats
import matplotlib.pyplot as plt


# gPhoton imports.
from pyarrow import parquet
from zstandard import ZstdCompressor
from astropy.io import fits as pyfits


import gfcat.gfcat_utils as gfu
import gPhoton.constants as c
import gPhoton.galextools as gt
import gPhoton.dbasetools as dt

# ------------------------------------------------------------------------------
def read_lc(csvfile, comment="|"):
    """
    Read a light curve csv file from gAperture.

    :param csvfile: The name of the csv file to read.

    :type csvfile: str

    :param comment: The character used to denote a comment row.

    :type comment: str

    :returns: pandas DataFrame -- The contents of the csv file.
    """

    return pd.io.parsers.read_csv(csvfile, comment=comment)


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def plot_lc(data_frame):
    """
    Plots a lightcurve from a CSV file data_frame - pandas DataFrame from
        read_lc()
    """

    plt.plot(data_frame.index.values, data_frame["flux"], "ko")

    plt.show()

    # @CHASE - Don't need a return here?@
    return


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def model_errors(catmag, band, sigma=3.0, mode="mag", trange=[1, 1600]):
    """
    Give upper and lower expected bounds as a function of the nominal
            magnitude of a source. Very useful for identifying outliers.

    :param catmag: Nominal AB magnitude of the source.

    :type catmag: float

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :param sigma: How many sigma out to set the bounds.

    :type sigma: float

    :param mode: Units in which to report bounds. Either 'cps' or 'mag'.

    :type mode: str

    :param trange: Set of integration times to compute the bounds on, in
        seconds.

    :type trange: list

    :returns: tuple -- A two-element tuple containing the lower and upper
        bounds, respectively.
    """

    if mode != "cps" and mode != "mag":
        print('mode must be set to "cps" or "mag"')
        exit(0)

    x = np.arange(trange[0], trange[1])

    cnt = gt.mag2counts(catmag, band)

    ymin = (cnt * x / x) - sigma * np.sqrt(cnt * x) / x

    ymax = (cnt * x / x) + sigma * np.sqrt(cnt * x) / x

    if mode == "mag":
        ymin = gt.counts2mag(ymin, band)
        ymax = gt.counts2mag(ymax, band)

    return ymin, ymax


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def data_errors(catmag, band, t, sigma=3.0, mode="mag"):
    """
    Given an array (of counts or mags), return an array of n-sigma error values.

    :param catmag: Nominal AB magnitude of the source.

    :type catmag: float

    :param t: Set of integration times to compute the bounds on, in seconds.

    :type t: list @CHASE - is this scalar or list? Also, consider trange
        instead of t to match first method?@

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :param sigma: How many sigma out to set the bounds.

    :type sigma: float

    :param mode: Units in which to report bounds. Either 'cps' or 'mag'.

    :type mode: str

    :returns: tuple -- A two-element tuple containing the lower and upper
        uncertainty, respectively.
    """

    if mode != "cps" and mode != "mag":
        print('mode must be set to "cps" or "mag"')
        exit(0)

    cnt = gt.mag2counts(catmag, band)

    ymin = (cnt * t / t) - sigma * np.sqrt(cnt * t) / t

    ymax = (cnt * t / t) + sigma * np.sqrt(cnt * t) / t

    if mode == "mag":
        ymin = gt.counts2mag(ymin, band)
        ymax = gt.counts2mag(ymax, band)

    return ymin, ymax


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def dmag_errors(t, band, sigma=3.0, mode="mag", mags=np.arange(13, 24, 0.1)):
    """
    Given an exposure time, give dmag error bars at a range of magnitudes.

    :param t: Set of integration times to compute the bounds on, in seconds.

    :type t: list @CHASE - is this scalar or list? Also, consider trange
        instead of t to match first method?@

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :param sigma: How many sigma out to set the bounds.

    :type sigma: float

    :param mode: Units in which to report bounds. Either 'cps' or 'mag'.

    :type mode: str

    :param mags: Set of magnitudes to compute uncertainties on.

    :type mags: numpy.ndarray

    :returns: tuple -- A three-element tuple containing the magnitudes and
        their lower and upper uncertainties, respectively.
    """

    cnts = gt.mag2counts(mags, band) * t

    ymin = (cnts - sigma / np.sqrt(cnts)) / t

    ymax = (cnts + sigma / np.sqrt(cnts)) / t

    if mode == "mag":
        ymin = mags - gt.counts2mag(ymin, band)
        ymax = mags - gt.counts2mag(ymax, band)

    return mags, ymin, ymax


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calculate_jd(galex_time):
    """
    Calculates the Julian date, in the TDB time standard, given a GALEX time.

    :param galex_time: A GALEX timestamp.

    :type galex_time: float

    :returns: float -- The time converted to a Julian date, in the TDB
        time standard.
    """

    if np.isfinite(galex_time):
        # Convert the GALEX timestamp to a Unix timestamp.
        this_unix_time = Time(
            galex_time + 315964800.0, format="unix", scale="utc"
        )

        # Convert the Unix timestamp to a Julian date, measured in the
        # TDB standard.
        this_jd_time = this_unix_time.tdb.jd
    else:
        this_jd_time = np.nan

    return this_jd_time


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calculate_jd_utc(galex_time):
    """
    Calculates the Julian date, in the UTC time standard, given a GALEX time.

    :param galex_time: A GALEX timestamp.

    :type galex_time: float

    :returns: float -- The time converted to a Julian date, in the UTC
        time standard.
    """

    if np.isfinite(galex_time):
        # Convert the GALEX timestamp to a Unix timestamp.
        this_unix_time = Time(
            galex_time + 315964800.0, format="unix", scale="utc"
        )

        # Convert the Unix timestamp to a Julian date, measured in the
        # UTC standard.
        this_jd_time = this_unix_time.utc.jd
    else:
        this_jd_time = np.nan

    return this_jd_time


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calculate_jd_tai(galex_time):
    """
    Calculates the Julian date, in the TAI time standard, given a GALEX time.

    :param galex_time: A GALEX timestamp.

    :type galex_time: float

    :returns: float -- The time converted to a Julian date, in the TAI
        time standard.
    """

    if np.isfinite(galex_time):
        # Convert the GALEX timestamp to a Unix timestamp.
        this_unix_time = Time(
            galex_time + 315964800.0, format="unix", scale="utc"
        )

        # Convert the Unix timestamp to a Julian date, measured in the
        # UTC standard.
        this_jd_time = this_unix_time.tai.jd
    else:
        this_jd_time = np.nan

    return this_jd_time


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def calculate_caldat(galex_time):
    """
    Calculates a Gregorian calendar date given a GALEX time, in the UTC
    time standard.

    :param galex_time: A GALEX timestamp.

    :type galex_time: float

    :returns: float -- The time converted to a Gregorian calendar date,
        in the UTC time standard.
    """

    if np.isfinite(galex_time):
        # Convert the GALEX timestamp to a Unix timestamp.
        this_unix_time = Time(
            galex_time + 315964800.0, format="unix", scale="utc"
        )

        # Convert the Unix timestamp to a calendar date, measured in the
        # UTC standard.
        this_caldat_time = this_unix_time.iso
    else:
        this_caldat_time = "NaN"

    return this_caldat_time


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def checkplot(
    csvfile,
    outfile=None,
    format="png",
    maxgap=500,
    imscale=4,
    nplots=10,
    cleanup=False,
):
    """
    Read a gAperture lighturve file and write diagnostic images of the
    brightness on a 'per-visit' basis with error bars and flagged bins
    indicated, alongside plots of common parameters that are correlated to
    false variabilty due to detector effects.

    :param csvfile: The filename of a gAperture lightcurve file.

    :type csvfile: string

    :param outfile: The base filename for output images.

    :type outfile: string

    :param format: The image format for output images.

    :type format: string

    :param maxgap: The maximum gap in seconds for bins to be considered
        contiguous (i.e. part of the same visit).

    :type maxgap: float

    :param imscale: A scale factor for image size.

    :type imscale: int

    :param nplots: The number of visits to be written to a single image before
        a new image is initiated.

    :type nplots: int

    :param cleanup: Close all matplotlib images when done. NOTE: Broken.

    :type cleanup: bool

    :returns: float -- The time converted to a Julian date, in the TAI
        time standard.
    """

    def crosscorr_title(a, b):
        return "{pearsonr}, {spearmanr}, {kendalltau}".format(
            pearsonr=round(scipy.stats.pearsonr(a, b)[0], 2),
            spearmanr=round(scipy.stats.spearmanr(a, b)[0], 2),
            kendalltau=round(scipy.stats.kendalltau(a, b)[0], 2),
        )

    lc = read_lc(csvfile)
    tranges = dt.distinct_tranges(np.array(lc["t0"]), maxgap=500)
    stepsz = np.median(lc["t1"] - lc["t0"])  # sort of a guess at stepsz
    n = 2  # temporary hacking variable for number of rows in figure
    for j in range(np.int(np.ceil(len(tranges) / float(nplots)))):
        plt.figure(
            figsize=(
                imscale * (len(tranges[j * nplots : (j + 1) * nplots])),
                imscale * n,
            )
        )
        for i, trange in enumerate(tranges[j * nplots : (j + 1) * nplots]):
            # Countrate
            plt.subplot(
                n,
                len(tranges[j * nplots : (j + 1) * nplots]),
                i + 1 + len(tranges[j * nplots : (j + 1) * nplots]) * 0,
            )
            plt.xticks([])
            if i == 0:
                plt.ylabel("cps_mcatbgsub")
            plt.ylim(
                (lc["cps_mcatbgsub"] - 2 * 5 * lc["cps_mcatbgsub_err"]).min(),
                (lc["cps_mcatbgsub"] + 2 * 5 * lc["cps_mcatbgsub_err"]).max(),
            )
            time_ix = np.where(
                (np.array(lc["t0"]) >= trange[0])
                & (np.array(lc["t1"]) <= trange[1])
            )
            if not len(time_ix[0]):
                continue
            tlim = (
                np.array(lc["t0"])[time_ix].min() - stepsz,
                np.array(lc["t1"])[time_ix].max() + stepsz,
            )
            plt.xlim(tlim[0], tlim[1])
            for nsigma in [5]:
                plt.errorbar(
                    np.array(lc["t_mean"])[time_ix],
                    np.array(lc["cps_mcatbgsub"])[time_ix],
                    yerr=nsigma * np.array(lc["cps_mcatbgsub_err"])[time_ix],
                    fmt="k.",
                )
            flag_ix = np.where(np.array(lc["flags"])[time_ix] > 0)
            plt.plot(
                np.array(lc["t_mean"])[time_ix][flag_ix],
                np.array(lc["cps_mcatbgsub"])[time_ix][flag_ix],
                "rx",
            )

            # Detector Radius
            plt.subplot(
                n,
                len(tranges[j * nplots : (j + 1) * nplots]),
                i + 1 + len(tranges[j * nplots : (j + 1) * nplots]) * 1,
            )
            plt.xticks([])
            if i == 0:
                plt.ylabel("detrad")
            plt.xlim(tlim[0], tlim[1])
            plt.ylim(
                np.array(lc["detrad"])[time_ix].min() - 2,
                np.array(lc["detrad"])[time_ix].max() + 2,
            )
            plt.plot(
                np.array(lc["t_mean"])[time_ix],
                np.array(lc["detrad"])[time_ix],
                "k.",
            )
            plt.plot(
                np.array(lc["t_mean"])[time_ix][flag_ix],
                np.array(lc["detrad"])[time_ix][flag_ix],
                "rx",
            )
            plt.title(
                crosscorr_title(
                    np.array(lc["cps_mcatbgsub_err"])[time_ix],
                    np.array(lc["detrad"])[time_ix],
                )
            )

        plt.tight_layout()
        if outfile:
            if len(tranges) > nplots:
                plt.savefig(
                    "{base}_{j}{type}".format(
                        base=outfile[:-4], j=j, type=outfile[-4:]
                    ),
                    dpi=300,
                )
            else:
                plt.savefig(outfile, dpi=300)
        if cleanup:  # This seems to not work for some reason... ?
            plt.close("all")
    return


# ------------------------------------------------------------------------------
def get_parquet_stats(fn, columns, row_group=0):
    group = parquet.read_metadata(fn).row_group(row_group)
    statistics = {}
    for column in group.to_dict()["columns"]:
        if column["path_in_schema"] in columns:
            statistics[column["path_in_schema"]] = column["statistics"]
    return statistics


def table_values(table, columns):
    return np.array([table[column].to_numpy() for column in columns]).T


def where_between(whatever, t0, t1):
    return np.where((whatever >= t0) & (whatever < t1))[0]


def unequally_stepped(array, rtol=1e-5, atol=1e-8):
    diff = array[1:] - array[:-1]
    unequal = np.where(~np.isclose(diff, mode(diff), rtol=rtol, atol=atol))
    return unequal, diff[unequal]


def write_gzip(
    whatever,
    outfile,
    compoptions=None,
    writemethod="writeto",
    writeoptions=None,
):
    if writeoptions is None:
        writeoptions = {}
    if compoptions is None:
        compoptions = {}
    gzipped = gzip.open(outfile, mode="wb", **compoptions)
    getattr(whatever, writemethod)(gzipped, **writeoptions)
    gzipped.close()


def write_zstd(
    whatever,
    out_fn,
    compoptions=None,
    writemethod="writeto",
    writeoptions=None,
):
    if writeoptions is None:
        writeoptions = {}
    if compoptions is None:
        compoptions = {}
    outfile = open(out_fn, mode="wb")
    zstdbuf = BytesIO()
    getattr(whatever, writemethod)(zstdbuf, **writeoptions)
    zstdbuf.seek(0)
    zcomp = ZstdCompressor(write_dict_id=True, **compoptions)
    zcomp.copy_stream(zstdbuf, outfile)
    outfile.close()


def get_fits_radec(header, endpoints_only=True):
    ranges = {}
    for coord, ix in zip(("ra", "dec"), (1, 2)):
        # explanatory variables
        steps = header[f"NAXIS{ix}"]
        stepsize = header[f"CDELT{ix}"]
        extent = steps * stepsize / 2
        center = header[f"CRVAL{ix}"]
        coord_endpoints = (center - extent, center + extent)
        if endpoints_only is True:
            ranges[coord] = coord_endpoints
        else:
            ranges[coord] = np.arange(*coord_endpoints, stepsize)
    return ranges


def imshow_clipped(data, clip_range=(None, 99), hdu_ix=0, cmap="Greys_r"):
    if isinstance(data, pyfits.HDUList):
        header = data[hdu_ix].header
        data = data[hdu_ix].data
    elif isinstance(data, fitsio.FITS):
        header = data[hdu_ix].read_header()
        data = data[hdu_ix].read()
    else:
        header = None
    if (len(data.shape) == 3) and (data.shape[0] == 1):
        data = data[0]
    clip_kwargs = {}
    for clip_amount, clip_kwarg in zip(clip_range, ("a_min", "a_max")):
        if clip_amount is not None:
            clip_kwargs[clip_kwarg] = np.percentile(data, clip_amount)
        else:
            clip_kwargs[clip_kwarg] = None
    fig, ax = plt.subplots()
    im = ax.imshow(np.clip(data, **clip_kwargs), cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)
    if header is not None:
        radec = get_fits_radec(header, endpoints_only=False)
        for data_coord, sky_coord in zip(("x", "y"), ("ra", "dec")):
            vals = radec[sky_coord]
            ticks = np.round(np.linspace(0, len(vals), 10))
            tick_labels = np.round(np.linspace(vals[0], vals[-1], 10), 2)
            getattr(ax, f"set_{data_coord}ticks")(ticks)
            getattr(ax, f"set_{data_coord}ticklabels")(tick_labels)
    return fig


class NestingDict(defaultdict):
    """
    shorthand for automatically-nesting dictionary -- i.e.,
    insert a series of keys at any depth into a NestingDict
    and it automatically creates all needed levels above.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = NestingDict

    __repr__ = dict.__repr__


def pyfits_zopen(filename):
    zdecomp = zstandard.ZstdDecompressor()
    zinbuf = BytesIO()
    with open(filename, mode="rb") as infile:
        zdecomp.copy_stream(infile, zinbuf)
    zinbuf.seek(0)
    return pyfits.open(zinbuf)


def diff_photonlist_and_movie_coords(movie_radec, photon_radec):
    diffs = {}
    minmax_funcs = {"min": min, "max": max}
    for coord, op in product(("ra", "dec"), ("min", "max")):
        diffs[f"{coord}_{op}"] = abs(
            minmax_funcs[op](movie_radec[coord]) - photon_radec[coord][op]
        )
    return diffs


def listify(thing):
    """Always a list, for things that want lists"""
    if isinstance(thing, Collection):
        if not isinstance(thing, str):
            return list(thing)
    return [thing]


def make_wcs_from_radec(radec):
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


def read_image(fn):
    if Path(fn).suffix == '.zstd':
        hdu = pyfits_zopen(fn)
    else:
        hdu = pyfits.open(fn)
    print(f"Opened {fn}.")
    image = hdu[0].data
    exptimes, tranges = [], []
    for i in range(hdu[0].header["N_FRAME"]):
        exptimes += [hdu[0].header[f"EXPT_{i}"]]
        tranges += [[hdu[0].header[f"T0_{i}"], hdu[0].header[f"T1_{i}"]]]
    skypos = (hdu[0].header["CRVAL1"], hdu[0].header["CRVAL2"])
    wcs = gfu.make_wcs(skypos)
    print("\tParsed file header.")
    try:
        flagmap = hdu[1].data
        edgemap = hdu[2].data
        print("\tRetrieved flag and edge maps.")
    except IndexError:
        flagmap = None
        edgemap = None
    return {
        'image': image,
        'flagmap': flagmap,
        'edgemap': edgemap,
        'wcs': wcs,
        'tranges': tranges,
        'exptimes': exptimes,
        'skypos': skypos
    }
