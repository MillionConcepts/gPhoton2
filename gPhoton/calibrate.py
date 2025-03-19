"""
.. module:: calibrate
   :synopsis: Numerous methods for calibrating the raw photon event data.
       Many of these instantiate or make use of specific detector hardware
       parameters / constants related to the "static" or detector-space event
       calibration, including walk, wiggle, linearity, post-CSP, and stim
       scaling corrections.
"""
from collections.abc import Sequence
from pathlib import Path

import numba
from numba import njit
import numpy as np

from gPhoton import cals
from gPhoton.aspect import aspect_tables
import gPhoton.constants as c
from gPhoton.types import GalexBand


# ------------------------------------------------------------------------------
def clk_cen_scl_slp(band: GalexBand, eclipse: int) -> tuple:
    """
    Return the detector clock, center, scale, and slope constants. These are
        empirically determined detector-space calibration parameters that help
        define the relationship between raw event data and the physical
        positions of events on the detector.

    :param band: The band to return the constants for, either 'FUV' or 'NUV'.

    :param eclipse: The eclipse number to return the constants for.

    :returns: tuple -- a tuple containing the x-clock, y-clock, x-center,
        y-center, x-scale, y-scale, x-slope, and y-slope constants.
    """

    band = band.upper()
    if band == "FUV":
        xclk, yclk = 1997, 1993
        xcen, ycen = 7200, 6670
        xscl, yscl = 7.78, 10.73
        xslp, yslp = 0.0, 0.0
    elif band == "NUV":
        xclk, yclk = 2007, 1992
        # Special value for post-CSP event.
        if eclipse >= 38150:
            yclk = 2016
        xcen, ycen = 7400, 6070
        xscl, yscl = 8.79, 14.01
        xslp, yslp = 0.53, 0.0
    else:
        raise ValueError("Band must be either FUV or NUV ... Exiting.")

    return xclk, yclk, xcen, ycen, xscl, yscl, xslp, yslp


# -----------------------------------------------------------------------------
def post_csp_caldata():
    """
    Loads the calibration data for after the CSP event.

    :returns: tuple -- A six-element tuple containing the wiggle, walk, and
        clock corrections. See the calibration paper for details.
    """

    print("Loading post-CSP wiggle file...")

    wig2fits, wig2head = cals.wiggle2()
    wig2 = np.zeros([128, 8, 32, 8])

    wig2[
        wig2fits["yy"], wig2fits["YB"], wig2fits["YA"], wig2fits["XB"]
    ] = wig2fits["ycor"]

    wig2data = {"start": wig2head["Y_AS_0"], "inc": wig2head["Y_AS_INC"]}

    print("Loading post-CSP walk file...")

    wlk2fits, wlk2head = cals.walk2()
    wlk2 = np.zeros([100, 8, 32])

    wlk2[wlk2fits["yy"], wlk2fits["yb"], wlk2fits["q"]] = wlk2fits["ycor"]

    wlk2data = {"start": wlk2head["Y_AS_0"], "inc": wlk2head["Y_AS_INC"]}

    print("Loading post-CSP clock file...")

    clk2fits, clk2head = cals.clock2()
    clk2 = np.zeros([100, 8])

    clk2[clk2fits["yy"], clk2fits["YB"]] = clk2fits["ycor"]

    clk2data = {"start": clk2head["Y_AS_0"], "inc": clk2head["Y_AS_INC"]}

    return wig2, wig2data, wlk2, wlk2data, clk2, clk2data


# ------------------------------------------------------------------------------
def avg_stimpos(band, eclipse):
    """
    Define the mean detector stim positions.

    :param band: The band to return the average stim positions for,
        either 'FUV' or 'NUV'.

    :type band: str

    :param eclipse: The eclipse number to return the average stim positions for.

    :type eclipse: int

    :returns: dict -- A dict containing the average x and y positions of the
        four stims (for the requested band).
    """

    if band == "FUV":
        avgstim = {
            "x1": -2541.88,
            "x2": 2632.06,
            "x3": -2541.53,
            "x4": 2631.68,
            "y1": 2455.28,
            "y2": 2455.02,
            "y3": -2550.89,
            "y4": -2550.92,
        }

    elif band == "NUV":
        if eclipse >= 38150:
            # The average stim positions after the clock change (post-CSP).
            avgstim = {
                "x1": -2722.53,
                "x2": 2470.29,
                "x3": -2721.98,
                "x4": 2471.09,
                "y1": 2549.96,
                "y2": 2550.10,
                "y3": -2538.57,
                "y4": -2538.62,
            }
        else:
            # The average stim positions for pre-CSP data (eclipse 37423).
            avgstim = {
                "x1": -2722.27,
                "x2": 2468.84,
                "x3": -2721.87,
                "x4": 2469.85,
                "y1": 2453.89,
                "y2": 2453.78,
                "y3": -2565.81,
                "y4": -2567.83,
            }

    else:
        raise ValueError("No valid band specified.")

    return avgstim


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def find_stims_index(x, y, band, eclipse, margin=90.001):
    """
    Given a list of detector x,y positions of events, returns four
        arrays that contain the indices of likely stim events for that stim,
        i.e., the first array contains positions for stim1, the second array has
        positions of stim2, etc.

        Example of how the return indexes are used: x[index1], y[index1] would
        give all of the event positions for stim1.

    :param x: Detector 'x' positions to identify likely stim events from.

    :type x: list

    :param y: Detector 'y' positions to identify likely stim events from.

    :type y: list

    :param band: The band to return the constants for, either 'FUV' or 'NUV'.

    :type band: str

    :param eclipse: The eclipse number to return the average stim positions for.

    :type eclipse: int

    :param margin: +/- extent in arcseconds defining search box

    :type margin: float

    :returns: tuple -- A four-element tuple containing arrays of indexes that
        correpond to the event positions for stim1, stim2, stim3, and stim4.
    """

    # [Future] This method could be programmed better. Consider using numpy
    # "where" and the logical '&' operator, instead of .nonzero()?

    # Plate scale (in arcsec/mm).
    pltscl = 68.754932
    # Plate scale in arcsec per micron.
    c.ASPUM = pltscl / 1000.0
    # [Future] Could define the plate scale (in both units) as a constant for
    # the module, since it is used in many places?

    x_as = np.array(x) * c.ASPUM
    y_as = np.array(y) * c.ASPUM

    avg = avg_stimpos(band, eclipse)

    index1 = (
        (x_as > (avg["x1"] - margin))
        & (x_as < (avg["x1"] + margin))
        & (y_as > (avg["y1"] - margin))
        & (y_as < (avg["y1"] + margin))
    ).nonzero()[0]
    index2 = (
        (x_as > (avg["x2"] - margin))
        & (x_as < (avg["x2"] + margin))
        & (y_as > (avg["y2"] - margin))
        & (y_as < (avg["y2"] + margin))
    ).nonzero()[0]
    index3 = (
        (x_as > (avg["x3"] - margin))
        & (x_as < (avg["x3"] + margin))
        & (y_as > (avg["y3"] - margin))
        & (y_as < (avg["y3"] + margin))
    ).nonzero()[0]
    index4 = (
        (x_as > (avg["x4"] - margin))
        & (x_as < (avg["x4"] + margin))
        & (y_as > (avg["y4"] - margin))
        & (y_as < (avg["y4"] + margin))
    ).nonzero()[0]

    return index1, index2, index3, index4


# ------------------------------------------------------------------------------
def rtaph_yap(ya, yb, yamc):
    """
    For post-CSP data, 'wrap' the YA value for YA in [0,1]. From rtaph.c.

    :param ya: Y axis wiggle.

    :type ya: numpy.ndarray

    :param yb: Y axis coarse clock.

    :type yb: numpy.ndarray

    :param yamc: Raw Y detector position in FEE pixels.

    :type yamc: numpy.ndarray

    :returns: numpy.ndarray
    """

    yap = np.append([], ya)
    ix = ((yb > 1) & (yb < 5)).nonzero()[0]

    ix1 = ((ya[ix] == 0) & (yamc[ix] > -50)).nonzero()[0]
    yap[ix[ix1]] += 32

    ix1 = ((ya[ix] == 1) & (yamc[ix] > -10)).nonzero()[0]
    yap[ix[ix1]] += 32

    return np.array(yap, dtype="int64") % 32


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def rtaph_yac(yactbl, ya, yb, yamc, eclipse):
    """
    Compute a the YAmC (yac) correction to the Y detector FEE position.

    :param yactbl: yac correction lookup table

    :type yactbl: numpy.ndarry

    :param ya: Y axis wiggle.

    :type ya: numpy.ndarray

    :param yb: Y axis coarse clock.

    :type yb: numpy.ndarray

    :param yamc: Raw Y detector position in FEE pixels.

    :type yamc: numpy.ndarray

    :param eclipse: The eclipse number to return the constants for.

    :type eclipse: int

    :returns: numpy.ndarray
    """

    yac = np.zeros(len(ya))
    if eclipse <= 37423:
        return yac

    yap = rtaph_yap(ya, yb, yamc)

    return yactbl[yap, yb]


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def rtaph_yac2(
    q, xb, yb, ya, y, wig2, wig2data, wlk2, wlk2data, clk2, clk2data
):
    """
    Compute a secondary correction to the YAmC (yac) detector FEE Y position.

    :param q: Detected pulse height

    :type q: numpy.ndarray

    :param xb: X axis coarse clock.

    :type xb: numpy.ndarray

    :param yb: Y axis coarse clock.

    :type yb: int

    :param ya: Y axis wiggle.

    :type ya: int

    :param y: Detector y position.

    :type y: numpy.ndarray

    :param c.ASPUM: Detector arcseconds per micrometer.

    :type c.ASPUM: float

    :param wig2: Secondary wiggle correction lookup table.

    :type wig2: numpy.ndarray

    :param wig2data: Secondary wiggle reference values.

    :type wig2data: dict

    :param wlk2: Secondary walk correction lookup table.

    :type wlk2: numpy.ndarray

    :param wlk2data: Secondary walk correction reference values.

    :type wlk2data: dict

    :param clk2: Secondary clock correction lookup table.

    :type clk2: numpy.ndarray

    :param clk2data: Secondary clock correction reference values.

    :type clk2data: dict

    :returns: numpy.ndarray -- Secondary YAmC corrections.
    """
    y_as = y * c.ASPUM
    yac_as = np.zeros(len(y_as))
    ix = ((y_as > -2000) & (y_as < 2000)).nonzero()[0]

    ii = (np.array(y_as, dtype="int64") - wig2data["start"]) / wig2data["inc"]
    yac_as[ix] = wig2[np.array(ii[ix], dtype="int64"), yb[ix], ya[ix], xb[ix]]

    ii = (np.array(y_as, dtype="int64") - wlk2data["start"]) / wlk2data["inc"]
    yac_as[ix] = wlk2[np.array(ii[ix], dtype="int64"), yb[ix], q[ix]]

    ii = (np.array(y_as, dtype="int64") - clk2data["start"]) / clk2data["inc"]
    yac_as[ix] = clk2[np.array(ii[ix], dtype="int64"), yb[ix]]

    return yac_as / c.ASPUM


def flat_scale_parameters(band):
    """
    Return the flat scaling parameters for a given band.

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :returns: tuple -- A three-element tuple containing the flat scaling
        parameters.
    """

    if band == "NUV":
        # flat_correct and flat_t0 never get used.
        # They are only retained in this code for historical purposes.
        #flat_correct = -0.0154
        #flat_t0 = 840418779.02
        flat_correct_0 = 1.9946352
        flat_correct_1 = -1.9679445e-09
        flat_correct_2 = 9.3025231e-19
    elif band == "FUV":
        #flat_correct = -0.0031
        #flat_t0 = 840418779.02
        flat_correct_0 = 1.2420282
        flat_correct_1 = -2.8843099e-10
        flat_correct_2 = 0.000
    else:
        raise ValueError("Band must be NUV or FUV.")

    return flat_correct_0, flat_correct_1, flat_correct_2


def compute_flat_scale(t, band, verbose=0):
    """
    Return the flat scale factor for a given time.
        These are empirically determined linear scales to the flat field
        as a function of time due to diminished detector response. They
        were determined by Tom Barlow and are in the original GALEX
        execute_pipeline but there is no published source of which I am aware.

    :param t: Time stamp(s) to retrieve the scale factor for.

    :type t: numpy.ndarray

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :param verbose: Verbosity level, a value of 0 is minimum verbosity.

    :type verbose: int

    :returns: numpy.ndarray
    """

    if verbose:
        print("Calculating flat scale for t=", t, ", and band=", band)

    (flat_correct_0, flat_correct_1, flat_correct_2) = flat_scale_parameters(
        band
    )

    t = np.array(t)

    flat_scale = (
        flat_correct_0 + (flat_correct_1 * t) + (flat_correct_2 * t) * t
    )

    # There's a bulk shift in the response after the CSP
    ix = np.where(t >= 881881215.995)

    if len(ix[0]):
        try:
            flat_scale[ix] *= 1.018
        except (TypeError, IndexError):
            # If it does not have '__getitem__' (because it's a scalar)
            flat_scale *= 1.018 if t >= 881881215.995 else 1.0

    if verbose:
        print("         flat scale = ", flat_scale)

    return flat_scale


def get_fuv_temp(eclipse: int, aspect_dir: None | str | Path = None) -> float:
    """return FUV detector temperature for a given eclipse."""
    return aspect_tables(
        eclipse=eclipse,
        tables="metadata",
        columns=["fuv_temp"],
        aspect_dir=aspect_dir
    )[0]["fuv_temp"][0].as_py()


def find_fuv_offset(
    eclipse: int,
    aspect_dir: None | str | Path = None,
) -> tuple[float, float]:
    """
    Computes NUV->FUV center offset based on lookup tables. Raises a
    ValueError if no FUV temperature was recorded in the scst (spacecraft
    state) file -- this generally indicates observations for which the FUV
    detector wasn't actually powered on.
    :param eclipse: GALEX eclipse number.
    :returns: tuple -- A two-element tuple containing the x and y offsets.
    """
    fodx_coef_0, fody_coef_0, fodx_coef_1, _ = (0.0, 0.0, 0.0, 0.0)
    fuv_temp = get_fuv_temp(eclipse, aspect_dir=aspect_dir)
    if (fuv_temp is None) or np.isnan(fuv_temp):
        raise ValueError("This is probably not a valid FUV observation.")
    print(f"Offsetting FUV image for eclipse {eclipse} at {fuv_temp} degrees.")
    fodx_coef_0 = cals.offset("x")[eclipse - 1, 1]
    fody_coef_0 = cals.offset("y")[eclipse - 1, 1]
    fodx_coef_1 = 0.0
    fody_coef_1 = 0.3597
    if (fuv_temp <= 20.0) or (fuv_temp >= 40.0):
        raise ValueError("FUV temperature out of range.")
    xoffset = fodx_coef_0 - (fodx_coef_1 * (fuv_temp - 29.0))
    yoffset = fody_coef_0 - (fody_coef_1 * (fuv_temp - 29.0))
    print(f"Setting FUV offsets to x={xoffset}, y={yoffset}")
    return xoffset, yoffset


# two components of the expensive center-and-scale step that can be
# accelerated effectively with numba

@njit(cache=True)
def plus7_mod32_minus16(array):
    """add 7, take result modulo 32, subtract 16"""
    return ((array + 7) % 32) - 16


@njit(
    (numba.int16[:], numba.int16[:], numba.int16[:], numba.int16, numba.int16, numba.int16[:], numba.int16[:]),
    cache=True
)
def center_scale_step_1(xa, yb, xb, xclk, yclk, xamc, yamc):
    """
    perform an expensive component of the center-and-scale pipeline
    """
    xraw0 = xb * xclk + xamc
    yraw0 = yb * yclk + yamc
    ya = (((yraw0 / (2 * yclk) - xraw0 / (2 * xclk)) + 10) * 32) + xa
    return xraw0, ya.astype(numba.int64), yraw0


def center_and_scale(band, data, eclipse):
    """
    center and scale photon events from a raw6 file based on mission
    calibration constants
    """
    (xclk, yclk, xcen, ycen, xscl, yscl, xslp, yslp) = clk_cen_scl_slp(
        band, eclipse
    )
    xraw0, ya, yraw0 = center_scale_step_1(
        data["xa"],
        data["yb"],
        data["xb"],
        xclk,
        yclk,
        data["xamc"],
        data["yamc"],
    )
    data["ya"] = (ya % 32).astype("int16")
    xraw = (
        xraw0 + np.array(plus7_mod32_minus16(data["xa"]), dtype="int64") * xslp
    ).astype(np.float32)
    data["x"] = ((xraw - xcen) * xscl).astype(np.float32)
    yraw = (
        yraw0 + np.array(plus7_mod32_minus16(data["ya"]), dtype="int64") * yslp
    ).astype(np.float32)
    data["y"] = ((yraw - ycen) * yscl).astype(np.float32)
    return data


def compute_shutter(timeslice, flagslice, trange, shutgap=0.05):
    ix = np.where(flagslice == 0)
    t = np.hstack([trange[0], np.unique(timeslice[ix]), trange[1]])
    ix = np.where(t[1:] - t[:-1] >= shutgap)
    return np.array(t[1:] - t[:-1])[ix].sum()


def compute_exptime(
    timeslice: np.ndarray,
    flagslice: np.ndarray,
    band: str,
    trange: Sequence[float],
) -> float:
    shutter = compute_shutter(timeslice, flagslice, trange)
    # Calculate deadtime
    model = {
        "NUV": [-0.000434730599193, 77.217817988],
        "FUV": [-0.000408075976406, 76.3000943221],
    }
    rawexpt = trange[1] - trange[0]
    rawexpt -= shutter
    if rawexpt == 0:
        return rawexpt
    ix = np.where(flagslice == 0)
    gcr = len(timeslice[ix]) / rawexpt
    feeclkratio = 0.966
    refrate = model[band][1] / feeclkratio
    scr = model[band][0] * gcr + model[band][1]
    deadtime = 1 - scr / feeclkratio / refrate
    return rawexpt * (1.0 - deadtime)
