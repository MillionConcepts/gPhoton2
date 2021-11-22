"""
.. module:: CalUtils
   :synopsis: Numerous methods for calibrating the raw photon event data.
       Many of these instantiate or make use of specific detector hardware
       parameters / constants related to the "static" or detector-space event
       calibration, including walk, wiggle, linearity, post-CSP, and stim
       scaling corrections.
"""

from __future__ import absolute_import, division, print_function

# Core and Third Party imports.
from astropy.io import fits as pyfits
from builtins import str
from builtins import range
import csv
import numpy as np
import pandas as pd

# gPhoton imports.
import gPhoton.cal as cal
import gPhoton.constants as c
from gPhoton.galextools import isPostCSP
from gPhoton.MCUtils import rms, print_inline, get_fits_header, get_tbl_data

# ------------------------------------------------------------------------------
def clk_cen_scl_slp(band, eclipse):
    """
    Return the detector clock, center, scale, and slope constants. These are
        empirically determined detector-space calibration parameters that help
        define the relationship between raw event data and its physical position
        on the detector.

    :param band: The band to return the constants for, either 'FUV' or 'NUV'.

    :type band: str

    :param eclipse: The eclipse number to return the constants for.

    :type eclipse: int

    :returns: tuple -- a tuple containing the x-clock, y-clock, x-center,
        y-center, x-scale, y-scale, x-slope, and y-slope constants.
    """

    band = band.upper()

    if band == "FUV":
        xclk, yclk = 1997.0, 1993.0
        xcen, ycen = 7200.0, 6670.0
        xscl, yscl = 7.78, 10.73
        xslp, yslp = 0.0, 0.0
    elif band == "NUV":
        xclk, yclk = 2007.0, 1992.0
        # Special value for post-CSP event.
        if eclipse >= 38150:
            yclk = 2016.0
        xcen, ycen = 7400.0, 6070.0
        xscl, yscl = 8.79, 14.01
        xslp, yslp = 0.53, 0.0
    else:
        raise ValueError("Band must be either fuv or nuv ... Exiting.")

    return xclk, yclk, xcen, ycen, xscl, yscl, xslp, yslp


# -----------------------------------------------------------------------------

def post_csp_caldata():
    """
    Loads the calibration data for after the CSP event.

    :returns: tuple -- A six-element tuple containing the wiggle, walk, and
        clock corrections. See the calibration paper for details.
    """

    print("Loading post-CSP wiggle file...")

    wig2fits, wig2head = cal.wiggle2()
    wig2 = np.zeros([128, 8, 32, 8])

    wig2[
        wig2fits['yy'], wig2fits['YB'], wig2fits['YA'], wig2fits['XB']
    ] = wig2fits['ycor']

    wig2data = {"start": wig2head["Y_AS_0"], "inc": wig2head["Y_AS_INC"]}

    print("Loading post-CSP walk file...")

    wlk2fits, wlk2head = cal.walk2()
    wlk2 = np.zeros([100, 8, 32])

    wlk2[wlk2fits['yy'], wlk2fits['yb'], wlk2fits['q']] = wlk2fits['ycor']

    wlk2data = {"start": wlk2head["Y_AS_0"], "inc": wlk2head["Y_AS_INC"]}

    print("Loading post-CSP clock file...")

    clk2fits, clk2head = cal.clock2()
    clk2 = np.zeros([100, 8])

    clk2[clk2fits['yy'], clk2fits['YB']] = clk2fits['ycor']

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
        if eclipse >= 38268:
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

    yap = np.array(yap, dtype="int64") % 32

    return yap


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
    if eclipse <= 37460:
        return yac

    yap = rtaph_yap(ya, yb, yamc)

    yac = yactbl[yap, yb]

    return yac


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

# ------------------------------------------------------------------------------
def compute_stimstats(raw6file, band, eclipse):
    """
    Computes statistics for stim events for the post-CSP stim-based correction.

    :param raw6file: The name of the raw6 FITS file to read.

    :type raw6file: str

    :param band: The band to return the stim data for, either 'FUV' or 'NUV'.

    :type band: str

    :param eclipse: The eclipse number to return the stim data for.

    :type eclipse: int

    :returns: tuple -- Six-element tuple containing information on the trend
        in relative positions of the stims over time used for the Post-CSP
        stim correction.
    """

    print("Computing stim statistics and post-CSP corrections...")

    # Plate scale (in arcsec/mm).
    pltscl = 68.754932
    # Plate scale in arcsec per micron.
    c.ASPUM = pltscl / 1000.0
    # [Future] If these are made module constants, can remove from this method.

    # Read in stim data from the FITS file.
    stim1, stim2, stim3, stim4 = raw6_to_stims(raw6file, band, eclipse)

    # Compute the mean positions (in arcseconds).
    stim1avg = [stim1["x"].mean() * c.ASPUM, stim1["y"].mean() * c.ASPUM]
    stim2avg = [stim2["x"].mean() * c.ASPUM, stim2["y"].mean() * c.ASPUM]
    stim3avg = [stim3["x"].mean() * c.ASPUM, stim3["y"].mean() * c.ASPUM]
    stim4avg = [stim4["x"].mean() * c.ASPUM, stim4["y"].mean() * c.ASPUM]

    print(
        "Init: Number of stim photons:",
        len(stim1["t"]),
        len(stim2["t"]),
        len(stim3["t"]),
        len(stim4["t"]),
    )
    print(
        "Init: Mean x values at stim positions (arcsec):",
        stim1avg[0],
        stim2avg[0],
        stim3avg[0],
        stim4avg[0],
    )
    print(
        "Init: Mean x values at stim positions (arcsec):",
        stim1avg[1],
        stim2avg[1],
        stim3avg[1],
        stim4avg[1],
    )
    print(
        "Init: Mean y values at stim positions (micron):",
        stim1avg[1] / c.ASPUM,
        stim2avg[1] / c.ASPUM,
        stim3avg[1] / c.ASPUM,
        stim4avg[1] / c.ASPUM,
    )

    # Compute the RMS around the mean (in arcseconds).
    stim1rms = [rms(stim1["x"] * c.ASPUM), rms(stim1["y"] * c.ASPUM)]
    stim2rms = [rms(stim2["x"] * c.ASPUM), rms(stim2["y"] * c.ASPUM)]
    stim3rms = [rms(stim3["x"] * c.ASPUM), rms(stim3["y"] * c.ASPUM)]
    stim4rms = [rms(stim4["x"] * c.ASPUM), rms(stim4["y"] * c.ASPUM)]

    # Compute the stim separation.
    stimsep = (
        (stim2avg[0] - stim1avg[0])
        + (stim4avg[0] - stim3avg[0])
        + (stim1avg[1] - stim3avg[1])
        + (stim2avg[1] - stim4avg[1])
    ) / 4.0
    print(
        "Init: RMS  x values at stim positions (arcsec):",
        stim1rms[0],
        stim2rms[0],
        stim3rms[0],
        stim4rms[0],
    )
    print(
        "Init: RMS  y values at stim positions (arcsec):",
        stim1rms[1],
        stim2rms[1],
        stim3rms[1],
        stim4rms[1],
    )
    print(
        "Init: (arcsec): Stim sep =",
        stimsep,
        "    Average: X RMS =",
        (stim1rms[0] + stim2rms[0] + stim3rms[0] + stim4rms[0]) / 4.0,
        "        Y RMS =",
        (stim1rms[1] + stim2rms[1] + stim3rms[1] + stim4rms[1]) / 4.0,
    )
    print("Raw stim separation is", stimsep)

    # Compute means and RMS values for each stim for each YA value stim1.
    for ya in range(32):
        ix = (stim1["ya"] == ya).nonzero()[0]
        ix = (stim2["ya"] == ya).nonzero()[0]
        ix = (stim3["ya"] == ya).nonzero()[0]
        ix = (stim4["ya"] == ya).nonzero()[0]

    # This returns the pre-CSP stim positions (because eclipse==0).
    avgstim = avg_stimpos(band, 0)

    # Compute Y scale and shift factors: yprime_as = (m * y_as) + B.
    y1, y2 = (stim1avg[1] + stim2avg[1]) / 2.0, (
        stim3avg[1] + stim4avg[1]
    ) / 2.0
    Y1, Y2 = (
        (avgstim["y1"] + avgstim["y2"]) / 2.0,
        (avgstim["y3"] + avgstim["y4"]) / 2.0,
    )
    My = (Y1 - Y2) / (y1 - y2)
    By = (Y1 - My * y1) / c.ASPUM
    print("Init: FODC: Y scale and shift (microns): My=", My, "By=", By)

    # Compute Y scale and shift factors: yprime_as = (m * y_as) + B.
    x1, x2 = (stim1avg[0] + stim3avg[0]) / 2.0, (
        stim2avg[0] + stim4avg[0]
    ) / 2.0
    X1, X2 = (
        (avgstim["x1"] + avgstim["x3"]) / 2.0,
        (avgstim["x2"] + avgstim["x4"]) / 2.0,
    )
    Mx = (X1 - X2) / (x1 - x2)
    Bx = (X1 - Mx * x1) / c.ASPUM
    print("Init: FODC: X scale and shift (microns): Mx=", Mx, "Bx=", Bx)

    stim1["xs"] = stim1["x"] * Mx + Bx
    stim1["ys"] = stim1["y"] * My + By
    stim2["xs"] = stim2["x"] * Mx + Bx
    stim2["ys"] = stim2["y"] * My + By
    stim3["xs"] = stim3["x"] * Mx + Bx
    stim3["ys"] = stim3["y"] * My + By
    stim4["xs"] = stim4["x"] * Mx + Bx
    stim4["ys"] = stim4["y"] * My + By

    # Compute the new mean positions (in arcseconds).
    stim1avgs = [stim1["xs"].mean() * c.ASPUM, stim1["ys"].mean() * c.ASPUM]
    stim2avgs = [stim2["xs"].mean() * c.ASPUM, stim2["ys"].mean() * c.ASPUM]
    stim3avgs = [stim3["xs"].mean() * c.ASPUM, stim3["ys"].mean() * c.ASPUM]
    stim4avgs = [stim4["xs"].mean() * c.ASPUM, stim4["ys"].mean() * c.ASPUM]

    print(
        "Scal: Number of stim photons:",
        len(stim1["xs"]),
        len(stim2["xs"]),
        len(stim3["xs"]),
        len(stim4["xs"]),
    )
    print(
        "Scal: Mean x values at stim positions (arcsec):",
        stim1avgs[0],
        stim2avgs[0],
        stim3avgs[0],
        stim4avgs[0],
    )
    print(
        "Scal: Mean y values at stim positions (arcsec):",
        stim1avgs[1],
        stim2avgs[1],
        stim3avgs[1],
        stim4avgs[1],
    )
    print(
        "Scal: Mean y values at stim positions (microns):",
        stim1avgs[1] / c.ASPUM,
        stim2avgs[1] / c.ASPUM,
        stim3avgs[1] / c.ASPUM,
        stim4avgs[1] / c.ASPUM,
    )

    # Compute the new RMS around the mean (in arcseconds).
    stim1rmss = [rms(stim1["xs"] * c.ASPUM), rms(stim1["ys"] * c.ASPUM)]
    stim2rmss = [rms(stim2["xs"] * c.ASPUM), rms(stim2["ys"] * c.ASPUM)]
    stim3rmss = [rms(stim3["xs"] * c.ASPUM), rms(stim3["ys"] * c.ASPUM)]
    stim4rmss = [rms(stim4["xs"] * c.ASPUM), rms(stim4["ys"] * c.ASPUM)]

    # Compute the stim separation.
    stimseps = (
        (stim2avgs[0] - stim1avgs[0])
        + (stim4avgs[0] - stim3avgs[0])
        + (stim1avgs[1] - stim3avgs[1])
        + (stim2avgs[1] - stim4avgs[1])
    ) / 4.0
    print(
        "Scal: RMS  x values at stim positions (arcsec):",
        stim1rmss[0],
        stim2rmss[0],
        stim3rmss[0],
        stim4rmss[0],
    )
    print(
        "Init: RMS  y values at stim positions (arcsec):",
        stim1rmss[1],
        stim2rmss[1],
        stim3rmss[1],
        stim4rmss[1],
    )
    print(
        "Init: (arcsec): Stim sep =",
        stimseps,
        "   Average: X RMS =",
        (stim1rmss[0] + stim2rmss[0] + stim3rmss[0] + stim4rmss[0]) / 4.0,
        "    Y RMS =",
        (stim1rmss[1] + stim2rmss[1] + stim3rmss[1] + stim4rmss[1]) / 4.0,
    )

    # Fit straight line to YA>2 and YB==2 points.
    # This could be written more efficiently, but it's an attempt at a faithful
    #  port of the GALEX code (from Tom Barlow) which was written in C.
    ix1 = ((stim1["ya"] > 2) & (stim1["yb"] == 2)).nonzero()[0]
    ix2 = ((stim2["ya"] > 2) & (stim2["yb"] == 2)).nonzero()[0]
    ix3 = ((stim3["ya"] > 2) & (stim3["yb"] == 2)).nonzero()[0]
    ix4 = ((stim4["ya"] > 2) & (stim4["yb"] == 2)).nonzero()[0]
    w8 = np.ones(len(ix1) + len(ix2) + len(ix3) + len(ix4))
    x8 = np.concatenate(
        (
            stim1["yap"][ix1],
            stim2["yap"][ix2],
            stim3["yap"][ix3],
            stim4["yap"][ix4],
        ),
        axis=0,
    )
    y8 = np.concatenate(
        (
            stim1["ys"][ix1] - stim1avgs[1] / c.ASPUM,
            stim2["ys"][ix2] - stim2avgs[1] / c.ASPUM,
            stim3["ys"][ix3] - stim3avgs[1] / c.ASPUM,
            stim4["ys"][ix4] - stim4avgs[1] / c.ASPUM,
        ),
        axis=0,
    )
    print("NOTE: Found,", len(w8), "points for YA correction fit.")

    yac_coef1, yac_coef0 = np.polyfit(x8, y8, 1)

    print("Scal: YA correction coef for YB=2:", yac_coef0, yac_coef1)

    # Compute yb shift factors == zero for all.
    yac_ybs = np.zeros(8)
    coef0_yb = np.zeros(8) + yac_coef0
    coef1_yb = np.zeros(8) + yac_coef1

    # Set user slope adjustment. Use best slope adjustments from September 2010.
    # YB == 2...
    slope_scale = 1.04
    print("NOTE: Using slope scale of,", slope_scale, "for YB==2.")
    rr1 = yac_coef1 * slope_scale
    rr0 = (yac_coef0 + (16.0 * yac_coef1)) - (16.0 * rr1)
    coef0_yb[2] = rr0
    coef1_yb[2] = rr1
    print("New: YA correction coef (YB==2):", coef0_yb[2], coef1_yb[2])

    # YB == 3,4...
    slope_scale = 1.06
    print("NOTE: Using slope scale of,", slope_scale, "for YB==3.")
    rr1 = yac_coef1 * slope_scale
    rr0 = (yac_coef0 + (16.0 * yac_coef1)) - (16.0 * rr1)
    coef0_yb[3] = rr0
    coef1_yb[3] = rr1
    coef0_yb[4] = rr0
    coef1_yb[4] = rr1
    print("New: YA correction coef (YB==3):", coef0_yb[3], coef1_yb[3])
    print("NOTE: Using slope scale of,", slope_scale, "for YB==4.")
    print("New: YA correction coef (YB==4):", coef0_yb[4], coef1_yb[4])

    # Fill in look up array.
    yac = np.zeros([40, 8])
    for yb in range(8):
        for ya in range(40):
            yac[ya][yb] = (
                coef0_yb[yb] + (float(ya) * coef1_yb[yb])
            ) + yac_ybs[yb]

    stim1["yac"] = yac[
        np.array(stim1["yap"], dtype="int64"),
        np.array(stim1["yb"], dtype="int64"),
    ]
    stim2["yac"] = yac[
        np.array(stim2["yap"], dtype="int64"),
        np.array(stim2["yb"], dtype="int64"),
    ]
    stim3["yac"] = yac[
        np.array(stim3["yap"], dtype="int64"),
        np.array(stim3["yb"], dtype="int64"),
    ]
    stim4["yac"] = yac[
        np.array(stim4["yap"], dtype="int64"),
        np.array(stim4["yb"], dtype="int64"),
    ]

    # [Future] The section below could be re-written more elegantly.
    # [Future] Does this return the correct values for YB==1?
    for yb in range(8):
        ix = ((stim1["yb"] == yb) & (stim1["ya"] > 4)).nonzero()[0]
        s1m = ((stim1["ys"] - stim1["yac"])[ix] * c.ASPUM).mean()
        s1r = rms((stim1["ys"] - stim1["yac"])[ix] * c.ASPUM)
        if len(ix) > 0:
            print(
                "Corrected stim 1: YB=",
                yb,
                " Num=",
                len(ix),
                " Mean=",
                s1m,
                " RMS=",
                s1r,
            )
    for yb in range(8):
        ix = ((stim2["yb"] == yb) & (stim2["ya"] > 4)).nonzero()[0]
        s2m = ((stim2["ys"] - stim2["yac"])[ix] * c.ASPUM).mean()
        s2r = rms((stim2["ys"] - stim2["yac"])[ix] * c.ASPUM)
        if len(ix) > 0:
            print(
                "Corrected stim 2: YB=",
                yb,
                " Num=",
                len(ix),
                " Mean=",
                s2m,
                " RMS=",
                s2r,
            )
    for yb in range(8):
        ix = ((stim3["yb"] == yb) & (stim3["ya"] > 4)).nonzero()[0]
        s3m = ((stim3["ys"] - stim3["yac"])[ix] * c.ASPUM).mean()
        s3r = rms((stim3["ys"] - stim3["yac"])[ix] * c.ASPUM)
        if len(ix) > 0:
            print(
                "Corrected stim 3: YB=",
                yb,
                " Num=",
                len(ix),
                " Mean=",
                s3m,
                " RMS=",
                s3r,
            )
    for yb in range(8):
        ix = ((stim4["yb"] == yb) & (stim4["ya"] > 4)).nonzero()[0]
        s4m = ((stim4["ys"] - stim4["yac"])[ix] * c.ASPUM).mean()
        s4r = rms((stim4["ys"] - stim4["yac"])[ix] * c.ASPUM)
        if len(ix) > 0:
            print(
                "Corrected stim 4: YB=",
                yb,
                " Num=",
                len(ix),
                " Mean=",
                s4m,
                " RMS=",
                s4r,
            )

    return Mx, Bx, My, By, stimsep, yac


# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def create_ssd(raw6file, band, eclipse, ssdfile=None):
    """
    Creates a Stim Separation Data (SSD) table file.

    :param raw6file: The name of the raw6 FITS file to read.

    :type raw6file: str

    :param band: The band to create the SSD file for, either 'FUV' or 'NUV'.

    :type band: str

    :param eclipse: The eclipse number to create the SSD file for.

    :type eclipse: int

    :param ssdfile: Name of stim separation data (SSD) output file to create.

    :type ssdfile: str

    :returns: tuple -- 2xN tuple containing slope and intercept of
        stim positions over time.
    """
    if ssdfile:
        print("Preparing SSD output file " + ssdfile)
        tbl = csv.writer(
            open(ssdfile, "wb"),
            delimiter=" ",
            quotechar="|",
            quoting=csv.QUOTE_MINIMAL,
        )
        tbl.writerow(["|sct", "|stim_sep", "|stim_num", "|sep_fit"])

    stim1, stim2, stim3, stim4 = raw6_to_stims(
        raw6file, band, eclipse, margin=20.0
    )
    # if not (stim1['t'].shape[0] and stim2['t'].shape[0] and
    #         stim3['t'].shape[0] and stim4['t'].shape[0]):
    #     # Missing a stim. Try again with the default margin of 90.
    #     stim1, stim2, stim3, stim4 = raw6_to_stims(raw6file, band, eclipse)
    #     if not (stim1['t'].shape[0] and stim2['t'].shape[0] and
    #             stim3['t'].shape[0] and stim4['t'].shape[0]):
    #         raise ValueError('Unable to locate a stim. Cant create SSD.')

    stimt = np.concatenate(
        [stim1["t"], stim2["t"], stim3["t"], stim4["t"]], axis=0
    )
    sortt = np.argsort(stimt)
    stimt = stimt[sortt]
    stimix = np.concatenate(
        [
            np.zeros(len(stim1["t"])) + 1,
            np.zeros(len(stim2["t"])) + 2,
            np.zeros(len(stim3["t"])) + 3,
            np.zeros(len(stim4["t"])) + 4,
        ],
        axis=0,
    )[sortt]
    stimx_as = (
        np.concatenate(
            [stim1["x"], stim2["x"], stim3["x"], stim4["x"]], axis=0
        )
        * c.ASPUM
    )[sortt]
    stimy_as = (
        np.concatenate(
            [stim1["y"], stim2["y"], stim3["y"], stim4["y"]], axis=0
        )
        * c.ASPUM
    )[sortt]
    pinc = 1000
    avt, sep, num = [], [], []

    for i in range(0, len(stimt) - pinc, pinc):
        ix1 = (stimix[i : i + pinc] == 1).nonzero()[0]
        ix2 = (stimix[i : i + pinc] == 2).nonzero()[0]
        ix3 = (stimix[i : i + pinc] == 3).nonzero()[0]
        ix4 = (stimix[i : i + pinc] == 4).nonzero()[0]
        sx1, sy1 = (
            np.mean(stimx_as[i : i + pinc][ix1]),
            np.mean(stimy_as[i : i + pinc][ix1]),
        )
        sx2, sy2 = (
            np.mean(stimx_as[i : i + pinc][ix2]),
            np.mean(stimy_as[i : i + pinc][ix2]),
        )
        sx3, sy3 = (
            np.mean(stimx_as[i : i + pinc][ix3]),
            np.mean(stimy_as[i : i + pinc][ix3]),
        )
        sx4, sy4 = (
            np.mean(stimx_as[i : i + pinc][ix4]),
            np.mean(stimy_as[i : i + pinc][ix4]),
        )
        stim_sep = (
            (sx2 - sx1) + (sx4 - sx3) + (sy1 - sy3) + (sy2 - sy4)
        ) / 4.0
        stim_avt = sum(stimt[i : i + pinc]) / len(stimt[i : i + pinc])
        stim_num = len(ix1) + len(ix2) + len(ix3) + len(ix4)
        avt.append(stim_avt)
        sep.append(stim_sep)
        num.append(stim_num)

    m, C = np.polyfit(avt, sep, 1)
    fit = C + np.array(avt) * m

    if ssdfile:
        for i in range(len(avt)):
            tbl.writerow([avt[i], sep[i], num[i], fit[i]])

    return C, m


# ------------------------------------------------------------------------------
