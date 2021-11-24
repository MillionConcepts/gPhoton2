"""
.. module:: CalUtils
   :synopsis: Numerous methods for calibrating the raw photon event data.
       Many of these instantiate or make use of specific detector hardware
       parameters / constants related to the "static" or detector-space event
       calibration, including walk, wiggle, linearity, post-CSP, and stim
       scaling corrections.
"""
import numpy as np

from gPhoton import cals
import gPhoton.constants as c


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

    clk2fits, clk2head = cal.clock2()
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
