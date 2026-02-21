import numpy as np
import warnings


def counts2mag(cps, band):
    """
    Converts GALEX counts per second to AB magnitudes.
        See: http://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html

    :param cps: The flux in counts per second.

    :type cps: float

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :returns: float -- The converted flux in AB magnitudes.
    """

    scale = 18.82 if band == 'FUV' else 20.08

    # This threw a warning if the countrate was negative which happens when
    #  the background is brighter than the source. Suppress.
    with np.errstate(invalid='ignore'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mag = -2.5 * np.log10(cps) + scale

    return mag


def mag2counts(mag, band):
    """
    Converts AB magnitudes to GALEX counts per second.
        See: http://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html

    :param mag: The AB magnitude to convert.

    :type mag: float

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :returns: float -- The converted flux in counts per second.
    """

    scale = 18.82 if band == 'FUV' else 20.08

    return 10.**(-(mag-scale)/2.5)

def counts2flux(cps, band):
    """
    Converts GALEX counts per second to flux (erg sec^-1 cm^-2 A^-1).
        See: http://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html

    :param cps: The flux in counts per second.

    :type cps: float

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :returns: float -- The converted flux in erg sec^-1 cm^-2 A^-1.
    """

    scale = 1.4e-15 if band == 'FUV' else 2.06e-16

    return scale*cps

def apcorrect1(radius, band):
    """
    Compute an apeture correction. First way. Uses the table data in Figure 4
        from Morissey, et al., 2007

    :param radius: The photometric radius, in degrees.

    :type radius: float

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :returns: float -- The aperture correction.
    """

    if not band in ['NUV', 'FUV']:
        print("Invalid band.")
        return

    aper = np.array([1.5, 2.3, 3.8, 6.0, 9.0, 12.8, 17.3, 30., 60., 90.])/3600.

    if radius > aper[-1]:
        return 0.

    if band == 'FUV':
        dmag = [1.65, 0.96, 0.36, 0.15, 0.1, 0.09, 0.07, 0.06, 0.03, 0.01]
    else:
        dmag = [2.09, 1.33, 0.59, 0.23, 0.13, 0.09, 0.07, 0.04, -0.00, -0.01]
        if radius > aper[-2]:
            return 0.

    if radius < aper[0]:
        return dmag[0]

    ix = np.where((aper-radius) >= 0.)
    x = [aper[ix[0][0]-1], aper[ix[0][0]]]
    y = [dmag[ix[0][0]-1], dmag[ix[0][0]]]
    m, C = np.polyfit(x, y, 1)

    return m*radius+C

def apcorrect2(radius, band):
    """
    Compute an aperture correction in mag based upon an aperture radius in
        degrees. Second way. Uses the data in Table 1 from
        http://www.galex.caltech.edu/researcher/techdoc-ch5.html

    :param radius: The photometric radius, in degrees.

    :type radius: float

    :param band: The band to use, either 'FUV' or 'NUV'.

    :type band: str

    :returns: float -- The aperture correction.
    """

    # [Future]: Handle arrays.

    if not band in ['NUV', 'FUV']:
        print("Invalid band.")
        return

    aper = np.array([1.5, 2.3, 3.8, 6.0, 9.0, 12.8, 17.3])/3600.

    if band == 'FUV':
        dmag = [1.65, 0.77, 0.2, 0.1, 0.07, 0.05, 0.04]
    else:
        dmag = [1.33, 0.62, 0.21, 0.12, 0.08, 0.06, 0.04]

    if radius > aper[-1]:
        # [Future]: Fix this, it isn't quite right...
        return dmag[-1]

    if radius < aper[0]:
        return dmag[0]

    ix = np.where((aper-radius) >= 0.)
    x = [aper[ix[0][0]-1], aper[ix[0][0]]]
    y = [dmag[ix[0][0]-1], dmag[ix[0][0]]]
    m, C = np.polyfit(x, y, 1)

    return m*radius+C
