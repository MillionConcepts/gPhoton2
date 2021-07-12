"""
Mission-level constants.
"""

DETSIZE = 1.25  # Detector size in degrees
# Plate scale (in arcsec/mm)
PLTSCL = 68.754932
# Plate scale (in arcsec/micron)
ASPUM = PLTSCL / 1000.0
ARCSECPERPIXEL = 1.5
DEGPERPIXEL = 0.000416666666666667
XI_XSC, XI_YSC, ETA_XSC, ETA_YSC = 0.0, 1.0, 1.0, 0.0
PIXEL_SIZE = 0.00166666666666667
PIXELS_PER_AXIS = 800
FILL_VALUE = DETSIZE / (PIXEL_SIZE * PIXELS_PER_AXIS)


# values for the post-CSP detector stim scaling
# and detector constant corrections.
Mx, Bx, My, By, STIMSEP = 1, 0, 1, 0, 0


