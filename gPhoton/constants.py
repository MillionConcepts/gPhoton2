"""
Mission-level constants.
"""

DETSIZE = 1.25  # Detector size in degrees
# Plate scale (in arcsec/mm)
PLTSCL = 68.754932
# Plate scale (in arcsec/micron)
ASPUM = PLTSCL / 1000.0
ARCSECPERPIXEL = 1.5
XI_XSC, XI_YSC, ETA_XSC, ETA_YSC = 0.0, 1.0, 1.0, 0.0

# values for the post-CSP detector stim scaling
# and detector constant corrections.
Mx, Bx, My, By, STIMSEP = 1, 0, 1, 0, 0
