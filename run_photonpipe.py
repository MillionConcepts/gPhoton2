from pathlib import Path

from gPhoton.pipeline import get_photonlist

photonpath = get_photonlist(12160, "NUV", recreate=True, threads=4)


print(photonpath)
