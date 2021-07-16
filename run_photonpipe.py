from pathlib import Path

from gPhoton import PhotonPipe
import gfcat.gfcat_utils as gfu


eclipse = 12000

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
    photonfile, band, raw6file=raw6file, verbose=2, chunksz=1000000, threads=4
)

