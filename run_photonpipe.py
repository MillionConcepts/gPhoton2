from pathlib import Path

from gPhoton import PhotonPipe
import gfcat.gfcat_utils as gfu


def run_photonpipe(eclipse,band='NUV',data_directory = "test_data",rerun=False):
    if eclipse>47000:
        print(f'CAUSE data w/ eclipse>47000 are not yet supported.')
        return
    raw6file = gfu.download_raw6(eclipse, band, data_directory=data_directory)
    if not raw6file:
        return
    photonfile = Path(
        data_directory,
        f"e{eclipse}",
        f"e{eclipse}-{'n' if band == 'NUV' else 'f'}d",
    )
    print(f"Photon data file: {photonfile}.parquet")
    if Path(str(photonfile)+".parquet").exists() and not rerun:
        return
    PhotonPipe.photonpipe(
        photonfile, band, raw6file=raw6file, verbose=2, chunksz=1000000,
        threads=None
    )

if __name__ == '__main__':
    run_photonpipe(42645)
    #run_this(12000)


