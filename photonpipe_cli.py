"""UNFINISHED, DO NOT USE"""

from pathlib import Path

import fire

import gfcat.gfcat_utils as gfu
from gPhoton import PhotonPipe

def photonpipe_cli_endpoint(
    eclipse,band,
    data_directory="test_data",
):
    if not band in ['NUV','FUV']:
        print(f'Invalid band: {band}')
        return
    raw6file = gfu.download_raw6(eclipse, band, data_directory=data_directory)
    photonfile = Path(
        data_directory,
        f"e{eclipse}",
        f"e{eclipse}-{band.lower()[0]}d",
    )
    print(f"Photon data file: {photonfile}.parquet")
    PhotonPipe.photonpipe(
        photonfile, band, raw6file=raw6file, verbose=2, chunksz=1000000,
        threads=None
    )

if __name__ == '__main__':
    fire.Fire(photonpipe_cli_endpoint)


