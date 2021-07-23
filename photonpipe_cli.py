"""UNFINISHED, DO NOT USE"""

from pathlib import Path

import fire

import gfcat.gfcat_utils as gfu
from gPhoton.PhotonPipe import photonpipe




def photonpipe_cli_endpoint(
    eclipse,
    raw6_source=None,
    raw6_output=None,

):
    eclipse = 23456
    band = "NUV"
    data_directory = "test_data"
    raw6file = gfu.download_raw6(eclipse, band, data_directory=data_directory)
    photonfile = Path(
        data_directory,
        f"e{eclipse}",
        f"e{eclipse}-{'n' if band == 'NUV' else 'f'}d",
    )
    print(f"Photon data file: {photonfile}")
    photonpipe(
        photonfile, band, raw6file=raw6file, verbose=2, chunksz=1000000,
        threads=4
    )

if __name__ == '__main__':
    from gPhoton.PhotonPipe import photonpipe
    fire.Fire(photonpipe_cli_endpoint)


