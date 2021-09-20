from pathlib import Path
from typing import Mapping

from gPhoton.gphoton_utils import pyfits_zopen
from gPhoton.photometry import find_sources
from gfcat.gfcat_utils import read_image


def run_source_finder(
    eclipse, band="NUV", image_dict: Mapping = None, image_fn: str = None
):
    pass



if __name__ == "__main__":
    eclipse = 22650
    band = "NUV"
    data_directory = Path("test_data", f"e{eclipse}")
    cnt = pyfits_zopen(Path(data_directory, "-full-cnt.fits.zstd"))
    flag = pyfits_zopen(Path(data_directory, "-full-flag.fits.zstd"))
    edge = pyfits_zopen(Path(data_directory, "-full-edge.fits.zstd"))
    cntmap, flagmap, edgemap, wcs, tranges, exptimes = read_image(cntfilename)