from pathlib import Path
import re
from typing import Union

import pandas as pd
from photutils import DAOStarFinder, CircularAperture, aperture_photometry

from gfcat.gfcat_utils import read_image


def find_sources(
    eclipse: int, datapath: str, image_dict, wcs, band: str = "NUV"
):
    # TODO, maybe: pop these into a handler function
    if not image_dict["cnt"].max():
        print(f"{eclipse} appears to contain nothing in {band}.")
        Path(Path(datapath).parent, f"No{band}").touch()
        return
    if image_dict["exptime"] < 600:
        print("Skipping low exposure time visit.")
        Path(f"{Path(datapath).parent}", "LowExpt").touch()
        return
    daofind = DAOStarFinder(fwhm=5, threshold=0.01)
    sources = daofind(image_dict["cnt"] / image_dict["exptime"])
    try:
        print(f"Located {len(sources)} sources.")
    except TypeError:
        print(f"{eclipse} {band} contains no sources.")
        Path(Path(datapath).parent, f"/No{band}").touch()
        return
    positions = (sources["xcentroid"], sources["ycentroid"])
    apertures = CircularAperture(positions, r=8.533333333333326)
    phot_table = aperture_photometry(image_dict["cnt"], apertures).to_pandas()
    flag_table = aperture_photometry(image_dict["flag"], apertures).to_pandas()
    edge_table = aperture_photometry(image_dict["edge"], apertures).to_pandas()
    phot_visit = pd.concat([
        sources.to_pandas(),
        phot_table[["xcenter", "ycenter", "aperture_sum"]]
    ])
    phot_visit["aperture_sum_mask"] = flag_table["aperture_sum"]
    phot_visit["aperture_sum_edge"] = edge_table["aperture_sum"]
    phot_visit["ra"] = [
        wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()[0]
        for pos in apertures.positions
    ]
    phot_visit["dec"] = [
        wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()[1]
        for pos in apertures.positions
    ]
    return phot_visit

#     for i, frame in enumerate(movmap):
#         mc.print_inline("Extracting photometry from frame #{i}".format(i=i))
#         phot_visit["aperture_sum_{i}".format(i=i)] = (
#             aperture_photometry(frame, apertures)
#             .to_pandas()["aperture_sum"]
#             .tolist()
#         )
#     photomfile = re.sub(r"-\w\w\.parquet", "-photom.csv", listfile)
#     print("Writing data to {f}".format(f=photomfile))
#     phot_visit.to_csv(photomfile, index=False)
#     pd.DataFrame(
#         {
#             "expt": exptimes,
#             "t0": np.array(tranges)[:, 0].tolist(),
#             "t1": np.array(tranges)[:, 1].tolist(),
#         }
#     ).to_csv(cntfilename.replace("-cnt.fits.gz", "-exptime.csv"), index=False)
#
#
# return
