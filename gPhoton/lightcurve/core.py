from pathlib import Path

import gPhoton.constants as c
from gPhoton.lightcurve._steps import (
    find_sources,
    count_full_depth_image,
    extract_photometry,
    write_exptime_file,
)
from gPhoton.reference import FakeStopwatch, eclipse_to_paths


def make_lightcurves(
    sky_arrays,
    eclipse,
    band,
    aperture_sizes,
    photonlist_path,
    source_catalog_file=None,
    threads=None,
    output_filenames=None,
    stopwatch: FakeStopwatch = FakeStopwatch(),
):
    if output_filenames is None:
        output_filenames = eclipse_to_paths(
            eclipse, Path(photonlist_path).parent, None
        )
    if source_catalog_file is not None:
        import pandas as pd

        sources = pd.read_csv(source_catalog_file)
        sources = sources.loc[sources["eclipse"] == eclipse]
        sources = sources[~sources.duplicated()].reset_index(drop=True)
    else:
        sources = None
    source_table = find_sources(
        eclipse,
        band,
        str(photonlist_path.parent),
        sky_arrays["image_dict"],
        sky_arrays["wcs"],
        source_table=sources,
    )
    stopwatch.click()
    # failure messages due to low exptime or no data
    if isinstance(source_table, str):
        return source_table
    if source_table is None:
        return "skipped photometry because DAOStarFinder found nothing"
    for aperture_size in aperture_sizes:
        aperture_size_px = aperture_size / c.ARCSECPERPIXEL
        source_table, apertures = count_full_depth_image(
            source_table,
            aperture_size_px,
            sky_arrays["image_dict"],
            sky_arrays["wcs"],
        )
        stopwatch.click()
        source_table = extract_photometry(
            sky_arrays["movie_dict"], source_table, apertures, threads
        )
        photomfile = (
            f"{output_filenames['photomfile']}"
            f"{str(aperture_size).replace('.', '_')}.csv"
        )
        print(f"writing source table to {photomfile}")
        source_table.to_csv(photomfile, index=False)
        stopwatch.click()
    write_exptime_file(output_filenames["expfile"], sky_arrays["movie_dict"])
    return "successful"
