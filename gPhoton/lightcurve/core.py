from pathlib import Path
from typing import Mapping, Collection, Callable

import gPhoton.constants as c
from gPhoton.lightcurve._steps import (
    find_sources,
    count_full_depth_image,
    extract_photometry,
    write_exptime_file,
    load_source_catalog
)
from gPhoton.reference import FakeStopwatch
from gPhoton.types import GalexBand


def make_lightcurves(
    sky_arrays: Mapping,
    e2p: Callable,
    eclipse: int,
    band: GalexBand,
    aperture_sizes: Collection[float],
    source_catalog_file=None,
    threads=None,
    stopwatch: FakeStopwatch = FakeStopwatch(),
    **_unused_options
):
    """
    make lightcurves from preprocessed structures generated from FITS images
    and movies, especially ones produced by the gPhoton.moviemaker pipeline.
    """
    if source_catalog_file is not None:
        sources = load_source_catalog(source_catalog_file, eclipse)
    else:
        sources = None
    source_table = find_sources(
        eclipse,
        band,
        str(Path(e2p()['photomfile']).parent),
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
        photometry_table, apertures = count_full_depth_image(
            source_table,
            aperture_size_px,
            sky_arrays["image_dict"],
            sky_arrays["wcs"],
        )
        stopwatch.click()
        if len(sky_arrays['movie_dict']) > 0:
            photometry_table = extract_photometry(
                sky_arrays["movie_dict"], photometry_table, apertures, threads
            )
            write_exptime_file(
                e2p()["expfile"], sky_arrays["movie_dict"]
            )
        photomfile = e2p(aperture=aperture_size)['photomfile']
        print(f"writing source table to {photomfile}")
        photometry_table.to_csv(photomfile, index=False)
        stopwatch.click()

    return "successful"
