from typing import Mapping

import gPhoton.constants as c
from gPhoton.lightcurve._steps import (
    find_sources,
    count_full_depth_image,
    extract_photometry,
    write_exptime_file,
    load_source_catalog
)
from gPhoton.reference import FakeStopwatch, PipeContext


def make_lightcurves(
    sky_arrays: Mapping,
    context: PipeContext,
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
        sources = load_source_catalog(source_catalog_file, context.eclipse)
    else:
        sources = None
    source_table = find_sources(
        context.eclipse,
        context.band,
        str(context.eclipse_path()),
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
    for aperture_size in context.aperture_sizes:
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
            write_exptime_file(context["expfile"], sky_arrays["movie_dict"])
        photomfile = context(aperture=aperture_size)['photomfile']
        print(f"writing source table to {photomfile}")
        photometry_table.to_csv(photomfile, index=False)
        stopwatch.click()

    return "successful"
