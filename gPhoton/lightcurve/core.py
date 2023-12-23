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


def make_lightcurves(sky_arrays: Mapping, ctx: PipeContext):
    """
    make lightcurves from preprocessed structures generated from FITS images
    and movies, especially ones produced by the gPhoton.moviemaker pipeline.
    """
    if ctx.source_catalog_file is not None:
        sources = load_source_catalog(ctx.source_catalog_file, ctx.eclipse)
        if not len(sources):
            print(f"skipped photometry because no sources were found {ctx.source_catalog_file}")
            return f"skipped photometry because no sources were found {ctx.source_catalog_file}"
    else:
        sources = None
    source_table = find_sources(
        ctx.eclipse,
        ctx.band,
        str(ctx.eclipse_path()),
        sky_arrays["image_dict"],
        sky_arrays["wcs"],
        source_table=sources,
    )
    ctx.watch.click()
    # failure messages due to low exptime or no data
    if isinstance(source_table, str):
        return source_table
    if source_table is None:
        return "skipped photometry because DAOStarFinder found nothing"
    for aperture_size in ctx.aperture_sizes:
        aperture_size_px = aperture_size / c.ARCSECPERPIXEL
        photometry_table, apertures = count_full_depth_image(
            source_table,
            aperture_size_px,
            sky_arrays["image_dict"],
            sky_arrays["wcs"],
        )
        ctx.watch.click()
        if len(sky_arrays['movie_dict']) > 0:
            photometry_table = extract_photometry(
                sky_arrays["movie_dict"],
                photometry_table,
                apertures,
                ctx.threads
            )
            write_exptime_file(ctx["expfile"], sky_arrays["movie_dict"])
        photomfile = ctx(aperture=aperture_size)['photomfile']
        print(f"writing source table to {photomfile}")
        photometry_table.to_csv(photomfile, index=False)
        ctx.watch.click()

    return "successful"
