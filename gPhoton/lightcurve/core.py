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
    else:
        sources = None
    source_table = find_sources(
        ctx, sky_arrays["image_dict"], sky_arrays["wcs"], source_table=sources,
    )
    ctx.watch.click()
    if ctx.do_background is True:
        from gPhoton.background import make_background_mask

        print("making background mask")
        sky_arrays['image_dict']["bg_mask"] = make_background_mask(
            source_table,
            sky_arrays['image_dict']['cnt'].shape,
            **ctx.bg_params,
            threads=ctx.threads
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
