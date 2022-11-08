from pathlib import Path
from typing import Mapping, Collection


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
    output_filenames,
    eclipse: int,
    band: GalexBand,
    leg: int,
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
    source_table, segment_map, extended_source_paths, extended_source_cat = find_sources(
        eclipse,
        band,
        str(Path(output_filenames['photomfiles'][0]).parent),
        sky_arrays["image_dict"],
        sky_arrays["wcs"],
        source_table=sources,
    )
    print("saving segment map and extended source mask to files")
    from gPhoton.lightcurve._plot import make_source_figs
    make_source_figs(
        source_table,
        segment_map,
        sky_arrays["image_dict"]["cnt"],
        eclipse,
        band,
        outpath=str(Path(output_filenames['image']).parent),
    )
    print("saving extended source catalogue")
    extended_source_cat.to_csv(
        output_filenames["extended_catalog"], index=None
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
                output_filenames["expfiles"][leg], sky_arrays["movie_dict"]
            )
        photomfile = (
            f"{output_filenames['photomfiles'][leg]}"
            f"{str(aperture_size).replace('.', '_')}.csv"
        )
        print(f"writing source table to {photomfile}")
        photometry_table.to_csv(photomfile, index=False)
        stopwatch.click()

    return "successful"
