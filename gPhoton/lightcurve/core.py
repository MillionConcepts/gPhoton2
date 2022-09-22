from pathlib import Path
from typing import Mapping, Collection

import numpy as np

import gPhoton.constants as c
from gPhoton.lightcurve._steps import (
    find_sources,
    count_full_depth_image,
    extract_photometry,
    write_exptime_file, load_source_catalog,
)
from gPhoton.reference import FakeStopwatch, eclipse_to_paths
from gPhoton.types import GalexBand


def make_lightcurves(
    sky_arrays: Mapping,
    eclipse: int,
    band: GalexBand,
    aperture_sizes: Collection[float],
    photonlist_path,
    source_catalog_file=None,
    threads=None,
    output_filenames=None,
    stopwatch: FakeStopwatch = FakeStopwatch(),
):
    """
    make lightcurves from preprocessed structures generated from FITS images
    and movies, especially ones produced by the gPhoton.moviemaker pipeline.
    """
    if output_filenames is None:
        output_filenames = eclipse_to_paths(
            eclipse, Path(photonlist_path).parent, None
        )[band]
    if source_catalog_file is not None:
        sources = load_source_catalog(source_catalog_file, eclipse)
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
        photometry_table, apertures = count_full_depth_image(
            source_table,
            aperture_size_px,
            sky_arrays["image_dict"],
            sky_arrays["wcs"],
        )
        stopwatch.click()
        photometry_table = extract_photometry(
            sky_arrays["movie_dict"], photometry_table, apertures, threads
        )
        photomfile = (
            f"{output_filenames['photomfile']}"
            f"{str(aperture_size).replace('.', '_')}.csv"
        )
        print(f"writing source table to {photomfile}")
        photometry_table.to_csv(photomfile, index=False)
        stopwatch.click()
    write_exptime_file(output_filenames["expfile"], sky_arrays["movie_dict"])
    return "successful"
