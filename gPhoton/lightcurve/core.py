from typing import Mapping
import warnings
import numpy as np
import pandas as pd
import gPhoton.constants as c
from gPhoton.lightcurve._steps import (
    count_full_depth_image,
    extract_photometry,
    write_exptime_file,
    load_source_catalog,
    format_source_catalog,
    check_empty_image)
from gPhoton.lightcurve.photometry_utils import (
    get_point_sources,
    check_point_in_extended,
    mask_for_extended_sources)
from gPhoton.reference import FakeStopwatch, PipeContext
from gPhoton.coadd import (zero_flag_and_edge, flag_and_edge_mask)


def make_lightcurves(sky_arrays: Mapping, ctx: PipeContext):
    """
    make lightcurves from preprocessed structures generated from FITS images
    and movies, especially ones produced by the gPhoton.moviemaker pipeline.
    """
    # check if it's an empty image before dealing with sources
    check_empty_image(ctx.eclipse, ctx.band, sky_arrays["image_dict"])

    image_dict = sky_arrays["image_dict"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # preparing images for source finding
        exptime = image_dict["exptimes"][0]
        masked_cnt_image = zero_flag_and_edge(
            image_dict["cnt"],
            image_dict["flag"],
            image_dict["edge"],
            copy=True
        ) / exptime
        masked_cnt_image = masked_cnt_image.astype(np.float32)
        flag_edge_mask = flag_and_edge_mask(
            image_dict["cnt"],
            image_dict["flag"],
            image_dict["edge"])

    if ctx.source_catalog_file is not None:
        # for input point source catalog
        sources = load_source_catalog(ctx.source_catalog_file, ctx.eclipse)
        if not len(sources):
            print(f"skipped photometry because no sources were found {ctx.source_catalog_file}")
            return f"skipped photometry because no sources were found {ctx.source_catalog_file}"
        source_table = format_source_catalog(sources, sky_arrays["wcs"])
    else:
        # if there's no input catalog
        outline_seg_map, source_table = get_point_sources(
            masked_cnt_image,
            ctx.band,
            flag_edge_mask,
            exptime)

    # set all extended sources IDs for point sources as Null unless
    # extended source finding is run
    source_table["extended_source"] = None
    source_table["extended_source"] = source_table["extended_source"].astype(pd.Int8Dtype())

    if ctx.extended_flagging:
        # find extended sources, tag point source catalog with
        # applicable extended source IDs
        masks, extended_source_cat = mask_for_extended_sources(
            masked_cnt_image,
            ctx.band,
            exptime)
        if extended_source_cat is not None:
            extended_name = ctx['extended_shapes']
            print(f"writing extended source table to {extended_name}")
            extended_source_cat.to_csv(
                extended_name, index=False  # added s and [leg]??
            )
        if ctx.source_catalog_file is None:
            # currently flagging input catalogs with extended source IDs
            # doesn't work because we use the segment map
            source_table = check_point_in_extended(
                outline_seg_map,
                masks,
                source_table)
            del outline_seg_map
        del masks
    del masked_cnt_image, flag_edge_mask
    ctx.watch.click()

    # failure messages due to low exptime or no data
    if isinstance(source_table, str):
        return source_table
    if source_table is None:
        return "skipped photometry because no sources were found"

    # photometry on point sources
    for aperture_size in ctx.aperture_sizes:
        aperture_size_px = aperture_size / c.ARCSECPERPIXEL
        photometry_table, apertures = count_full_depth_image(
            source_table,
            aperture_size_px,
            sky_arrays["image_dict"],
            sky_arrays["wcs"],
        )
        ctx.watch.click()
        if sky_arrays['movie_dict'] is not None:
            if len(sky_arrays['movie_dict']) > 0:
                photometry_table = extract_photometry(
                    sky_arrays["movie_dict"],
                    photometry_table,
                    apertures,
                    ctx.threads
                )
                write_exptime_file(ctx["expfile"], sky_arrays["movie_dict"])
            else:
                write_exptime_file(ctx["expfile"], sky_arrays["image_dict"])

        photomfile = ctx(aperture=aperture_size)['photomfile']
        print(f"writing source table to {photomfile}")
        photometry_table.to_csv(photomfile, index=False)
        ctx.watch.click()

    return "source finding and photometry successful"
