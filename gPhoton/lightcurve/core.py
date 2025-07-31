from typing import Mapping
from datetime import datetime, timezone
import warnings
import numpy as np
import pandas as pd
from skimage.draw import disk
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
from gPhoton.coadd import (zero_flag, flag_mask)
from gPhoton import __version__

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


        # edge mask just for source finding (not a flag, just demarcates
        # bad area for source finding @ image edge where there are no sources
        # or the coverage is not "full"
        flag_edge_mask = (image_dict["coverage"] != 1)
        # mask hotspots
        masked_cnt_image = zero_flag(
            image_dict["cnt"],
            image_dict['flag'],
            copy=True
        )
        masked_cnt_image = masked_cnt_image.astype(np.float32)
        masked_cnt_image[flag_edge_mask] = np.nan

    # use pre made source catalog file or find sources
    if ctx.source_catalog_file is not None:
        # for input point source catalog
        sources = load_source_catalog(ctx.source_catalog_file, ctx.eclipse)
        if not len(sources):
            print(f"skipped photometry because no sources were found {ctx.source_catalog_file}")
            return f"skipped photometry because no sources were found {ctx.source_catalog_file}"
        source_table = format_source_catalog(sources, sky_arrays["wcs"])
    else:
        # if there's no input catalog
        outline_seg_map, source_table, bkg_sub_cnt = get_point_sources(
            masked_cnt_image,
            flag_edge_mask,
            sky_arrays['photon_count'],
            exptime)

    # set all extended sources IDs for point sources as Null unless
    # extended source finding is run
    source_table["extended_source"] = None
    source_table["extended_source"] = source_table["extended_source"].astype(pd.Int16Dtype())

    if ctx.extended_flagging:
        # find extended sources, tag point source catalog with
        # applicable extended source IDs
        masks, extended_source_cat = mask_for_extended_sources(
            bkg_sub_cnt/exptime,
            sky_arrays['photon_count'])
        if ctx.source_catalog_file is None:
            source_table, extended_source_cat = check_point_in_extended(
                outline_seg_map,
                masks,
                source_table,
                extended_source_cat)
            # drop rows where there are no point sources ID'd
            extended_source_cat = extended_source_cat[
                ~((extended_source_cat['area_density'] == 0) | (extended_source_cat['source_count'] == 0))
            ]
            del outline_seg_map
        if extended_source_cat is not None:
            extended_name = ctx['extended_shapes']
            print(f"writing extended source table to {extended_name}")
            extended_source_cat.to_csv(
                extended_name, index=False
            )
        del masks
    del masked_cnt_image, flag_edge_mask
    ctx.watch.click()

    # failure messages due to low exptime or no data
    if isinstance(source_table, str):
        return source_table
    if source_table is None:
        return "skipped photometry because no sources were found"

    # photometry on point sources
    print(f"Len source table: {len(source_table)} ")
    for aperture_size in ctx.aperture_sizes:
        aperture_size_px = aperture_size / c.ARCSECPERPIXEL
        photometry_table, apertures = count_full_depth_image(
            source_table,
            aperture_size_px,
            sky_arrays["image_dict"],
            sky_arrays["wcs"],
            ctx
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

        photomfile = ctx(aperture=aperture_size)['photomfile']
        print(f"writing source table to {photomfile}")
        if ctx.ftype == "parquet":
            import pyarrow as pa
            # idk if pandas is a problem here
            photometry_table = pa.Table.from_pandas(photometry_table, preserve_index=False)
            metadata = {
                b"TELESCOP": b"GALEX",
                b"ECLIPSE": str(ctx.eclipse).encode(),
                b"LEG": str(ctx.leg).encode(),
                b"BANDNAME": str(ctx.band).encode(),
                b"BAND": str(1 if ctx.band == "NUV" else 2).encode(),
                b"ORIGIN": b"Million Concepts",
                b"DATE": datetime.now(timezone.utc).replace(microsecond=0).isoformat().encode(),
                b"TIMESYS": b"UTC",
                b"VERSION": f"v{__version__}".encode(),
            }
            photometry_table = photometry_table.replace_schema_metadata(metadata)
            pa.parquet.write_table(
                photometry_table,
                photomfile,
                # maximize interop with other parquet readers
                version="1.0",
                store_schema=True
            )
        elif ctx.ftype == "csv":
            photometry_table.to_csv(photomfile, index=False)
            # no metadata for csv
        else:
            raise ValueError(f"unknown photometry format '{ctx.ftype}'")
        ctx.watch.click()

    return "source finding and photometry successful"
