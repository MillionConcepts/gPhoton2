"""
methods for generating lightcurves from FITS images/movies, especially those
produced by the gPhoton.moviemaker pipeline. They are principally intended
for use as components of the primary gPhoton.lightcurve pipeline, called as
part of the course of running gPhoton.lightcurve.core.make_lightcurves(), and
may not suitable for independent use.
"""
import gc

import astropy.wcs
import numpy as np
import pandas as pd
import scipy.sparse

from multiprocessing import Pool
from pathlib import Path
from typing import Union, Optional, Mapping
import warnings

from photutils import CircularAperture, aperture_photometry

from gPhoton._dev.profile import pm, di, print_references
from gPhoton.highsearcher import LocalHIGHSlocal
from gPhoton.pretty import print_inline
from gPhoton.types import Pathlike

from astropy.table import Table
from astropy.convolution import convolve


def count_full_depth_image(
    source_table: pd.DataFrame,
    aperture_size: float,
    image_dict: Mapping[str, np.ndarray],
    system: astropy.wcs.WCS
):
    source_table = source_table.reset_index(drop=True)
    positions = source_table[["xcentroid", "ycentroid"]].to_numpy()
    print(f"length of positions: {len(positions)}")
    print(f"length of source table: {len(source_table)}")

    apertures = CircularAperture(positions, r=aperture_size)
    print(f"length of apertures: {len(apertures)}")

    print("Performing aperture photometry on primary image.")
    phot_table = aperture_photometry(image_dict["cnt"], apertures).to_pandas()
    print("Performing aperture photometry on flag maps.")
    flag_table = aperture_photometry(image_dict["flag"], apertures).to_pandas()
    print("Performing aperture photometry on edge maps.")
    edge_table = aperture_photometry(image_dict["edge"], apertures).to_pandas()
    source_table = pd.concat(
        [source_table, phot_table[["xcenter", "ycenter", "aperture_sum"]]],
        axis=1
    )
    source_table["aperture_sum_mask"] = flag_table["aperture_sum"]
    source_table["aperture_sum_edge"] = edge_table["aperture_sum"]
    # TODO: this isn't necessary for specified catalog positions. but
    #  should we do a sanity check?
    if "ra" not in source_table.columns:
        world = [
            system.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()
            for pos in apertures.positions
        ]
        print(f"{len(world)} world length")
        source_table["ra"] = [coord[0] for coord in world]
        source_table["dec"] = [coord[1] for coord in world]
    return source_table, apertures


def find_sources(
    eclipse: int,
    band: str,
    datapath: Union[str, Path],
    image_dict,
    wcs,
    source_table: Optional[pd.DataFrame] = None
):
    from gPhoton.coadd import zero_flag_and_edge, flag_and_edge_mask
    if not image_dict["cnt"].max():
        print(f"{eclipse} appears to contain nothing in {band}.")
        Path(datapath, f"No{band}").touch()
        return f"{eclipse} appears to contain nothing in {band}."
    exptime = image_dict["exptimes"][0]
    if source_table is None:
        print("Extracting extended and point sources.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # for some reason image segmentation still finds sources in the
            # masked cnt image? so we mask twice: once by zeroing and then
            # again by making a bool mask and feeding it to image segmentation

            masked_cnt_image = zero_flag_and_edge(
                image_dict["cnt"],
                image_dict["flag"],
                image_dict["edge"],
                copy=True
            ) / exptime
            masked_cnt_image = masked_cnt_image.astype(np.float32)
            # masked_cnt_image /= exptime
            flag_edge_mask = flag_and_edge_mask(
                image_dict["cnt"],
                image_dict["flag"],
                image_dict["edge"])
            image_dict["cnt"] = scipy.sparse.coo_matrix(image_dict["cnt"])
            source_table, segment_map, extended_source_paths, extended_source_cat = \
                get_point_and_extended_sources(masked_cnt_image, band, flag_edge_mask, exptime)
            del masked_cnt_image
            gc.collect()
            image_dict["cnt"] = image_dict["cnt"].toarray()
        try:
            print(f"Located {len(source_table)} sources")
        except TypeError:
            print(f"{eclipse} {band} contains no sources.")
            Path(datapath, f"No{band}").touch()
            return None, None
    else:
        print(f"Using specified catalog of {len(source_table)} sources.")
        positions = np.vstack(
            [
                wcs.wcs_world2pix([position], 1, ra_dec_order=True)
                for position in source_table[["ra", "dec"]].values
            ]
        )
        source_table[["xcentroid", "ycentroid"]] = positions
    return source_table, segment_map, extended_source_paths, extended_source_cat


def get_point_and_extended_sources(cnt_image: np.ndarray, band: str, f_e_mask, exposure_time):

    """
    Main function for extracting point and extended sources from an eclipse.
    Image segmentation and point source extraction occurs in a separate function.
    The threshold for being a source in NUV is set at 1.5 times the background
    rms values (2d array), while for FUV it is 3 times. The FUV setting isn't
    ~perfect~.
    Extended source extraction occurs in helper functions.
    """

    print(f"{exposure_time} s exposure time")

    # cnt_image is no longer background subtracted
    # DAO threshold is now based on power law relationship with exposure time
    print("Masking for extended sources.")
    masks, extended_source_cat = mask_for_extended_sources(cnt_image, band)
    print("Segmenting and deblending extended sources.")
    deblended_seg_map, outline_seg_map, seg_sources = image_segmentation(cnt_image, band, f_e_mask, exposure_time)
    print("Checking for extended source overlap with point sources.")
    # checking for overlap between extended source mask and segmentation image
    seg_sources = check_point_in_extended(outline_seg_map, masks, seg_sources)

    #seg_sources.dropna() was old setting
    return seg_sources, deblended_seg_map, masks, extended_source_cat

# This is a copy of the 
# photutils.segmentation.segmentation.SegmentationImage.outline_segments
# from v1.7, after which it was deprecated without a functionally
# identical replacement.
def outline_segments(self, mask_background=False):
    """
    Outline the labeled segments.

    The "outlines" represent the pixels *just inside* the segments,
    leaving the background pixels unmodified.

    Parameters
    ----------
    mask_background : bool, optional
        Set to `True` to mask the background pixels (labels = 0) in
        the returned array.  This is useful for overplotting the
        segment outlines.  The default is `False`.

    Returns
    -------
    boundaries : `~numpy.ndarray` or `~numpy.ma.MaskedArray`
        An array with the same shape of the segmentation array
        containing only the outlines of the labeled segments.  The
        pixel values in the outlines correspond to the labels in the
        segmentation array.  If ``mask_background`` is `True`, then
        a `~numpy.ma.MaskedArray` is returned.
    """
    from scipy.ndimage import (generate_binary_structure, grey_dilation,
                                grey_erosion)

    # edge connectivity
    footprint = generate_binary_structure(self._ndim, 1)

    # mode='constant' ensures outline is included on the array borders
    eroded = grey_erosion(self.data, footprint=footprint, mode='constant',
                            cval=0.0)
    dilated = grey_dilation(self.data, footprint=footprint,
                            mode='constant', cval=0.0)

    outlines = ((dilated != eroded) & (self.data != 0)).astype(int)
    outlines *= self.data

    if mask_background:
        outlines = np.ma.masked_where(outlines == 0, outlines)

    return outlines

def image_segmentation(cnt_image: np.ndarray, band: str, f_e_mask, exposure_time):

    from photutils.segmentation import detect_sources
    from photutils.segmentation import make_2dgaussian_kernel
    from photutils.segmentation import SourceCatalog
    from photutils.segmentation import deblend_sources
    import sys
    import gc

    print("Estimating background and threshold.")
    cnt_image, threshold = estimate_background_and_threshold(
        cnt_image, f_e_mask, band, exposure_time
    )
    kernel = make_2dgaussian_kernel(fwhm=3, size=(3, 3))
    convolved_data = convolve(cnt_image, kernel)
    # changing "npixels" in detect sources to 3 ID's more small sources
    # but also more spurious looking ones..
    print("Segmenting and deblending point sources.")
    segment_map = detect_sources(
        convolved_data, threshold, npixels=4, mask=f_e_mask
    )
    del threshold, kernel
    gc.collect()

    print("Deblending sources.")
    deblended_segment_map = deblend_sources(convolved_data,
                                            segment_map,
                                            npixels=8,
                                            nlevels=20,
                                            contrast=0.004,
                                            mode='linear',
                                            progress_bar=False)

    del segment_map
    gc.collect()

    #outline_seg_map = deblended_segment_map.outline_segments()
    # refactor to make it a funcion instead of a method; see function def above
    outline_seg_map = outline_segments(deblended_segment_map)

    # can add more columns w/ outputs listed in photutils image seg documentation
    columns = ['label', 'xcentroid', 'ycentroid', 'area', 'segment_flux',
               'elongation', 'eccentricity', 'equivalent_radius', 'orientation',
               'max_value', 'maxval_xindex', 'maxval_yindex', 'min_value',
               'minval_xindex', 'minval_yindex', 'bbox_xmin', 'bbox_xmax',
               'bbox_ymin', 'bbox_ymax']

    seg_sources = SourceCatalog(cnt_image, deblended_segment_map, convolved_data=convolved_data) \
        .to_table(columns=columns).to_pandas()
    seg_sources.astype({'label': 'int32'})
    seg_sources = seg_sources.set_index("label", drop=True).dropna(axis=0, how='any')

    return deblended_segment_map.data, outline_seg_map, seg_sources


def estimate_threshold(bkg_rms, band, exposure_time):
    print("Calculating source threshold.")
    if band == "NUV" and exposure_time > 800:
        threshold = np.multiply(1.5, bkg_rms)
    elif band == "NUV" and exposure_time <= 800:
        threshold = np.multiply(3, bkg_rms)
    else:
        threshold = np.multiply(3, bkg_rms)

    return threshold


def estimate_background_and_threshold(cnt_image: np.ndarray, f_e_mask, band, exposure_time):
    cnt_image, bkg_rms = estimate_background(cnt_image, f_e_mask)
    threshold = estimate_threshold(bkg_rms, band, exposure_time)
    return cnt_image, threshold

def estimate_background(cnt_image, f_e_mask):
    from photutils.background import Background2D, MedianBackground
    from astropy.stats import SigmaClip

    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg = Background2D(cnt_image,
                       (50, 50),
                       filter_size=(3, 3),
                       bkg_estimator=bkg_estimator,
                       sigma_clip=sigma_clip,
                       mask=f_e_mask)
    cnt_image -= bkg.background
    del bkg.background
    del bkg._unfiltered_background_mesh
    del bkg.background_mesh
    del bkg._bkg_stats
    rms = bkg.background_rms
    return cnt_image, rms.astype(np.float32)


def mask_for_extended_sources(cnt_image: np.ndarray, band: str):
    print("Running DAO for extended source ID.")
    dao_sources = dao_handler(cnt_image)
    print(f"Found {len(dao_sources)} peaks with DAO.")
    masks, extended_source_cat = get_extended(dao_sources, cnt_image.shape, band)
    return masks, extended_source_cat


def check_point_in_extended(outline_seg_map, masks, seg_sources):
    """
    Checks if the borders of any segments are inside of
    each extended source, which are paths stored in the
    dictionary "masks". Adds the ID of overlapping extended
    source to the DF holding pt sources (seg_sources).
    """
    seg_outlines = np.nonzero(outline_seg_map)
    seg_outlines_vert = np.vstack((seg_outlines[0], seg_outlines[1])).T
    seg_sources["extended_source"] = 0 # added to fix key error?
    for key in masks:
        inside_extended = masks[key].contains_points(seg_outlines_vert)
        segments_in_extended = outline_seg_map[seg_outlines][inside_extended]
        seg_sources.loc[segments_in_extended, "extended_source"] = int(key)
    seg_sources.to_csv("seg_sources_in_extented.csv")
    return seg_sources

def dao_handler(cnt_image: np.ndarray):
    """Combining two different DAO runs to get more "sources"."""
    dao_sources1 = dao_finder(cnt_image, fwhm=5, threshold=0.01)
    #dao_sources1 = dao_finder_1(cnt_image, None)
    dao_sources2 = dao_finder(cnt_image, fwhm=3, threshold=0.02)
    #dao_sources2 = dao_finder_2(cnt_image, None)
    dao_sources = pd.concat([dao_sources1, dao_sources2])
    return dao_sources

def dao_finder(cnt_image:np.ndarray,# exposure_time:float,
               fwhm:float=5, threshold:float=0.01,
               sigma_radius:float=1.5, ratio:float=1, theta:float=0):
    """
    DAO Star Finder Arguments:
       fwhm = full width half maximum of gaussian kernel
       threshold = pixel value below which not to select a source
       ratio = how round kernel is (1 = circle)
       theta = angle of kernel wrt to x-axis in degrees
       This uses a modified version of DAOStarFinder that returns
       less information called LocalHIGHSlocal.
    """
    print(threshold,fwhm,sigma_radius,ratio,theta)
    daofind = LocalHIGHSlocal(
        fwhm=fwhm, sigma_radius=sigma_radius,
        threshold=threshold, ratio=ratio, theta=theta
    )
    dao_sources = daofind.find_peaks(cnt_image)
    return dao_sources

def dao_finder_1(cnt_image: np.ndarray, exposure_time):
    threshold = 0.01
    daofind = LocalHIGHSlocal(
        fwhm=5, sigma_radius=1.5, threshold=threshold, ratio=1, theta=0
    )
    dao_sources = daofind.find_peaks(cnt_image)
    return dao_sources

def dao_finder_2(cnt_image: np.ndarray, exposure_time):
    threshold = 0.02
    daofind2 = LocalHIGHSlocal(
        fwhm=3, sigma_radius=1.5, threshold=threshold, ratio=1, theta=0
    )
    dao_sources2 = daofind2.find_peaks(cnt_image)
    return dao_sources2


def get_extended(dao_sources: pd.DataFrame, image_size, band: str):
    """
    DBSCAN groups input local maximum locations from DAOStarFinder according to
    a max. separation distance called "epsilon" to be considered in the same group.
    extended sources are considered dense collections of many local maximums.
    """
    from photutils.psf.groupstars import DBSCANGroup
    positions = np.transpose((dao_sources['x_peak'], dao_sources['y_peak']))

    starlist = Table()

    x_0 = list(zip(*positions))[0]
    y_0 = list(zip(*positions))[1]

    starlist['x_0'] = x_0
    starlist['y_0'] = y_0
    print(len(starlist))
    epsilon = 40 if band == "NUV" else 50 # nuv was 40
    dbscan_group = DBSCANGroup(crit_separation=epsilon)
    dbsc_star_groups = dbscan_group(starlist)
    dbsc_star_groups = dbsc_star_groups.group_by('group_id')
    # combining hull shapes for all extended sources to make a single
    # hull "mask" that shows extent of extended sources. pixel value of
    # each hull is the ID for that extended source.
    extended_source_cat = pd.DataFrame(columns=["id", "hull_area", "num_dao_points", "hull_vertices"])
    masks = {}
    gID = 1
    # this way of adding hull masks needs work because sometimes convex hulls overlap
    for i, group in enumerate(dbsc_star_groups.groups):
        if len(group) > 100:
            path, extended_hull_data = get_hull_path(group, gID, image_size, 10)
            extended_source_cat = pd.concat([extended_source_cat, extended_hull_data])
            masks[gID] = path
            gID += 1

    return masks, extended_source_cat


def get_hull_path(group, groupID: int, imageSize: tuple, critSep):
    """
    calculates convex hull of pts in group and uses Path to make a mask of
    each convex hull, assigning a number to each hull as they are made
    """
    import matplotlib.path
    from scipy.spatial import ConvexHull

    ny, nx = imageSize  # imageSize is a tuple of width, height

    xypos = np.transpose([group['y_0'], group['x_0']]) # switched x and y
    hull = ConvexHull(xypos)
    hull_verts = tuple(zip(xypos[hull.vertices, 0], xypos[hull.vertices, 1]))

    hull_data_dict = {'id': groupID, 'hull_area': hull.area,
                      'num_dao_points': hull.npoints, 'hull_vertices': hull_verts}
    extended_hull_data = pd.DataFrame(data=hull_data_dict)

    # path takes data as: an array, masked array or sequence of pairs.
    poly_path = matplotlib.path.Path(hull_verts)
    return poly_path, extended_hull_data


def extract_frame(frame, apertures):
    if isinstance(frame, scipy.sparse.spmatrix):
        frame = frame.toarray()
    return aperture_photometry(frame, apertures)["aperture_sum"].data


def _extract_photometry_unthreaded(movie, apertures):
    photometry = {}
    for ix, frame in enumerate(movie):
        print_inline(f"extracting frame {ix}")
        photometry[ix] = extract_frame(frame, apertures)
    return photometry


# it's foolish to run this multithreaded unless you _are_ unpacking sparse
# matrices, but I won't stop you.
def _extract_photometry_threaded(movie, apertures, threads):
    pool = Pool(threads)
    photometry = {}
    for ix, frame in enumerate(movie):
        photometry[ix] = pool.apply_async(extract_frame, (frame, apertures))
    pool.close()
    pool.join()
    return {ix: result.get() for ix, result in photometry.items()}


def extract_photometry(movie_dict, source_table, apertures, threads):
    photometry_tables = []
    for key in ["cnt", "flag", "edge"]:
        title = "primary movie" if key == "cnt" else f"{key} map"
        print(f"extracting photometry from {title}")
        if threads is None:
            photometry = _extract_photometry_unthreaded(
                movie_dict[key], apertures
            )
        else:
            photometry = _extract_photometry_threaded(
                movie_dict[key], apertures, threads
            )
        frame_indices = sorted(photometry.keys())
        if key in ("edge", "flag"):
            column_prefix = f"aperture_sum_{key}"
        else:
            column_prefix = "aperture_sum"
        photometry = {
            f"{column_prefix}_{ix}": photometry[ix] for ix in frame_indices
        }
        photom = pd.DataFrame.from_dict(photometry)
        photom.index = np.arange(1, len(photom) + 1)
        photometry_tables.append(photom)
    return pd.concat([source_table, *photometry_tables], axis=1)


def write_exptime_file(expfile: Pathlike, movie_dict) -> None:
    exptime_table = pd.DataFrame(
        {
            "expt": movie_dict["exptimes"],
            "t0": [trange[0] for trange in movie_dict["tranges"]],
            "t1": [trange[1] for trange in movie_dict["tranges"]],
        }
    )
    print(f"writing exposure time table to {expfile}")
    # noinspection PyTypeChecker
    exptime_table.to_csv(expfile, index=False)



def _load_csv_catalog(
    source_catalog_file: Pathlike, eclipse: int
) -> pd.DataFrame:
    sources = pd.read_csv(source_catalog_file)
    return sources.loc[sources["eclipse"] == eclipse]


def _load_parquet_catalog(
    source_catalog_file: Pathlike, eclipse: int
) -> pd.DataFrame:
    from pyarrow import parquet

    return parquet.read_table(
        source_catalog_file,
        filters=[('eclipse', '=', eclipse)],
        columns=['ra', 'dec']
    ).to_pandas()


def load_source_catalog(
    source_catalog_file: Pathlike, eclipse: int
) -> pd.DataFrame:
    source_catalog_file = Path(source_catalog_file)
    if source_catalog_file.suffix == ".csv":
        format_ = "csv"
    elif source_catalog_file.suffix == ".parquet":
        format_ = "parquet"
    else:
        raise ValueError(
            "Couldn't automatically determine source catalog format from the "
            "extension {source_catalog_file.suffix}. Please pass a .csv or "
            ".parquet file with at least the columns 'eclipse', 'ra', 'dec'."
        )
    try:
        if format_ == ".csv":
            sources = _load_csv_catalog(source_catalog_file, eclipse)
        else:
            sources = _load_parquet_catalog(source_catalog_file, eclipse)
        sources = sources[['ra', 'dec']]
    except KeyError:
        raise ValueError(
            "The source catalog file must specify source positions in "
            "columns named 'ra' and 'dec' with a reference column named "
            "'eclipse'."
        )
    return sources[~sources.duplicated()].reset_index(drop=True)


