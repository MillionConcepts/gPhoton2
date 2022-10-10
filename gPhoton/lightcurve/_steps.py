"""
methods for generating lightcurves from FITS images/movies, especially those
produced by the gPhoton.moviemaker pipeline. They are principally intended
for use as components of the primary gPhoton.lightcurve pipeline, called as
part of the course of running gPhoton.lightcurve.core.make_lightcurves(), and
may not suitable for independent use.
"""

from multiprocessing import Pool
from typing import Union, Optional, Mapping
import warnings

import astropy.wcs
import numpy as np
import pandas as pd
import scipy.sparse

from multiprocessing import Pool
from pathlib import Path
from typing import Union, Optional, Mapping
import warnings

from photutils import CircularAperture, aperture_photometry

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
    positions = source_table[["xcentroid", "ycentroid"]].values

    apertures = CircularAperture(positions, r=aperture_size)
    print("Performing aperture photometry on primary image.")
    phot_table = aperture_photometry(image_dict["cnt"], apertures).to_pandas()
    phot_table = phot_table.set_index("id", drop=True)
    print("Performing aperture photometry on flag maps.")
    flag_table = aperture_photometry(image_dict["flag"], apertures).to_pandas()
    flag_table = flag_table.set_index("id", drop=True)
    print("Performing aperture photometry on edge maps.")
    edge_table = aperture_photometry(image_dict["edge"], apertures).to_pandas()
    edge_table = edge_table.set_index("id", drop=True)
    source_table = pd.concat(
        [source_table, phot_table[["xcenter", "ycenter", "aperture_sum"]]],
        axis=1)
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
        print("Extracting sources with DAOFIND.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # for some reason image segmentation still finds sources in the masked cnt image?
            # so we mask twice: once by zeroing and then again by making a bool mask and
            # feeding it to image segmentation

            masked_cnt_image = zero_flag_and_edge(
                image_dict["cnt"],
                image_dict["flag"],
                image_dict["edge"])

            flag_edge_mask = flag_and_edge_mask(
                image_dict["cnt"],
                image_dict["flag"],
                image_dict["edge"])

            source_table, segment_map, extended_source_mask, extended_source_cat = \
                get_point_and_extended_sources(masked_cnt_image / exptime, band, flag_edge_mask)
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
    return source_table, segment_map, extended_source_mask, extended_source_cat


def get_point_and_extended_sources(cnt_image: np.ndarray, band: str, f_e_mask):

    """
    Main function for extracting point and extended sources from an eclipse.
    Image segmentation and point source extraction occurs in this function.
    The threshold for being a source in NUV is set at 1.5 times the background
    rms values (2d array), while for FUV it is 3 times. The FUV setting isn't
    ~perfect~.
    Extended source extraction occurs in helper functions.
    """

    from photutils.segmentation import detect_sources
    from photutils.background import Background2D, MedianBackground
    from photutils.utils import make_random_cmap
    from photutils.segmentation import make_2dgaussian_kernel
    from photutils.segmentation import SourceCatalog
    from photutils.segmentation import deblend_sources

    bkg_estimator = MedianBackground()
    bkg = Background2D(cnt_image,
                       (50, 50),
                       filter_size=(3, 3),
                       bkg_estimator=bkg_estimator,
                       mask=f_e_mask)
    cnt_image -= bkg.background
    threshold = 1.5 * bkg.background_rms if band == "NUV" else .002
    kernel = make_2dgaussian_kernel(fwhm=3, size=(3, 3))
    convolved_data = convolve(cnt_image, kernel)
    # changing "npixels" in detect sources to 3 produces more small sources
    # but also more spurious looking ones..
    segment_map = detect_sources(convolved_data,
                                 threshold,
                                 npixels=4,
                                 mask=f_e_mask)
    deblended_segment_map = deblend_sources(convolved_data,
                                            segment_map,
                                            npixels=8,
                                            nlevels=20,
                                            contrast=0.004,
                                            mode='linear',
                                            progress_bar=False)

    # can add more columns w/ outputs listed in photutils image seg documentation
    columns = ['label', 'xcentroid', 'ycentroid', 'area', 'segment_flux',
               'elongation', 'eccentricity', 'equivalent_radius', 'orientation',
               'max_value', 'maxval_xindex', 'maxval_yindex', 'min_value',
               'minval_xindex', 'minval_yindex', 'bbox_xmin', 'bbox_xmax',
               'bbox_ymin', 'bbox_ymax']

    # cnt_image is background subtracted (could do not background subtracted? hopefully
    # less "spurious" DAO detections this way)
    extended_source_mask, extended_source_cat = mask_for_extended_sources(cnt_image, band)
    # checking for overlap between extended source mask and segmentation image
    mask_and_map_matches = list(np.stack((deblended_segment_map, extended_source_mask),
                                         axis=2).flatten())
    iterator = iter(mask_and_map_matches)
    # x & y of masked segments
    sources_in_extended = set(list(zip(iterator, iterator)))

    seg_sources = SourceCatalog(cnt_image, deblended_segment_map, convolved_data=convolved_data)\
        .to_table(columns=columns).to_pandas()
    seg_sources.astype({'label': 'int32'})
    seg_sources = seg_sources.set_index("label", drop=True).dropna(axis=0, how='any')

    for i in sources_in_extended:
        if i[0] != 0:
            seg_sources.loc[i[0], "extended_source"] = i[1]

    return seg_sources.dropna(), segment_map, extended_source_mask, extended_source_cat


def mask_for_extended_sources(cnt_image: np.ndarray, band: str):
    dao_sources = dao_finder_modified(cnt_image)
    extended_mask, extended_source_cat = get_extended_mask(dao_sources, cnt_image.shape, band)
    return extended_mask, extended_source_cat


def dao_finder_modified(cnt_image: np.ndarray):
    """
    DAO Star Finder Arguments:
       fwhm = full width half maximum of gaussian kernel
       threshold = pixel value below which not to select a source
       ratio = how round kernel is (1 = circle)
       theta = angle of kernel wrt to x-axis in degrees
       Combining two different DAO runs to get more "sources".
       This uses a modified version of DAOStarFinder that returns
       less information called LocalHIGHSlocal.
    """
    daofind = LocalHIGHSlocal(fwhm=5,
                              sigma_radius=1.5,
                              threshold=0.008,
                              ratio=1,
                              theta=0)
    daofind2 = LocalHIGHSlocal(fwhm=3,
                               sigma_radius=1.5,
                               threshold=0.02,
                               ratio=1,
                               theta=0)
    dao_sources = daofind.find_peaks(cnt_image)
    dao_sources2 = daofind2.find_peaks(cnt_image)
    dao_sources = pd.concat([dao_sources, dao_sources2])
    print(f"# of DAO sources found: {len(dao_sources)}")

    return dao_sources


def get_extended_mask(dao_sources: pd.DataFrame, image_size, band: str):
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

    epsilon = 28 if band == "NUV" else 45
    dbscan_group = DBSCANGroup(crit_separation=epsilon)
    dbsc_star_groups = dbscan_group(starlist)
    dbsc_star_groups = dbsc_star_groups.group_by('group_id')

    # combining hull shapes for all extended sources to make a single
    # hull "mask" that shows extent of extended sources. pixel value of
    # each hull is the ID for that extended source.
    extended_source_cat = pd.DataFrame(columns=["id", "hull_area", "num_dao_points", "hull_vertices"])
    mask = np.full(image_size, 0)
    gID = 1
    for i, group in enumerate(dbsc_star_groups.groups):
        if len(group) > 65:
            newMask, extended_hull_data = get_hull_mask(group, gID, image_size, 10)
            extended_source_cat = pd.concat([extended_source_cat, extended_hull_data])
            mask = np.add(mask, newMask)
            gID += 1

    return mask, extended_source_cat


def get_hull_mask(group, groupID: int, imageSize: tuple, critSep):
    """
    calculates convex hull of pts in group and uses Path to make a mask of
    each convex hull, assigning a number to each hull as they are made
    """
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull

    ny, nx = imageSize  # imageSize is a tuple of width, height

    xypos = np.transpose([group['x_0'], group['y_0']])
    hull = ConvexHull(xypos)
    hull_verts = tuple(zip(xypos[hull.vertices, 0], xypos[hull.vertices, 1]))

    hull_data_dict = {'id': groupID, 'hull_area': hull.area,
                      'num_dao_points': hull.npoints, 'hull_vertices': hull_verts}
    extended_hull_data = pd.DataFrame(data=hull_data_dict)

    # path takes data as: an array, masked array or sequence of pairs.
    poly_path = Path(hull_verts)
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    # the buffer area / nature of convex hull means sometimes extended objects overlap, so the methodology
    # of combining them may have to change
    hull_mask = poly_path.contains_points(points)  #, radius=critSep * 1.5) radius adds a small buffer area to the mask
    hull_mask = hull_mask.reshape((ny, nx))
    hull_mask = hull_mask.astype(int) * groupID
    return hull_mask, extended_hull_data


def make_source_figs(source_table: pd.DataFrame,
                     segment_map: np.ndarray,
                     extended_source_mask: np.ndarray,
                     cnt_image: np.ndarray,
                     eclipse,
                     band: str,
                     outpath=".",
                     name="cnt"):
    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = 300
    mpl.use("agg")
    # segmentation map
    fig = fig_plot(
        segment_map,
        f"e{eclipse}_{band}_segmented",
    )
    fig.savefig(
        Path(outpath, f"e{eclipse}-{band[0].lower()}d-segmentation.jpg")
    )
    # extended source map
    fig = fig_plot(
        extended_source_mask,
        f"e{eclipse}_{band}_extended_mask",
    )
    fig.savefig(
        Path(outpath, f"e{eclipse}-{band[0].lower()}d-extended-mask.jpg")
    )
    # sources plotted on eclipse as circles
    fig = fig_plot_sources(
        cnt_image,
        source_table,
        f"e{eclipse}_{band}_extended_mask",
    )
    fig.savefig(
        Path(outpath, f"e{eclipse}-{band[0].lower()}d-sources-on-image.jpg")
    )


def fig_plot(array, name):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4, 4), dpi=250)
        cmap = plt.get_cmap('viridis', np.max(array) - np.min(array) + 1)
        cmap.set_under(color='white')
        plt.imshow(array, cmap=cmap, vmin=0.5)
        plt.title(name)
        return fig


def fig_plot_sources(array, sources, name):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6, 6), dpi=350)
    plt.imshow(centile_clip(array), cmap='viridis', interpolation='none')
    plt.title(name)
    for index, point in sources.dropna().iterrows():
        xypos = np.transpose([point['xcentroid'], point['ycentroid']])
        if point["extended_source"] == 0:
            c = "white"
        else:
            c = "red"
        method_r = point['equivalent_radius']
        ap = CircularAperture(xypos, r=method_r)
        ap.plot(color=c, lw=.75)
    return fig


def centile_clip(image, centiles=(0, 90)):
    finite = np.ma.masked_invalid(image)
    bounds = np.percentile(finite[~finite.mask].data, centiles)
    result = np.ma.clip(finite, *bounds)
    if isinstance(image, np.ma.MaskedArray):
        return result
    return result.data


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


from photutils.detection.core import find_peaks, _StarFinderKernel
from photutils.utils._convolution import _filter_data

class LocalHIGHSlocal:
    def __init__(
        self,
        threshold,
        fwhm,
        ratio=1.0,
        theta=0.0,
        sigma_radius=1.5,
        box_size=None,
        use_box_size=False
    ):
        kernel = _StarFinderKernel(fwhm, ratio, theta, sigma_radius)
        self.kernel = kernel.data
        self.footprint = kernel.mask.astype(bool)
        self.peaks = None
        self.convolved = None
        self.box_size = box_size
        self.use_box_size = use_box_size
        self.threshold = threshold
        self.fwhm = fwhm
        self.ratio = ratio
        self.theta = theta
        self.sigma_radius = sigma_radius

    def convolve_image(self, image):
        self.convolved = _filter_data(image, self.kernel)
        return self.convolved

    def find_peaks(self, image):
        convolved = self.convolve_image(image)
        peak_find_kwargs = {'data': convolved, 'threshold': self.threshold}
        if self.use_box_size is True:
            if self.box_size is None:
                raise ValueError(
                    "Must define a box size to use a box instead of a "
                    "kernel footprint."
                )
            peak_find_kwargs['box_size'] = self.box_size
        else:
            peak_find_kwargs['footprint'] = self.footprint

        self.peaks = find_peaks(**peak_find_kwargs).to_pandas()
        return self.peaks