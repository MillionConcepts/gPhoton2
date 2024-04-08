import warnings

import numpy as np
import pandas as pd


def find_peaks(data, threshold, box_size=3, footprint=None):
    """
    Find local peaks in an image that are above above a specified
    threshold value.

    Peaks are the maxima above the ``threshold`` within a local region.
    The local regions are defined by either the ``box_size`` or
    ``footprint`` parameters.  ``box_size`` defines the local region
    around each pixel as a square box.  ``footprint`` is a boolean array
    where `True` values specify the region shape.

    If multiple pixels within a local region have identical intensities,
    then the coordinates of all such pixels are returned.  Otherwise,
    there will be only one peak pixel per local region.  Thus, the
    defined region effectively imposes a minimum separation between
    peaks unless there are identical peaks within the region.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    threshold : float or array-like
        The data value to be used for the
        detection threshold.

    box_size : scalar or tuple, optional
        The size of the local region to search for peaks at every point
        in ``data``.  If ``box_size`` is a scalar, then the region shape
        will be ``(box_size, box_size)``.  Either ``box_size`` or
        ``footprint`` must be defined.  If they are both defined, then
        ``footprint`` overrides ``box_size``.

    footprint : `~numpy.ndarray` of bools, optional
        A boolean array where `True` values describe the local footprint
        region within which to search for peaks at every point in
        ``data``.  ``box_size=(n, m)`` is equivalent to
        ``footprint=np.ones((n, m))``.  Either ``box_size`` or
        ``footprint`` must be defined.  If they are both defined, then
        ``footprint`` overrides ``box_size``.

    Returns
    -------
    output : `~pd.DataFrame` or `None`
        A DataFrame containing the x and y pixel location of the peaks and
        their values. If no peaks are found then `None` is returned.
    """
    from scipy.ndimage import maximum_filter

    data = np.asanyarray(data)
    if footprint is not None:
        size_kwarg = {"footprint": footprint}
    else:
        size_kwarg = {"size": box_size}
    peak_goodmask = np.logical_and(
        data == maximum_filter(data, mode="constant", cval=0.0, **size_kwarg),
        data > threshold,
    )
    y_peaks, x_peaks = peak_goodmask.nonzero()
    if len(x_peaks) == 0:
        warnings.warn("No local peaks were found.")
        return None
    return pd.DataFrame(
        np.array([x_peaks, y_peaks, data[y_peaks, x_peaks]]).T,
        columns=["x_peak", "y_peak", "peak_value"],
    )


def _filter_data(data, kernel, mode="constant", fill_value=0.0):
    """
    Convolve a 2D image with a 2D kernel.

    The kernel may either be a 2D `~numpy.ndarray` or a
    `~astropy.convolution.Kernel2D` object.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.

    kernel : array-like (2D)
        The 2D kernel used to filter the input ``data``. Filtering the
        ``data`` will smooth the noise and maximize detectability of
        objects with a shape similar to the kernel.

    mode : {'constant', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
        The ``mode`` determines how the array borders are handled.  For
        the ``'constant'`` mode, values outside the array borders are
        set to ``fill_value``.  The default is ``'constant'``.

    fill_value : scalar, optional
        Value to fill data values beyond the array borders if ``mode``
        is ``'constant'``.  The default is ``0.0``.


    Returns
    -------
    result : `~numpy.ndarray`
        The convolved image.
    """
    if kernel is None:
        return data

    from scipy import ndimage

    # NOTE: if data is int and kernel is float, ndimage.convolve
    # will return an int image. If the data dtype is int, we make the
    # data float so that a float image is always returned
    # TODO: should they not be doing this?
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(float)
    result = ndimage.convolve(data, kernel, mode=mode, cval=fill_value)
    return result


# noinspection PyProtectedMember
class LocalHIGHSlocal:
    """ Find local peaks in an image using a convolution kernel. """
    def __init__(
        self,
        threshold,
        fwhm,
        ratio=1.0,
        theta=0.0,
        sigma_radius=1.5,
        box_size=None,
        use_box_size=False,
    ):
        from photutils.detection.core import _StarFinderKernel

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
        peak_find_kwargs = {"data": convolved, "threshold": self.threshold}
        if self.use_box_size is True:
            if self.box_size is None:
                raise ValueError(
                    "Must define a box size to use a box instead of a "
                    "kernel footprint."
                )
            peak_find_kwargs["box_size"] = self.box_size
        else:
            peak_find_kwargs["footprint"] = self.footprint

        self.peaks = find_peaks(**peak_find_kwargs)
        return self.peaks

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

    from photutils.segmentation import (detect_sources, make_2dgaussian_kernel,
                                        SourceCatalog, deblend_sources)
    import sys
    from scipy.ndimage import convolve

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
    #gc.collect()

    deblended_segment_map = deblend_sources(convolved_data,
                                            segment_map,
                                            npixels=8,
                                            nlevels=20,
                                            contrast=0.004,
                                            mode='linear',
                                            progress_bar=False)

    del segment_map
    
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


def mask_for_extended_sources(cnt_image: np.ndarray, band: str, exposure_time):
    print("Running DAO for extended source ID.")
    dao_sources = dao_handler(cnt_image, exposure_time)
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
    #seg_sources.to_csv("seg_sources_in_extented.csv") # for debug only
    return seg_sources


def dao_handler(cnt_image: np.ndarray, exposure_time):
    # run DAO twice to get more sources
    dao_sources1 = dao_finder(cnt_image, threshold=0.01, fwhm=5)
    dao_sources2 = dao_finder(cnt_image,threshold=0.02, fwhm=3)
    dao_sources = pd.concat([dao_sources1, dao_sources2])
    return dao_sources

def dao_finder(cnt_image: np.ndarray, threshold: float = 0.01,
               fwhm: float = 5, sigma_radius: float = 1.5, ratio: float = 1,
               theta: float = 0):
    daofind = LocalHIGHSlocal(
        fwhm=fwhm, sigma_radius=sigma_radius,
        threshold=threshold, ratio=ratio, theta=theta
    )
    dao_sources = daofind.find_peaks(cnt_image)
    return dao_sources

def get_extended(dao_sources: pd.DataFrame, image_size, band: str):
    """
    DBSCAN groups input local maximum locations from DAOStarFinder according to
    a max. separation distance called "epsilon" to be considered in the same group.
    extended sources are considered dense collections of many local maximums.
    """
    from photutils.psf.groupstars import DBSCANGroup # replace w/ photutils.psf.SourceGrouper
    positions = np.transpose((dao_sources['x_peak'], dao_sources['y_peak']))

    from astropy.table import Table
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
