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

        kernel = _StarFinderKernel(fwhm,
                                   ratio=ratio,
                                   theta=theta,
                                   sigma_radius=sigma_radius
                                   )
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
# identical replacement. The closest replacement,
# SegmentationImage.to_patches, is more memory-intensive and requires
# more steps.
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


def get_point_sources(cnt_image: np.ndarray, band: str, f_e_mask, exposure_time):
    """
    Uses image segmentation to identify point sources.
    The threshold for being a source in NUV is set at 1.5 times the background
    rms values (2d array) for eclipses over 800 sec, while for NUV under 800s and
    for all FUV it is 3 times bkg rms. Then there is a minimum threshold for FUV
    of the upper quartile of all threshold values over 0.0005.
    This was called image_segmentation historically.
    Returns an array with source outlines and a source catalog.
    """

    from photutils.segmentation import (detect_sources, make_2dgaussian_kernel,
                                        SourceCatalog, deblend_sources)
    from scipy.ndimage import convolve

    print("Estimating background and threshold.")
    cnt_image, threshold = estimate_background_and_threshold(
        cnt_image, band, exposure_time
    )
    kernel = make_2dgaussian_kernel(fwhm=3, size=(3, 3))
    convolved_data = convolve(cnt_image, kernel)
    # changing "npixels" in detect sources to 3 ID's more small sources
    # but also more spurious looking ones..
    print("Segmenting and deblending point sources.")
    segment_map = detect_sources(
        convolved_data,
        threshold,
        npixels=4,
        mask=f_e_mask
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

    seg_sources = SourceCatalog(cnt_image, deblended_segment_map, convolved_data=convolved_data
                                ).to_table(columns=columns).to_pandas()
    seg_sources.astype({'label': 'int32'})
    seg_sources = seg_sources.set_index("label", drop=True).dropna(axis=0, how='any')

    # for source finding troubleshooting purposes:
    # from astropy.io import fits
    # deblended_data = deblended_segment_map.data.astype(np.int32)
    # hdu = fits.PrimaryHDU(deblended_data)
    # hdul = fits.HDUList([hdu])
    # hdul.writeto(f'deblended_segmentation_{band}.fits', overwrite=True)

    return outline_seg_map, seg_sources


def estimate_threshold(bkg_rms, band, exposure_time):
    print("Calculating source threshold.")
    if band == "NUV" and exposure_time > 800:
        threshold = np.multiply(1.5, bkg_rms)
    elif band == "NUV" and exposure_time <= 800:
        print(f"low exposure time: {exposure_time} ")
        threshold = np.multiply(3, bkg_rms)
        filtered_thresh = threshold[threshold > 0.0005]
        if filtered_thresh is None or len(filtered_thresh) == 0:
            print("cnt array is sparse")
            print(f"exposure time: {exposure_time} ")
            threshold = np.full_like(threshold, .08)
        else:
            upper_quartile = np.percentile(filtered_thresh, 75)
            threshold[threshold < upper_quartile] = upper_quartile
    else:
        threshold = np.multiply(3, bkg_rms)
        # minimum threshold for FUV to avoid ID'ing background
        filtered_thresh = threshold[threshold > 0.0005]
        if filtered_thresh is None or len(filtered_thresh) == 0:
            print("cnt array is sparse")
            print(f"exposure time: {exposure_time} ")
            threshold = np.full_like(threshold, .04)
        else:
            upper_quartile = np.percentile(filtered_thresh, 75)
            threshold[threshold < upper_quartile] = upper_quartile
    # min threshold for both bands at low exposure times
    if exposure_time < 150:
        print("Applying low exp time min threshold")
        if band == "NUV":
            threshold[threshold < .08] = .08
        else:
            threshold[threshold < .04] = .04
    return threshold


def estimate_background_and_threshold(cnt_image: np.ndarray, band, exposure_time):

    cnt_image, bkg_rms = estimate_background(cnt_image, band)
    threshold = estimate_threshold(bkg_rms, band, exposure_time)

    return cnt_image, threshold


def estimate_background(cnt_image: np.ndarray, band):
    from photutils.background import Background2D, MedianBackground
    from astropy.stats import SigmaClip

    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    # remove f_e mask from background 2d, could consider
    # using the coverage_mask option for no data areas
    bkg = Background2D(cnt_image,
                       (50, 50),
                       filter_size=(3, 3),
                       bkg_estimator=bkg_estimator,
                       sigma_clip=sigma_clip)
    if band == "NUV":
        cnt_image -= bkg.background
    # no longer have to explicitly delete all bkg components bc of
    # photutils 2.0 update
    rms = bkg.background_rms

    return cnt_image, rms.astype(np.float32)


def mask_for_extended_sources(cnt_image: np.ndarray, band: str, exposure_time):
    print("Running DAO for extended source ID.")
    dao_sources = dao_handler(cnt_image, exposure_time)
    print(f"Found {len(dao_sources)} peaks with DAO.")
    masks, extended_source_cat = get_extended(dao_sources, band)
    return masks, extended_source_cat


def check_point_in_extended(outline_seg_map: np.ndarray, masks, source_table):
    """
    Checks if the borders of any segments are inside of
    each extended source, which are paths stored in the
    dictionary "masks". Adds the ID of overlapping extended
    source to the DF holding pt sources (seg_sources).
    0 = not in an extended source, but extended source
    detection was run.
    """
    seg_outlines = np.nonzero(outline_seg_map)
    seg_outlines_vert = np.vstack((seg_outlines[0], seg_outlines[1])).T
    source_table["extended_source"] = source_table["extended_source"].fillna(0)
    for key in masks:
        inside_extended = masks[key].contains_points(seg_outlines_vert)
        segments_in_extended = outline_seg_map[seg_outlines][inside_extended]
        source_table.loc[segments_in_extended, "extended_source"] = int(key)
    #seg_sources.to_csv("seg_sources_in_extented.csv") # for debug only
    return source_table


def dao_handler(cnt_image: np.ndarray, exposure_time):
    # runs DAO twice with diff kernel sizes to get more sources
    #TODO: experimenting with dao threshold being based on exp time again, then document
    # dao_1 was thresh = 0.01, dao_2 was thresh = 0.02
    threshold = np.multiply(np.power(exposure_time, -0.86026), 3)
    dao_sources1 = dao_finder(cnt_image, threshold=threshold, fwhm=5)
    dao_sources2 = dao_finder(cnt_image,threshold=threshold, fwhm=3)
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


def get_extended(dao_sources: pd.DataFrame, band: str):
    """
    DBSCAN groups input local maximum locations from DAOStarFinder according to
    a max. separation distance called "epsilon" to be considered in the same group.
    extended sources are considered dense collections of many local maximums.
    """
    from photutils.psf.groupers import SourceGrouper
    positions = np.transpose((dao_sources['x_peak'], dao_sources['y_peak']))

    from astropy.table import Table
    starlist = Table()

    x_0 = list(zip(*positions))[0]
    y_0 = list(zip(*positions))[1]
    starlist['x_0'] = x_0
    starlist['y_0'] = y_0

    epsilon = 40 if band == "NUV" else 50
    dbscan_group = SourceGrouper(min_separation=epsilon)
    # need to filter starlist to a max size to lessen memory usage
    max_stars = 20000
    num_stars = len(starlist)
    if num_stars > max_stars:
        indices_to_keep = np.linspace(0, num_stars - 1, max_stars, dtype=int)
        starlist = starlist[indices_to_keep]
        epsilon = 20
    dbscan_group = SourceGrouper(min_separation=epsilon)
    dbsc_star_groups = dbscan_group(starlist['x_0'],starlist['y_0'])
    dbsc_star_groups = pd.DataFrame({'group_id':dbsc_star_groups}).groupby('group_id')
    # combining hull shapes for all extended sources to make a single
    # hull "mask" that shows extent of extended sources. pixel value of
    # each hull is the ID for that extended source.
    masks = {}
    # todo: this way of adding hull masks needs work because sometimes convex hulls overlap
    extended_source_list = []
    gID = 1
    extended_source_cat = pd.DataFrame(columns=["id", "hull_area", "num_dao_points", "hull_vertices"])
    for i in dbsc_star_groups.groups.keys():
        group = dbsc_star_groups.groups[i]
        if len(group) > 100:
            # meant to avoid large stars that aren't really
            # "extended" sources with ~<100 dao sources
            path, extended_hull_data = get_hull_path(
                pd.DataFrame({'y_0': np.array(starlist['y_0'])[group],
                              'x_0': np.array(starlist['x_0'])[group]}),
                # group,
                gID
                )
            extended_source_list.append(extended_hull_data)
            masks[gID] = path
            gID += 1
    if len(extended_source_list)>0:
        extended_source_cat = pd.concat(extended_source_list, ignore_index=True)
        extended_source_cat.columns = ["id", "hull_area", "num_dao_points", "hull_vertices"]
    if gID>=128:
        print("There are too many extended sources identified, you can't label more than 128.")
        return
    return masks, extended_source_cat


def get_hull_path(group, group_id: int):
    """
    calculates convex hull of pts in group and uses Path to make a mask of
    each convex hull, assigning a number to each hull as they are made
    """
    import matplotlib.path
    from scipy.spatial import ConvexHull

    xypos = np.transpose([group['y_0'], group['x_0']]) # switched x and y
    hull = ConvexHull(xypos)
    hull_verts = tuple(zip(xypos[hull.vertices, 0], xypos[hull.vertices, 1]))

    hull_data_dict = {'id': group_id, 'hull_area': hull.area,
                      'num_dao_points': hull.npoints, 'hull_vertices': hull_verts}
    extended_hull_data = pd.DataFrame(data=hull_data_dict)

    # path takes data as: an array, masked array or sequence of pairs.
    poly_path = matplotlib.path.Path(hull_verts)
    return poly_path, extended_hull_data
