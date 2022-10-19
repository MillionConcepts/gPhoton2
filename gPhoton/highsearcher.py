# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides tools for finding local peaks in an astronomical
image.
"""
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
