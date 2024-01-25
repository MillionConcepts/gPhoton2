import numba as nb
import numpy as np

import gPhoton.constants as c


def interpolate_aspect_solutions(
    aspect_slice, asptime, t, eta_vec, xi_vec, ok_indices
):
    dxi, deta = np.zeros(len(t)), np.zeros(len(t))
    aspect_ratio = t[ok_indices] - asptime[aspect_slice] / (
        asptime[aspect_slice + 1] - asptime[aspect_slice]
    )
    dxi[ok_indices] = (
        xi_vec[aspect_slice + 1] - xi_vec[aspect_slice]
    ) * aspect_ratio
    deta[ok_indices] = (
        eta_vec[aspect_slice + 1] - eta_vec[aspect_slice]
    ) * aspect_ratio
    return deta, dxi


def find_null_indices(
    aspflags: np.ndarray,
    aspect_slice: np.ndarray,
    asptime: np.ndarray,
    flags: np.ndarray,
    ok_indices: np.ndarray,
) -> (np.ndarray, np.ndarray):
    flag_slice = flags[ok_indices]
    cut = (
        ((asptime[aspect_slice + 1] - asptime[aspect_slice]) == 1)
        & (aspflags[aspect_slice] % 2 == 0)
        & (aspflags[aspect_slice + 1] % 2 == 0)
        & (aspflags[aspect_slice - 1] % 2 == 0)
        & (flag_slice == 0)
        & (flag_slice != 7)
    )
    flags[ok_indices[~cut]] = 12
    off_detector_flags = [2, 5, 7, 8, 9, 10, 11, 12]
    null_ix = flags == 2
    for off_detector_flag in off_detector_flags[1:]:
        null_ix = null_ix | (flags == off_detector_flag)

    # null_ix = np.isin(flags, off_detector_flags)
    return null_ix, flags


def unfancy_detector_coordinates(
    band, dx, dy, flags, xp_as, xshift, yp_as, yshift
):
    flip = {"NUV": 1, "FUV": -1}[band]
    # TODO: is xi always 0? so can half of this be removed?
    #  probably? go look at the C codeunf
    # The detectors aren't oriented the same way.
    y_component = (yp_as + dy + yshift) * flip * 10
    x_component = (xp_as + dx + xshift) * flip * 10
    xi = nb.f4(c.XI_XSC) * y_component + nb.f4(c.XI_YSC) * x_component
    # TODO, similarly with eta_ysc?
    eta = nb.f4(c.ETA_XSC) * y_component + nb.f4(c.ETA_YSC) * x_component
    col, row = xi_eta_to_col_row(xi, eta)
    cut = (
        (col > 0)
        & (col < 799)
        & (row > 0)
        & (row < 799)
        & (flags == 0)
    )
    flags[~cut] = 6
    return xi, eta, col, row, flags


def xi_eta_to_col_row(xi, eta):
    half_det = nb.f4(c.DETSIZE / 2)
    fill = nb.f4(c.FILL_VALUE)
    half_pix = nb.f4(c.PIXELS_PER_AXIS / 2)
    col = ((xi / 36000) / half_det * fill + 1) * half_pix
    row = ((eta / 36000) / half_det * fill + 1) * half_pix
    return col, row


def make_corners(floor_x, floor_y, fptrx, fptry, ok_indices):
    blt = (fptrx - floor_x)[ok_indices]
    blu = (fptry - floor_y)[ok_indices]
    inv_blt = 1 - blt
    inv_blu = 1 - blu
    corners = (
        inv_blt * inv_blu,
        blt * inv_blu,
        inv_blt * blu,
        blt * blu,
    )
    return corners


def sum_corners(cal_data, cal_ix0, cal_ix1, cal_ix2, cal_ix3, corners):
    wr = cal_data.ravel()
    return (
        corners[0] * wr[cal_ix0]
        + corners[1] * wr[cal_ix1]
        + corners[2] * wr[cal_ix2]
        + corners[3] * wr[cal_ix3]
    )


def or_reduce_minus_999(walk, walk_ix, x):
    if x is None:
        return walk[walk_ix] != -999
    return x | (walk[walk_ix] != -999)


def init_wiggle_arrays(floor_x, floor_y, fptrx, fptry, ix, xa, ya):
    wigx, wigy = np.zeros_like(fptrx), np.zeros_like(fptrx)
    blt = (fptrx - floor_x)[ix]
    blu = (fptry - floor_y)[ix]
    floor_x = floor_x[ix]
    floor_y = floor_y[ix]
    xa_ix = xa[ix]
    ya_ix = ya[ix]
    return blt, blu, floor_x, floor_y, wigx, wigy, xa_ix, ya_ix


def unfancy_distortion_component(
    cube_x0,
    cube_dx,
    cube_y0,
    cube_dy,
    cube_d0,
    cube_dd,
    cube_nc,
    cube_nr,
    flags,
    ok_indices,
    stim_coefficients,
    t,
    xp_as,
    yp_as,
):
    ss = stim_coefficients[0] + (t * stim_coefficients[1])  # stim separation
    col, row, depth = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    col[ok_indices] = (xp_as[ok_indices] - cube_x0) / cube_dx
    row[ok_indices] = (yp_as[ok_indices] - cube_y0) / cube_dy
    depth[ok_indices] = (ss[ok_indices] - cube_d0) / cube_dd
    # TODO: does this still happen?
    # [Future]: This throws an error sometimes like the following, may need
    # fixing...
    """PhotonPipe.py:262: RuntimeWarning: invalid value encountered in
                    less depth[((depth < 0)).nonzero()[0]] = 0.
                    PhotonPipe.py:263: RuntimeWarning: invalid value 
                    encountered in
                    greater_equal depth[((depth >= cube_nd)).nonzero()[0]] = 
                    -1.
                    ERROR: IndexError: index -9223372036854775808 is out of 
                    bounds for
                    axis 0 with size 17 [PhotonPipe]
                    depth[((depth < 0)).nonzero()[0]] = 0.
                    depth[((depth >= cube_nd)).nonzero()[0]] = -1.
                    """
    cut = (
        (col > -1)
        & (col < cube_nc)
        & (row > -1)
        & (row < cube_nr)
        & (flags == 0)
        & (np.floor(depth) <= 16)
    )
    flags[~cut] = 11
    ok_indices = np.nonzero(cut)[0]
    xshift, yshift = np.zeros(len(t)), np.zeros(len(t))
    return col, depth, ok_indices, row, xshift, yshift


# If you don't use numba, this function is horribly slow. Use this instead:
# def float_between_wiggled_points(blt_u, floor_xy, wig_xy, xya_ix):
#     return wig_xy[xya_ix, floor_xy] * (1 - blt_u) \
#            + wig_xy[xya_ix, floor_xy + 1] * blt_u
def float_between_wiggled_points(blt_u, floor_xy, wig_xy, xya_ix):
    wigix = list()
    wigix1 = list()
    for npix in range(len(xya_ix)):
        wigix.append(wig_xy[xya_ix[npix], floor_xy[npix]])
        wigix1.append(wig_xy[xya_ix[npix], floor_xy[npix] + 1])
    return np.array(wigix) * (1 - blt_u) + np.array(wigix1) * blt_u


nb.jit_module(nopython=True, cache=True)
