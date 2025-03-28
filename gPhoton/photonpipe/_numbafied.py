"""
Computational primitives used by photonpipe._steps and accelerated
via the numba JIT.
"""

# Caution: Some of the functions in this file are written strangely
# in order to type-check correctly and get efficient code generation
# from numba at the same time.

import numba as nb
import numpy as np

import gPhoton.constants as c
from gPhoton.numba_utilz import jit
from gPhoton.types import GalexBand, NDArray, NFloat, NInt


@jit
def interpolate_aspect_solutions(
    aspect_slice: NDArray[NFloat],
    asptime: NDArray[NFloat],
    t: NDArray[NFloat],
    eta_vec: NDArray[NFloat],
    xi_vec: NDArray[NFloat],
    ok_indices: NDArray[np.intp],
) -> tuple[NDArray[NFloat], NDArray[NFloat]]:
    aspect_ratio = t[ok_indices] - asptime[aspect_slice] / (
        asptime[aspect_slice + 1] - asptime[aspect_slice]
    )
    dxi = np.zeros_like(t)
    dxi[ok_indices] = (
        xi_vec[aspect_slice + 1] - xi_vec[aspect_slice]
    ) * aspect_ratio
    deta = np.zeros_like(t)
    deta[ok_indices] = (
        eta_vec[aspect_slice + 1] - eta_vec[aspect_slice]
    ) * aspect_ratio
    return deta, dxi


@jit
def find_null_indices(
    aspflags: NDArray[NFloat],
    aspect_slice: NDArray[NFloat],
    asptime: NDArray[NFloat],
    flags: NDArray[NInt],
    ok_indices: NDArray[np.intp],
) -> tuple[NDArray[NFloat], NDArray[NInt]]:
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


@jit
def unfancy_detector_coordinates(
    band: GalexBand,
    dx: NDArray[NFloat],
    dy: NDArray[NFloat],
    flags: NDArray[NInt],
    xp_as: NDArray[NFloat],
    xshift: NDArray[NFloat],
    yp_as: NDArray[NFloat],
    yshift: NDArray[NFloat],
) -> tuple[
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NInt],
]:
    flip = {"NUV": 1, "FUV": -1}[band]
    # TODO: is xi always 0? so can half of this be removed?
    #  probably? go look at the C code
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


@jit
def xi_eta_to_col_row(
    xi: NDArray[NFloat],
    eta: NDArray[NFloat],
) -> tuple[NDArray[NFloat], NDArray[NFloat]]:
    half_det = nb.f4(c.DETSIZE / 2)
    fill = nb.f4(c.FILL_VALUE)
    half_pix = nb.f4(c.PIXELS_PER_AXIS / 2)
    col = ((xi / 36000) / half_det * fill + 1) * half_pix
    row = ((eta / 36000) / half_det * fill + 1) * half_pix
    return col, row


@jit
def make_corners(
    floor_x: NDArray[NFloat],
    floor_y: NDArray[NFloat],
    fptrx: NDArray[NFloat],
    fptry: NDArray[NFloat],
    ok_indices: NDArray[np.intp],
) -> tuple[
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
]:
    blt = (fptrx - floor_x)[ok_indices]
    blu = (fptry - floor_y)[ok_indices]
    inv_blt = 1 - blt
    inv_blu = 1 - blu
    return (
        inv_blt * inv_blu,
        blt * inv_blu,
        inv_blt * blu,
        blt * blu,
    )


@jit
def sum_corners(
    cal_data: NDArray[NFloat],
    cal_ix0: NDArray[np.intp],
    cal_ix1: NDArray[np.intp],
    cal_ix2: NDArray[np.intp],
    cal_ix3: NDArray[np.intp],
    corners: tuple[NDArray[NFloat], NDArray[NFloat],
                   NDArray[NFloat], NDArray[NFloat]],
) -> NDArray[NFloat]:
    wr = cal_data.ravel()
    rv = corners[0] * wr[cal_ix0]
    rv += corners[1] * wr[cal_ix1]
    rv += corners[2] * wr[cal_ix2]
    rv += corners[3] * wr[cal_ix3]
    return rv


@jit
def or_reduce_minus_999(
    walk: NDArray[NFloat],
    walk_ix: NDArray[np.intp],
    x: NDArray[np.bool_] | None
) -> NDArray[np.bool_]:
    y = np.not_equal(walk[walk_ix], -999)
    if x is not None:
        return y | x
    return y

@jit
def init_wiggle_arrays(
    floor_x: NDArray[NFloat],
    floor_y: NDArray[NFloat],
    fptrx: NDArray[NFloat],
    fptry: NDArray[NFloat],
    ix: NDArray[np.intp],
    xa: NDArray[NFloat],
    ya: NDArray[NFloat],
) -> tuple[
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
]:
    wigx = np.zeros_like(fptrx)
    wigy = np.zeros_like(fptry)
    blt = (fptrx - floor_x)[ix]
    blu = (fptry - floor_y)[ix]
    floor_x = floor_x[ix]
    floor_y = floor_y[ix]
    xa_ix = xa[ix]
    ya_ix = ya[ix]
    return blt, blu, floor_x, floor_y, wigx, wigy, xa_ix, ya_ix


@jit
def unfancy_distortion_component(
    cube_x0: NDArray[NFloat],
    cube_dx: NDArray[NFloat],
    cube_y0: NDArray[NFloat],
    cube_dy: NDArray[NFloat],
    cube_d0: NDArray[NFloat],
    cube_dd: NDArray[NFloat],
    cube_nc: NDArray[NFloat],
    cube_nr: NDArray[NFloat],
    flags: NDArray[NInt],
    ok_indices: NDArray[np.intp],
    stim_coefficients: NDArray[NFloat], # this one is Nx2
    t: NDArray[NFloat],
    xp_as: NDArray[NFloat],
    yp_as: NDArray[NFloat],
) -> tuple[
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[np.intp],
    NDArray[NFloat],
    NDArray[NFloat],
    NDArray[NFloat],
]:
    ss = stim_coefficients[0] + (t * stim_coefficients[1])  # stim separation
    col = np.zeros_like(t)
    col[ok_indices] = (xp_as[ok_indices] - cube_x0) / cube_dx

    row = np.zeros_like(t)
    row[ok_indices] = (yp_as[ok_indices] - cube_y0) / cube_dy

    depth = np.zeros_like(t)
    depth[ok_indices] = (ss[ok_indices] - cube_d0) / cube_dd

    # TODO: does this still happen?
    # [Future]: This throws an error sometimes like the following,
    # may need fixing...
    #     PhotonPipe.py:262: RuntimeWarning: invalid value encountered in
    #         less depth[((depth < 0)).nonzero()[0]] = 0.
    #     PhotonPipe.py:263: RuntimeWarning: invalid value encountered in
    #         greater_equal depth[((depth >= cube_nd)).nonzero()[0]] = -1.
    #     ERROR: IndexError: index -9223372036854775808 is out of bounds for
    #         axis 0 with size 17 [PhotonPipe]
    #     depth[((depth < 0)).nonzero()[0]] = 0.
    #     depth[((depth >= cube_nd)).nonzero()[0]] = -1.

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
    xshift = np.zeros_like(t)
    yshift = np.zeros_like(t)
    return col, depth, ok_indices, row, xshift, yshift


# If you don't use numba, this function is horribly slow. Use this instead:
# def float_between_wiggled_points(blt_u, floor_xy, wig_xy, xya_ix):
#     return wig_xy[xya_ix, floor_xy] * (1 - blt_u) \
#            + wig_xy[xya_ix, floor_xy + 1] * blt_u
@jit
def float_between_wiggled_points(
    blt_u: NDArray[NFloat],
    floor_xy: NDArray[NFloat],
    wig_xy: NDArray[NFloat],
    xya_ix: NDArray[np.intp]
) -> NDArray[NFloat]:
    wigix = []
    wigix1 = []
    for npix in range(len(xya_ix)):
        wigix.append(wig_xy[xya_ix[npix], floor_xy[npix]])
        wigix1.append(wig_xy[xya_ix[npix], floor_xy[npix] + 1])
    rv = np.array(wigix, dtype=np.float32) * (1 - blt_u)
    rv += np.array(wigix1, dtype=np.float32) * blt_u
    return rv
