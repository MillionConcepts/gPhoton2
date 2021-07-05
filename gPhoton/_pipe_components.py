import os
from collections import defaultdict
from functools import reduce
from itertools import product
from operator import or_

import numpy as np
import pyarrow
from astropy.io import fits as pyfits

import gPhoton.constants as c
from gPhoton.CalUtils import (
    compute_stimstats,
    post_csp_caldata,
    rtaph_yac,
    rtaph_yac2,
    find_stims_index,
    clk_cen_scl_slp,
    avg_stimpos,
    rtaph_yap,
)
from gPhoton.FileUtils import load_aspect, web_query_aspect, download_data
from gPhoton.MCUtils import print_inline
from gPhoton.gnomonic import gnomfwd_simple, gnomrev_simple


class NestingDict(defaultdict):
    """
    shorthand for automatically-nesting dictionary -- i.e.,
    insert a series of keys at any depth into a NestingDict
    and it automatically creates all needed levels above.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_factory = NestingDict

    __repr__ = dict.__repr__


def retrieve_aspect_solution(aspfile, eclipse, retries, verbose):
    print_inline("Loading aspect data...")
    if aspfile:
        (aspra, aspdec, asptwist, asptime, aspheader, aspflags) = load_aspect(
            aspfile
        )
    else:
        (
            aspra,
            aspdec,
            asptwist,
            asptime,
            aspheader,
            aspflags,
        ) = web_query_aspect(eclipse, retries=retries)
    minasp, maxasp = min(asptime), max(asptime)
    trange = [minasp, maxasp]
    if verbose > 1:
        print("			trange= ( {t0} , {t1} )".format(t0=trange[0], t1=trange[1]))
    ra0, dec0, roll0 = aspheader["RA"], aspheader["DEC"], aspheader["ROLL"]
    if verbose > 1:
        print(
            "			[avgRA, avgDEC, avgROLL] = [{RA}, {DEC}, "
            "{ROLL}]".format(
                RA=aspra.mean(), DEC=aspdec.mean(), ROLL=asptwist.mean()
            )
        )
    # This projects the aspect solutions onto the MPS field centers.
    print_inline("Computing aspect vectors...")
    (xi_vec, eta_vec) = gnomfwd_simple(
        aspra, aspdec, ra0, dec0, -asptwist, 1.0 / 36000.0, 0.0
    )
    return aspdec, aspflags, aspra, asptime, asptwist, eta_vec, xi_vec


def process_chunk(
    aspect,
    band,
    cal_data,
    chunk,
    chunkid,
    disthead,
    dbscale,
    mask,
    maskfill,
    npixx,
    npixy,
    stim_coef0,
    stim_coef1,
    xoffset,
    yoffset,
):
    eta, flags, xi = apply_on_detector_corrections(
        band,
        cal_data,
        chunk,
        chunkid,
        disthead,
        mask,
        maskfill,
        npixx,
        npixy,
        stim_coef0,
        stim_coef1,
        xoffset,
        yoffset,
    )
    dec, ra, flags = apply_aspect_solution(
        aspect,
        chunkid,
        eta,
        flags,
        chunk["t"],
        xi,
    )

    return pyarrow.Table.from_arrays(
        arrays=[
            np.array(chunk["t"] * dbscale, dtype="int64"),
            chunk["x"],
            chunk["y"],
            chunk["xa"],
            chunk["ya"],
            chunk["q"],
            xi,
            eta,
            ra,
            dec,
            flags,
        ],
        names=[
            "t",
            "x",
            "y",
            "xa",
            "ya",
            "q",
            "xi",
            "eta",
            "ra",
            "dec",
            "flags",
        ],
    )


def apply_aspect_solution(
    aspect,
    chunkid,
    eta,
    flags,
    t,
    xi,
):
    aspdec, aspflags, aspra, asptime, asptwist, eta_vec, xi_vec = aspect
    # This gives the index of the aspect time that comes _before_
    # each photon time. Without the '-1' it will give the index
    # of the aspect time _after_ the photon time.
    print_inline(chunkid + "Mapping photon times to aspect times...")
    aspix = np.digitize(t, asptime) - 1
    print_inline(chunkid + "Applying dither correction...")
    # Use only photons that are bracketed by valid aspect solutions
    # and have been not themselves been flagged as invalid.
    cut = (
        (aspix > 0)
        & (aspix < (len(asptime) - 1))
        & ((flags == 0) | (flags == 6))
    )
    flags[~cut] = 7
    ok_indices = np.nonzero(cut)[0]
    aspect_slice = aspix[ok_indices]
    deta, dxi = interpolate_aspect_solutions(
        aspect_slice, asptime, chunkid, eta_vec, ok_indices, t, xi_vec
    )
    print_inline(chunkid + "Mapping to sky...")
    ra, dec = np.zeros(len(t)), np.zeros(len(t))
    ra[ok_indices], dec[ok_indices] = gnomrev_simple(
        xi[ok_indices] + dxi[ok_indices],
        eta[ok_indices] + deta[ok_indices],
        aspra[aspect_slice],
        aspdec[aspect_slice],
        -asptwist[aspect_slice],
        1 / 36000.0,
        0.0,
    )
    null_ix, flags = find_null_indices(
        aspflags, aspect_slice, asptime, flags, ok_indices
    )
    ra[null_ix] = np.nan
    dec[null_ix] = np.nan
    return dec, ra, flags


def interpolate_aspect_solutions(
    aspect_slice, asptime, chunkid, eta_vec, ok_indices, t, xi_vec
):
    print_inline(chunkid + "Interpolating aspect solutions...")
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


def find_null_indices(aspflags, aspect_slice, asptime, flags, ok_indices):
    flag_slice = flags[ok_indices]
    cut = (
        ((asptime[aspect_slice + 1] - asptime[aspect_slice]) == 1)
        & (aspflags[aspect_slice] % 2 == 0)
        & (aspflags[aspect_slice + 1] % 2 == 0)
        & (aspflags[aspect_slice - 1] % 2 == 0)
        & (flag_slice == 0)
        & (flag_slice != 7)
    )
    # TODO: the original version of this was:
    #  flags[np.where(cut == False)[0]] = 12. I'm pretty sure this is wrong,
    #  because it appears to index locations not considered by the above logic,
    #  and leads to different results at different chunk sizes.
    flags[ok_indices][~cut] = 12
    off_detector_flags = [2, 5, 7, 8, 9, 10, 11, 12]
    null_ix = np.isin(flags, off_detector_flags)
    return null_ix, flags


def apply_on_detector_corrections(
        band,
        cal_data,
        chunk,
        chunkid,
        disthead,
        mask,
        maskfill,
        npixx,
        npixy,
        stim_coef0,
        stim_coef1,
        xoffset,
        yoffset,
):
    q, t, x, xa, xb, y, ya, yamc, yb = (
        chunk["q"],
        chunk["t"],
        chunk["x"],
        chunk["xa"],
        chunk["xb"],
        chunk["y"],
        chunk["ya"],
        chunk["yamc"],
        chunk["yb"],
    )
    flags = np.zeros(len(t))
    fptrx, fptry = apply_wiggle_correction(chunkid, x, y)
    # This and other lines like it below are to verify that the
    # event is still on the detector.
    cut = (fptrx > 0.0) & (fptrx < 479.0) & (fptry > 0.0) & (fptry < 479.0)
    flags[~cut] = 8
    ok_indices = np.nonzero(cut)[0]
    fptrx, fptry, xdig, ydig = wiggle_and_dig(
        chunkid,
        fptrx,
        fptry,
        ok_indices,
        cal_data["wiggle"],
        x,
        xa,
        y,
        ya,
    )
    floor_x = np.array(fptrx, dtype="int64")
    floor_y = np.array(fptry, dtype="int64")
    flags, ok_indices = post_wiggle_update_indices_and_flags(
        flags, fptrx, fptry, floor_x, floor_y, q, cal_data["walk"]
    )
    fptrx, fptry, xp_as, yp_as = perform_spatial_nonlinearity_correction(
        chunkid,
        fptrx,
        fptry,
        floor_x,
        floor_y,
        ok_indices,
        q,
        cal_data["walk"],
        xdig,
        ydig,
    )
    cut = (
        (fptrx > 0.0)
        & (fptrx < 479.0)
        & (fptry > 0.0)
        & (fptry < 479.0)
        & (flags == 0)
    )
    flags[~cut] = 10
    ok_indices = np.nonzero(cut)[0]
    dx, dy = compute_detector_orientation(
        fptrx, fptry, cal_data["linearity"], ok_indices
    )
    xshift, yshift, flags, ok_indices = apply_stim_distortion_correction(
        chunkid,
        disthead,
        cal_data["distortion"],
        flags,
        ok_indices,
        stim_coef0,
        stim_coef1,
        t,
        xoffset,
        xp_as,
        yoffset,
        yp_as,
    )
    eta, xi, flags = apply_hotspot_mask(
        band,
        chunkid,
        dx,
        dy,
        flags,
        mask,
        maskfill,
        npixx,
        npixy,
        xp_as,
        xshift,
        yp_as,
        yshift,
    )
    return eta, flags, xi


def apply_hotspot_mask(
    band,
    chunkid,
    dx,
    dy,
    flags,
    mask,
    maskfill,
    npixx,
    npixy,
    xp_as,
    xshift,
    yp_as,
    yshift,
):
    print_inline(chunkid + "Applying hotspot mask...")
    # The detectors aren't oriented the same way.
    flip = 1.0
    if band == "FUV":
        flip = -1.0
    # TODO: is xi always 0? so can half of this be removed?
    xi = c.XI_XSC * (flip * (yp_as + dy + yshift) * 10.0) + c.XI_YSC * (
        flip * (xp_as + dx + xshift) * 10.0
    )
    # TODO, similarly with eta_ysc?
    eta = c.ETA_XSC * (flip * (yp_as + dy + yshift) * 10.0) + c.ETA_YSC * (
        flip * (xp_as + dx + xshift) * 10.0
    )
    col = ((xi / 36000.0) / (c.DETSIZE / 2.0) * maskfill + 1.0) / 2.0 * npixx
    row = ((eta / 36000.0) / (c.DETSIZE / 2.0) * maskfill + 1.0) / 2.0 * npixy
    cut = (
        (col > 0.0)
        & (col < 799.0)
        & (row > 0.0)
        & (row < 799.0)
        & (flags == 0)
    )
    flags[~cut] = 16
    ok_indices = np.nonzero(cut)[0]
    cut[ok_indices] = (
        mask[
            np.array(col[ok_indices], dtype="int64"),
            np.array(row[ok_indices], dtype="int64"),
        ]
        == 1.0
    )
    flags[~cut] = 6
    return eta, xi, flags


def compute_detector_orientation(fptrx, fptry, linearity, ok_indices):
    floor_x = np.array(fptrx, dtype="int64")
    floor_y = np.array(fptry, dtype="int64")
    corners = make_corners(floor_x, floor_y, fptrx, fptry, ok_indices)
    lin_indices = make_cal_indices(
        linearity["x"].shape, ok_indices, (floor_y, floor_x)
    )
    dx, dy = scale_corners(
        linearity["x"],
        linearity["y"],
        lin_indices,
        corners,
        fptrx.shape,
        ok_indices,
    )

    return dx, dy


def make_corners(floor_x, floor_y, fptrx, fptry, ok_indices):
    blt = (fptrx - floor_x)[ok_indices]
    blu = (fptry - floor_y)[ok_indices]
    corners = (
        (1 - blt) * (1 - blu),
        blt * (1 - blu),
        (1 - blt) * blu,
        blt * blu,
    )
    return corners


def post_wiggle_update_indices_and_flags(
    flags, fptrx, fptry, floor_x, floor_y, q, walk
):
    cut = (
        (fptrx > 0.0)
        & (fptrx < 479.0)
        & (fptry > 0.0)
        & (fptry < 479.0)
        & (flags == 0)
    )
    flags[~cut] = 9
    ok_indices = np.nonzero(cut)[0]
    walk_indices = make_cal_indices(
        walk["x"].shape, ok_indices, (q, floor_y, floor_x)
    )
    cut[ok_indices] = check_walk_flags(walk_indices, walk["x"], walk["y"])
    # TODO: flag is the same here intentionally?
    flags[~cut] = 9
    ok_indices = np.nonzero(cut)[0]
    return flags, ok_indices


def sum_corners(cal, cal_indices, corners):
    wr = cal.ravel()
    return sum([corners[ix] * wr[cal_indices[ix]] for ix in range(4)])


def scale_corners(cal_x, cal_y, cal_indices, corners, base_shape, ok_indices):
    out_x, out_y = np.zeros(base_shape), np.zeros(base_shape)
    out_x[ok_indices] = sum_corners(cal_x, cal_indices, corners)
    out_y[ok_indices] = sum_corners(cal_y, cal_indices, corners)
    return out_x, out_y


def perform_spatial_nonlinearity_correction(
    chunkid,
    fptrx,
    fptry,
    floor_x,
    floor_y,
    ix,
    q,
    walk,
    xdig,
    ydig,
):
    corners = make_corners(floor_x, floor_y, fptrx, fptry, ix)
    walk_indices = make_cal_indices(walk["x"].shape, ix, (q, floor_y, floor_x))
    walkx, walky = scale_corners(
        walk["x"], walk["y"], walk_indices, corners, fptrx.shape, ix
    )
    print_inline(chunkid + "Applying spatial non-linearity correction...")
    xp_as = (xdig - walkx) * c.ASPUM
    yp_as = (ydig - walky) * c.ASPUM
    fptrx = xp_as / 10.0 + 240.0
    fptry = yp_as / 10.0 + 240.0
    return fptrx, fptry, xp_as, yp_as


def make_cal_indices(shape, ok_indices, index_arrays):
    ix_slices = [index_array[ok_indices] for index_array in index_arrays]
    return (
        np.ravel_multi_index(
            np.array([*ix_slices[:-2], ix_slices[-2], ix_slices[-1]]), shape
        ),
        np.ravel_multi_index(
            np.array([*ix_slices[:-2], ix_slices[-2], ix_slices[-1] + 1]),
            shape,
        ),
        np.ravel_multi_index(
            np.array([*ix_slices[:-2], ix_slices[-2] + 1, ix_slices[-1]]),
            shape,
        ),
        np.ravel_multi_index(
            np.array([*ix_slices[:-2], ix_slices[-2] + 1, ix_slices[-1] + 1]),
            shape,
        ),
    )


def check_walk_flags(walk_indices, walk_x, walk_y):
    return reduce(
        or_,
        [
            walk[walk_indices[ix]] != -999
            for walk, ix in product((walk_x.ravel(), walk_y.ravel()), range(4))
        ],
    )


def wiggle_and_dig(chunkid, fptrx, fptry, ix, wiggle, x, xa, y, ya):
    # floating point corrections...?
    floor_x = np.array(fptrx, dtype="int64")
    floor_y = np.array(fptry, dtype="int64")
    blt = (fptrx - floor_x)[ix]
    blu = (fptry - floor_y)[ix]
    floor_x = floor_x[ix]
    floor_y = floor_y[ix]
    xa_ix = xa[ix]
    ya_ix = ya[ix]
    wigx, wigy = np.zeros_like(fptrx), np.zeros_like(fptrx)
    wigx[ix] = (1 - blt) * (wiggle["x"][xa_ix, floor_x]) + blt * (
        wiggle["x"][xa_ix, floor_x + 1]
    )
    wigy[ix] = (1 - blu) * (wiggle["y"][ya_ix, floor_y]) + blu * (
        wiggle["y"][ya_ix, floor_y + 1]
    )
    xdig = x + wigx / (10.0 * c.ASPUM)
    ydig = y + wigy / (10.0 * c.ASPUM)
    print_inline(chunkid + "Applying walk correction...")
    xdig_as = xdig * c.ASPUM
    ydig_as = ydig * c.ASPUM
    fptrx = xdig_as / 10.0 + 240.0
    fptry = ydig_as / 10.0 + 240.0
    return fptrx, fptry, xdig, ydig


def apply_wiggle_correction(chunkid, x, y):
    print_inline(chunkid + "Applying wiggle correction...")
    x_as = x * c.ASPUM
    y_as = y * c.ASPUM
    fptrx = x_as / 10.0 + 240.0
    fptry = y_as / 10.0 + 240.0
    return fptrx, fptry


def compute_stimstats_2(stims, band):
    print("Computing stim statistics and post-CSP corrections...")
    stim1avg = [stims[1]["x"].mean() * c.ASPUM, stims[1]["y"].mean() * c.ASPUM]
    stim2avg = [stims[2]["x"].mean() * c.ASPUM, stims[2]["y"].mean() * c.ASPUM]
    stim3avg = [stims[3]["x"].mean() * c.ASPUM, stims[3]["y"].mean() * c.ASPUM]
    stim4avg = [stims[4]["x"].mean() * c.ASPUM, stims[4]["y"].mean() * c.ASPUM]
    # Compute the stim separation.
    stimsep = (
        (stim2avg[0] - stim1avg[0])
        + (stim4avg[0] - stim3avg[0])
        + (stim1avg[1] - stim3avg[1])
        + (stim2avg[1] - stim4avg[1])
    ) / 4.0
    # Compute means and RMS values for each stim for each YA value stim1.
    # TODO: the code in the original compute_stimstats for this does nothing,
    #  assigning these calculations to an unused variable (ix) repeatedly.

    # This returns the pre-CSP stim positions (because eclipse==0).
    avgstim = avg_stimpos(band, 0)

    # Compute Y scale and shift factors: yprime_as = (m * y_as) + B.
    y1, y2 = (stim1avg[1] + stim2avg[1]) / 2.0, (
        stim3avg[1] + stim4avg[1]
    ) / 2.0
    Y1, Y2 = (
        (avgstim["y1"] + avgstim["y2"]) / 2.0,
        (avgstim["y3"] + avgstim["y4"]) / 2.0,
    )
    My = (Y1 - Y2) / (y1 - y2)
    By = (Y1 - My * y1) / c.ASPUM
    print("Init: FODC: Y scale and shift (microns): My=", My, "By=", By)

    # Compute X scale and shift factors: yprime_as = (m * x_as) + B.
    x1, x2 = (stim1avg[0] + stim3avg[0]) / 2.0, (
        stim2avg[0] + stim4avg[0]
    ) / 2.0
    X1, X2 = (
        (avgstim["x1"] + avgstim["x3"]) / 2.0,
        (avgstim["x2"] + avgstim["x4"]) / 2.0,
    )
    Mx = (X1 - X2) / (x1 - x2)
    Bx = (X1 - Mx * x1) / c.ASPUM
    print("Init: FODC: X scale and shift (microns): Mx=", Mx, "Bx=", Bx)

    stims[1]["xs"] = stims[1]["x"] * Mx + Bx
    stims[1]["ys"] = stims[1]["y"] * My + By
    stims[2]["xs"] = stims[2]["x"] * Mx + Bx
    stims[2]["ys"] = stims[2]["y"] * My + By
    stims[3]["xs"] = stims[3]["x"] * Mx + Bx
    stims[3]["ys"] = stims[3]["y"] * My + By
    stims[4]["xs"] = stims[4]["x"] * Mx + Bx
    stims[4]["ys"] = stims[4]["y"] * My + By

    # Compute the new mean positions (in arcseconds).
    stim1avgs = [
        stims[1]["xs"].mean() * c.ASPUM,
        stims[1]["ys"].mean() * c.ASPUM,
    ]
    stim2avgs = [
        stims[2]["xs"].mean() * c.ASPUM,
        stims[2]["ys"].mean() * c.ASPUM,
    ]
    stim3avgs = [
        stims[3]["xs"].mean() * c.ASPUM,
        stims[3]["ys"].mean() * c.ASPUM,
    ]
    stim4avgs = [
        stims[4]["xs"].mean() * c.ASPUM,
        stims[4]["ys"].mean() * c.ASPUM,
    ]

    # Fit straight line to YA>2 and YB==2 points.
    # This could be written more efficiently, but it's an attempt at a faithful
    #  port of the GALEX code (from Tom Barlow) which was written in C.
    ix1 = ((stims[1]["ya"] > 2) & (stims[1]["yb"] == 2)).nonzero()[0]
    ix2 = ((stims[2]["ya"] > 2) & (stims[2]["yb"] == 2)).nonzero()[0]
    ix3 = ((stims[3]["ya"] > 2) & (stims[3]["yb"] == 2)).nonzero()[0]
    ix4 = ((stims[4]["ya"] > 2) & (stims[4]["yb"] == 2)).nonzero()[0]
    w8 = np.ones(len(ix1) + len(ix2) + len(ix3) + len(ix4))
    x8 = np.concatenate(
        (
            stims[1]["yap"][ix1],
            stims[2]["yap"][ix2],
            stims[3]["yap"][ix3],
            stims[4]["yap"][ix4],
        ),
        axis=0,
    )
    y8 = np.concatenate(
        (
            stims[1]["ys"][ix1] - stim1avgs[1] / c.ASPUM,
            stims[2]["ys"][ix2] - stim2avgs[1] / c.ASPUM,
            stims[3]["ys"][ix3] - stim3avgs[1] / c.ASPUM,
            stims[4]["ys"][ix4] - stim4avgs[1] / c.ASPUM,
        ),
        axis=0,
    )

    print("NOTE: Found,", len(w8), "points for YA correction fit.")
    yac_coef1, yac_coef0 = np.polyfit(x8, y8, 1)

    print("Scal: YA correction coef for YB=2:", yac_coef0, yac_coef1)
    # Compute yb shift factors == zero for all.
    yac_ybs = np.zeros(8)
    coef0_yb = np.zeros(8) + yac_coef0
    coef1_yb = np.zeros(8) + yac_coef1
    # Set user slope adjustment. best slope adjustments from September 2010.
    # YB == 2...
    slope_scale = 1.04
    print("NOTE: Using slope scale of,", slope_scale, "for YB==2.")
    rr1 = yac_coef1 * slope_scale
    rr0 = (yac_coef0 + (16.0 * yac_coef1)) - (16.0 * rr1)
    coef0_yb[2] = rr0
    coef1_yb[2] = rr1
    print("New: YA correction coef (YB==2):", coef0_yb[2], coef1_yb[2])
    # YB == 3,4...
    slope_scale = 1.06
    print("NOTE: Using slope scale of,", slope_scale, "for YB==3.")
    rr1 = yac_coef1 * slope_scale
    rr0 = (yac_coef0 + (16.0 * yac_coef1)) - (16.0 * rr1)
    coef0_yb[3] = rr0
    coef1_yb[3] = rr1
    coef0_yb[4] = rr0
    coef1_yb[4] = rr1
    print("New: YA correction coef (YB==3):", coef0_yb[3], coef1_yb[3])
    print("NOTE: Using slope scale of,", slope_scale, "for YB==4.")
    print("New: YA correction coef (YB==4):", coef0_yb[4], coef1_yb[4])
    # Fill in look up array.
    yac = np.zeros([40, 8])
    for yb in range(8):
        for ya in range(40):
            yac[ya][yb] = (
                coef0_yb[yb] + (float(ya) * coef1_yb[yb])
            ) + yac_ybs[yb]
    for stim_ix in range(1, 5):
        stims[stim_ix]["yac"] = yac[
            np.array(stims[stim_ix]["yap"], dtype="int64"),
            np.array(stims[stim_ix]["yb"], dtype="int64"),
        ]
    return Mx, Bx, My, By, stimsep, yac


# TODO: the original version of this relies on stims computed using a
#  90-arcsecond window, while still using 20-arcsecond window in the rest of
#  the function. is this right?
def perform_yac_correction(band, eclipse, stims, data):
    Mx, Bx, My, By, stimsep, yactbl = compute_stimstats_2(stims, band)
    wig2, wig2data, wlk2, wlk2data, clk2, clk2data = post_csp_caldata()
    corrected_x = Mx * data["x"] + Bx
    corrected_y = My * data["y"] + By
    yac = rtaph_yac(yactbl, data["ya"], data["yb"], data["yamc"], eclipse)
    corrected_y -= yac
    yac = rtaph_yac2(
        data["q"],
        data["xb"],
        data["yb"],
        data["ya"],
        corrected_y,
        wig2,
        wig2data,
        wlk2,
        wlk2data,
        clk2,
        clk2data,
    )
    corrected_y += yac
    data["x"] = corrected_x
    data["y"] = corrected_y


def apply_yac_correction(band, eclipse, q, raw6file, x, xb, y, ya, yamc, yb):
    (Mx, Bx, My, By, stimsep, yactbl) = compute_stimstats(
        raw6file, band, eclipse
    )
    wig2, wig2data, wlk2, wlk2data, clk2, clk2data = post_csp_caldata()
    x = Mx * x + Bx
    y = My * y + By
    yac = rtaph_yac(yactbl, ya, yb, yamc, eclipse)
    y = y - yac
    yac = rtaph_yac2(
        q,
        xb,
        yb,
        ya,
        y,
        wig2,
        wig2data,
        wlk2,
        wlk2data,
        clk2,
        clk2data,
    )
    y = y + yac
    return x, y


def apply_stim_distortion_correction(
    chunkid,
    disthead,
    distortion,
    flags,
    ok_indices,
    stim_coef0,
    stim_coef1,
    t,
    xoffset,
    xp_as,
    yoffset,
    yp_as,
):
    print_inline(chunkid + "Applying stim distortion correction...")
    (
        cube_x0,
        cube_dx,
        cube_y0,
        cube_dy,
        cube_d0,
        cube_dd,
        cube_nd,
        cube_nc,
        cube_nr,
    ) = (
        disthead["DC_X0"],
        disthead["DC_DX"],
        disthead["DC_Y0"],
        disthead["DC_DY"],
        disthead["DC_D0"],
        disthead["DC_DD"],
        disthead["NAXIS3"],
        disthead["NAXIS1"],
        disthead["NAXIS2"],
    )
    ss = stim_coef0 + (t * stim_coef1)  # stim separation
    col, row, depth = np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))
    col[ok_indices] = (xp_as[ok_indices] - cube_x0) / cube_dx
    row[ok_indices] = (yp_as[ok_indices] - cube_y0) / cube_dy
    depth[ok_indices] = (ss[ok_indices] - cube_d0) / cube_dd
    # [Future]: This throws an error sometimes like the following, may need
    # fixing...
    """PhotonPipe.py:262: RuntimeWarning: invalid value encountered in
                less depth[((depth < 0)).nonzero()[0]] = 0.
                PhotonPipe.py:263: RuntimeWarning: invalid value encountered in
                greater_equal depth[((depth >= cube_nd)).nonzero()[0]] = -1.
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
        & (np.array(depth, dtype="int64") <= 16)
    )
    flags[~cut] = 11
    ok_indices = np.nonzero(cut)[0]
    xshift, yshift = np.zeros(len(t)), np.zeros(len(t))
    raveled_ix = np.ravel_multi_index(
        np.array(
            [
                depth[ok_indices].astype(np.int64),
                row[ok_indices].astype(np.int64),
                col[ok_indices].astype(np.int64),
            ]
        ),
        distortion["x"].shape,
    )
    xshift[ok_indices] = distortion["x"].ravel()[raveled_ix]
    yshift[ok_indices] = distortion["y"].ravel()[raveled_ix]
    xshift = (xshift * c.ARCSECPERPIXEL) + xoffset
    yshift = (yshift * c.ARCSECPERPIXEL) + yoffset
    return xshift, yshift, flags, ok_indices


def decode_telemetry(band, chunkbeg, chunkend, chunkid, eclipse, raw6hdulist):
    t, photonbytes = unpack_raw6(chunkbeg, chunkend, chunkid, raw6hdulist)
    data = bitwise_decode_photonbytes(band, eclipse, photonbytes)
    data["t"] = t.byteswap().newbyteorder()
    # data["t"] = t
    return data


def unpack_raw6(chunkbeg, chunkend, chunkid, raw6hdulist):
    print_inline(chunkid + "Unpacking raw6 data...")
    t = np.array(raw6hdulist[1].data.field("t")[chunkbeg:chunkend])
    photonbytes = {}
    for byte in range(5):
        photonbytes[byte + 1] = np.array(
            raw6hdulist[1].data.field(f"phb{byte + 1}")[chunkbeg:chunkend],
            dtype=np.int16,
        )
    return t, photonbytes


def create_ssd_from_decoded_data(data, band, eclipse, verbose, margin=90.001):
    all_stim_indices = find_stims_index(
        data["x"], data["y"], band, eclipse, margin
    )
    stims = {}
    for stim_ix in range(1, 5):
        stim_indices = all_stim_indices[stim_ix - 1]
        stims[stim_ix] = {
            field: value[stim_indices] for field, value in data.items()
        }
        stims[stim_ix]["ix"] = np.full(
            stims[stim_ix]["x"].shape, stim_ix, dtype="int16"
        )
        # TODO: only necessary for post-CSP, but probably not a big
        #  optimization issue
        stims[stim_ix]["yap"] = rtaph_yap(
            data["ya"][stim_indices],
            data["yb"][stim_indices],
            data["yamc"][stim_indices],
        )
    # second part of create_ssd
    stimt = np.concatenate([stim["t"] for stim in stims.values()], axis=0)
    sortt = np.argsort(stimt)
    stimt = stimt[sortt]
    stimix = np.concatenate(
        [stim["ix"] for stim in stims.values()],
        axis=0,
    )[sortt]
    stimx_as = (
        np.concatenate([stim["x"] for stim in stims.values()], axis=0)
        * c.ASPUM
    )[sortt]
    stimy_as = (
        np.concatenate([stim["y"] for stim in stims.values()], axis=0)
        * c.ASPUM
    )[sortt]
    pinc = 1000
    avt, sep, num = [], [], []

    for i in range(0, len(stimt) - pinc, pinc):
        ix1 = (stimix[i : i + pinc] == 1).nonzero()[0]
        ix2 = (stimix[i : i + pinc] == 2).nonzero()[0]
        ix3 = (stimix[i : i + pinc] == 3).nonzero()[0]
        ix4 = (stimix[i : i + pinc] == 4).nonzero()[0]
        sx1, sy1 = (
            np.mean(stimx_as[i : i + pinc][ix1]),
            np.mean(stimy_as[i : i + pinc][ix1]),
        )
        sx2, sy2 = (
            np.mean(stimx_as[i : i + pinc][ix2]),
            np.mean(stimy_as[i : i + pinc][ix2]),
        )
        sx3, sy3 = (
            np.mean(stimx_as[i : i + pinc][ix3]),
            np.mean(stimy_as[i : i + pinc][ix3]),
        )
        sx4, sy4 = (
            np.mean(stimx_as[i : i + pinc][ix4]),
            np.mean(stimy_as[i : i + pinc][ix4]),
        )
        stim_sep = (
            (sx2 - sx1) + (sx4 - sx3) + (sy1 - sy3) + (sy2 - sy4)
        ) / 4.0
        stim_avt = sum(stimt[i : i + pinc]) / len(stimt[i : i + pinc])
        stim_num = len(ix1) + len(ix2) + len(ix3) + len(ix4)
        avt.append(stim_avt)
        sep.append(stim_sep)
        num.append(stim_num)

    m, C = np.polyfit(avt, sep, 1)
    if verbose > 1:
        print("	    stim_coef0, stim_coef1 = " + str(C) + ", " + str(m))

    return stims, C, m


def bitwise_decode_photonbytes(band, eclipse, photonbytes):
    print_inline("Band is {band}.".format(band=band))
    data = {}
    (xclk, yclk, xcen, ycen, xscl, yscl, xslp, yslp) = clk_cen_scl_slp(
        band, eclipse
    )
    # Bitwise "decoding" of the raw6 telemetry.
    data["q"] = ((photonbytes[4] & 3) << 3) + ((photonbytes[5] & 224) >> 5)
    data["xb"] = photonbytes[1] >> 5
    data["xamc"] = (
        np.array(((photonbytes[1] & 31) << 7), dtype="int16")
        + np.array(((photonbytes[2] & 254) >> 1), dtype="int16")
        - np.array(((photonbytes[1] & 16) << 8), dtype="int16")
    )
    data["yb"] = ((photonbytes[2] & 1) << 2) + ((photonbytes[3] & 192) >> 6)
    data["yamc"] = (
        np.array(((photonbytes[3] & 63) << 6), dtype="int16")
        + np.array(((photonbytes[4] & 252) >> 2), dtype="int16")
        - np.array(((photonbytes[3] & 32) << 7), dtype="int16")
    )
    data["xa"] = (
        ((photonbytes[5] & 16) >> 4)
        + ((photonbytes[5] & 3) << 3)
        + ((photonbytes[5] & 12) >> 1)
    )
    xraw0 = data["xb"] * xclk + data["xamc"]
    yraw0 = data["yb"] * yclk + data["yamc"]
    data["ya"] = (
        np.array(
            (
                (((yraw0 / (2 * yclk) - xraw0 / (2 * xclk)) + 10) * 32)
                + data["xa"]
            ),
            dtype="int64",
        )
        % 32
    )
    xraw = (
        xraw0 + np.array((((data["xa"] + 7) % 32) - 16), dtype="int64") * xslp
    )
    yraw = (
        yraw0 + np.array((((data["ya"] + 7) % 32) - 16), dtype="int64") * yslp
    )
    # Centering and scaling.
    data["x"] = (xraw - xcen) * xscl
    data["y"] = (yraw - ycen) * yscl
    return data


def retrieve_raw6(eclipse, band, outbase):
    if not eclipse:
        raise ValueError("Must specify eclipse if no raw6file.")
    else:
        raw6file = download_data(
            eclipse, band, "raw6", datadir=os.path.dirname(outbase)
        )
        if raw6file is None:
            raise ValueError("Unable to retrieve raw6 file for this eclipse.")
    return raw6file


def retrieve_scstfile(band, eclipse, outbase, scstfile):
    if not scstfile:
        if not eclipse:
            raise ValueError("Must specifiy eclipse if no scstfile.")
        else:
            scstfile = download_data(
                eclipse, band, "scst", datadir=os.path.dirname(outbase)
            )
        if scstfile is None:
            raise ValueError("Unable to retrieve SCST file for this eclipse.")
    return scstfile


def get_eclipse_from_header(eclipse, raw6file):
    hdulist = pyfits.open(raw6file)
    hdr = hdulist[0].header
    hdulist.close()
    if eclipse and (eclipse != hdr["eclipse"]):  # just a consistency check
        print(
            "Warning: eclipse mismatch {e0} vs. {e1} (header)".format(
                e0=eclipse, e1=hdr["eclipse"]
            )
        )
    eclipse = hdr["eclipse"]
    return eclipse


def unpack_data_chunk(data, chunkbeg, chunkend, copy=True):
    chunk = {}
    for variable_name, variable in data.items():
        if copy is True:
            chunk[variable_name] = variable[chunkbeg:chunkend].copy()
        else:
            chunk[variable_name] = variable[chunkbeg:chunkend]
    return chunk
