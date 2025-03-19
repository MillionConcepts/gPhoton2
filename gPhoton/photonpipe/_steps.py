from itertools import product
from statistics import mean
from collections.abc import Mapping

import numpy as np
import pandas as pd
from cytoolz import keyfilter
from dustgoggles.structures import NestingDict

from gPhoton import cals, constants as c
from gPhoton.calibrate import (
    avg_stimpos,
    compute_flat_scale,
    rtaph_yac,
    rtaph_yac2,
    rtaph_yap,
    post_csp_caldata,
    find_stims_index,
)
from gPhoton.coords.gnomonic import gnomrev_simple
from gPhoton.photonpipe._numbafied import (
    interpolate_aspect_solutions,
    find_null_indices,
    unfancy_detector_coordinates,
    make_corners,
    or_reduce_minus_999,
    init_wiggle_arrays,
    float_between_wiggled_points,
    unfancy_distortion_component,
    sum_corners
)


from gPhoton.pretty import print_inline
from gPhoton.sharing import (
    reference_shared_memory_arrays,
    send_to_shared_memory,
)
from gPhoton.types import GalexBand


# variables actually used later in the pipeline
PIPELINE_VARIABLES = (
    "ra", "dec", "t", "detrad", "flags", "response", "mask"
)


def process_chunk_in_unshared_memory(
    aspect,
    band,
    cal_data,
    chunk,
    chunkid,
    stim_coefficients,
    xoffset,
    yoffset,
    write_intermediate_variables
):
    chunk = apply_all_corrections(
        aspect,
        band,
        cal_data,
        chunk,
        chunkid,
        stim_coefficients,
        xoffset,
        yoffset,
    )
    output = chunk | calibrate_photons_inline(band, cal_data, chunk, chunkid)
    if write_intermediate_variables is not True:
        output = keyfilter(lambda key: key in PIPELINE_VARIABLES, output)
    return output


def apply_all_corrections(
    aspect,
    band,
    cal_data,
    chunk,
    chunkid,
    stim_coefficients,
    xoffset,
    yoffset,
):
    chunk |= apply_on_detector_corrections(
        band,
        cal_data,
        chunk,
        chunkid,
        stim_coefficients,
        xoffset,
        yoffset,
    )
    return chunk | apply_aspect_solution(aspect, chunk, chunkid)


def process_chunk_in_shared_memory(
    aspect,
    band,
    cal_block_info,
    block_info,
    chunk_title,
    stim_coefficients,
    xoffset,
    yoffset,
    write_intermediate_variables
):
    chunk_blocks, chunk = reference_shared_memory_arrays(block_info)
    cal_data = {}
    all_cal_blocks = []
    for cal_name, cal_info in cal_block_info.items():
        cal_blocks, cal_arrays = reference_shared_memory_arrays(cal_info)
        all_cal_blocks.append(cal_blocks)
        cal_data[cal_name] = cal_arrays
    chunk = apply_all_corrections(
        aspect,
        band,
        cal_data,
        chunk,
        chunk_title,
        stim_coefficients,
        xoffset,
        yoffset,
    )
    chunk |= calibrate_photons_inline(band, cal_data, chunk, chunk_title)
    if write_intermediate_variables is not True:
        chunk = keyfilter(lambda key: key in PIPELINE_VARIABLES, chunk)
    processed_block_info = send_to_shared_memory(chunk)
    for block in chunk_blocks.values():
        block.close()
        block.unlink()
    return processed_block_info


def calibrate_photons_inline(band, cal_data, chunk, chunkid):
    print_inline(f"{chunkid}Applying detector-space calibrations...")
    col, row = chunk["col"], chunk["row"]
    # compute all data on detector to allow maximally flexible selection of
    # edge radii downstream
    det_indices = np.where((col > 0) & (col < 800) & (row > 0) & (row < 800))
    col_ix = col[det_indices].astype(np.int32)
    row_ix = row[det_indices].astype(np.int32)
    # fields that are only meaningful / calculable on the detector
    det_fields = {
        "mask": cal_data["mask"]["array"][col_ix, row_ix] == 0,
        "flat": cal_data["flat"]["array"][col_ix, row_ix],
        "scale": compute_flat_scale(
            chunk["t"][det_indices], band
        ).astype("f4"),
    }
    del col_ix, row_ix
    det_fields["response"] = det_fields["flat"] * det_fields["scale"]
    output_columns = {
        "col": col,
        "row": row,
        "detrad": np.sqrt((col - 400) ** 2 + (row - 400) ** 2),
    }
    for field, values in det_fields.items():
        if field == "mask":
            print_inline(chunkid + "Applying hotspot mask...")
        output_columns[field] = np.full(
            chunk["t"].size, np.nan, dtype=values.dtype
        )
        output_columns[field][det_indices] = values
    return output_columns


def apply_aspect_solution(aspect, chunk, chunkid):
    # This gives the index of the aspect_data time that comes _before_
    # each photon time. Without the '-1' it will give the index
    # of the aspect_data time _after_ the photon time.
    print_inline(chunkid + "Mapping photon times to aspect_data times...")
    aspix = np.digitize(chunk["t"], aspect["time"]) - 1
    print_inline(chunkid + "Applying dither correction...")
    # Use only photons that are bracketed by valid aspect_data solutions
    # and have been not themselves been flagged as invalid.
    flags = chunk["flags"]
    cut = (
        (aspix > 0)
        & (aspix < (len(aspect["time"]) - 1))
        & ((flags == 0) | (flags == 6))
    )
    flags[~cut] = 7
    ok_indices = np.nonzero(cut)[0]
    aspect_slice = aspix[ok_indices]
    print_inline(chunkid + "Interpolating aspect_data solutions...")
    deta, dxi = interpolate_aspect_solutions(
        aspect_slice,
        aspect["time"],
        chunk["t"],
        aspect["eta"],
        aspect["xi"],
        ok_indices,
    )
    print_inline(chunkid + "Mapping to sky...")
    ra, dec = np.zeros(chunk["t"].shape), np.zeros(chunk["t"].shape)
    ra[ok_indices], dec[ok_indices] = gnomrev_simple(
        chunk["xi"][ok_indices] + dxi[ok_indices],
        chunk["eta"][ok_indices] + deta[ok_indices],
        aspect["ra"][aspect_slice],
        aspect["dec"][aspect_slice],
        -aspect["roll"][aspect_slice],
        1 / 36000.0,
        0.0,
        0.0
    )
    null_ix, flags = find_null_indices(
        aspect["flags"], aspect_slice, aspect["time"], flags, ok_indices
    )
    ra[null_ix] = np.nan
    dec[null_ix] = np.nan
    return {"ra": ra, "dec": dec, "flags": flags}


def apply_on_detector_corrections(
    band,
    cal_data,
    chunk,
    chunkid,
    stim_coefficients,
    xoffset,
    yoffset,
):
    q = chunk["q"]
    t = chunk["t"]
    x = chunk["x"]
    xa = chunk["xa"]
    y = chunk["y"]
    ya = chunk["ya"]
    flags = np.zeros(len(t), dtype=np.uint8)
    fptrx, fptry = apply_wiggle_correction(chunkid, x, y)
    # This and other lines like it below are to verify that the
    # event is still on the detector.
    cut = (fptrx > 0.0) & (fptrx < 479.0) & (fptry > 0.0) & (fptry < 479.0)
    flags[~cut] = 8
    ok_indices = np.nonzero(cut)[0]
    print_inline(chunkid + "Applying walk correction...")
    fptrx, fptry, xdig, ydig = wiggle_and_dig(
        fptrx,
        fptry,
        ok_indices,
        cal_data["wiggle"]["x"],  # unpacking for numba
        cal_data["wiggle"]["y"],
        x,
        xa,
        y,
        ya,
    )
    floor_x, floor_y = fptrx.astype("i4"), fptry.astype("i4")
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
        cal_data["distortion"],
        flags,
        ok_indices,
        stim_coefficients,
        t,
        xoffset,
        xp_as,
        yoffset,
        yp_as,
    )
    xi, eta, col, row, flags = unfancy_detector_coordinates(
        band,
        dx,
        dy,
        flags,
        xp_as,
        xshift,
        yp_as,
        yshift,
    )
    return {"xi": xi, "eta": eta, "col": col, "row": row, "flags": flags}


def compute_detector_orientation(fptrx, fptry, linearity, ok_indices):
    floor_x, floor_y = fptrx.astype("i4"), fptry.astype("i4")
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


def scale_corners(cal_x, cal_y, cal_indices, corners, base_shape, ok_indices):
    out_x, out_y = np.zeros(base_shape, "f4"), np.zeros(base_shape, dtype="f4")
    out_x[ok_indices] = sum_corners(cal_x, *cal_indices, corners)
    out_y[ok_indices] = sum_corners(cal_y, *cal_indices, corners)
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
    x = None
    for walk, ix in product((walk_x.ravel(), walk_y.ravel()), range(4)):
        x = or_reduce_minus_999(walk, walk_indices[ix], x)
    return x


def wiggle_and_dig(fptrx, fptry, ix, wig_x, wig_y, x, xa, y, ya):
    # floating point corrections...?

    floor_x, floor_y = fptrx.astype("i4"), fptry.astype("i4")
    blt, blu, floor_x, floor_y, wigx, wigy, xa_ix, ya_ix = init_wiggle_arrays(
        floor_x, floor_y, fptrx, fptry, ix, xa, ya
    )
    # avoiding advanced indexing for numba...
    thing_x = float_between_wiggled_points(blt, floor_x, wig_x, xa_ix)
    thing_y = float_between_wiggled_points(blu, floor_y, wig_y, ya_ix)
    wigx[ix] = thing_x
    wigy[ix] = thing_y
    xdig = x + wigx / (10.0 * c.ASPUM)
    ydig = y + wigy / (10.0 * c.ASPUM)
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
    stimavg = {
        stim: {
            "x": stims[stim]["x"].mean() * c.ASPUM,
            "y": stims[stim]["y"].mean() * c.ASPUM,
        }
        for stim in range(1, 5)
    }
    # Compute the stim separation.
    stimsep = (
        (stimavg[2]["x"] - stimavg[1]["x"])
        + (stimavg[4]["x"] - stimavg[3]["x"])
        + (stimavg[1]["y"] - stimavg[3]["y"])
        + (stimavg[2]["y"] - stimavg[4]["y"])
    ) / 4
    # Compute means and RMS values for each stim for each YA value stim1.

    # This returns the pre-CSP stim positions (because eclipse==0).
    pre_csp_avg = avg_stimpos(band, 0)

    # Compute scale and shift factors, e.g. yprime_as = (m * y_as) + B.
    scale, shift = {}, {}
    for axis in ("x", "y"):
        # order of corners differs between axes
        c1, c2, c3, c4 = (1, 2, 3, 4) if axis == "y" else (1, 3, 2, 4)
        side_1, side_2 = (
            (stimavg[c1][axis] + stimavg[c2][axis]) / 2.0,
            (stimavg[c3][axis] + stimavg[c4][axis]) / 2.0,
        )
        side_1_0, side_2_0 = (
            mean((pre_csp_avg[f"{axis}{c1}"], pre_csp_avg[f"{axis}{c2}"])),
            mean((pre_csp_avg[f"{axis}{c3}"], pre_csp_avg[f"{axis}{c4}"])),
        )
        scale[axis] = (side_1_0 - side_2_0) / (side_1 - side_2)
        shift[axis] = (side_1_0 - scale[axis] * side_1) / c.ASPUM
        print(
            f"Init: FODC: {axis} scale and shift (microns): "
            f"{scale[axis]}, {shift[axis]}"
        )
    for stim, axis in product(range(1, 5), ("x", "y")):
        stims[stim][f"{axis}s"] = stims[stim][axis] * scale[axis] + shift[axis]

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
    # noinspection PyTupleAssignmentBalance
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
    return scale["x"], shift["x"], scale["y"], shift["y"], stimsep, yac


def perform_yac_correction(
    band: GalexBand,
    eclipse: int,
    stims: Mapping[int, Mapping],
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    perform y-axis corrections to raw detector positions.
    :param band: "NUV" or "FUV:
    :param eclipse: GALEX eclipse number
    :param stims: mapping of stim # to stim characteristics, produced by
    create_ssd_from_decoded_data()
    :param data: dataframe of L0 event data produced by load_raw6()
    :return: yac-corrected version of input data
    """
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
    return data


def apply_stim_distortion_correction(
    chunkid,
    distortion,
    flags,
    ok_indices,
    stim_coefficients,
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
    ) = distortion[
        "header"
    ].astype(np.float32)  # unpacking for numba compiler introspection
    col, depth, ok_indices, row, xshift, yshift = unfancy_distortion_component(
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
    )
    try:
        raveled_ix = np.ravel_multi_index(
            np.array(
                [
                    depth[ok_indices].astype("i4"),
                    row[ok_indices].astype("i4"),
                    col[ok_indices].astype("i4"),
                ]
            ),
            distortion["x"].shape,
        )
    except ValueError as value_error:
        if "invalid entry in coordinates array" in str(value_error):
            raise ValueError("bad distortion correction solution. Quitting.")
        raise
    xshift[ok_indices] = distortion["x"].ravel()[raveled_ix]
    yshift[ok_indices] = distortion["y"].ravel()[raveled_ix]
    xshift = (xshift * c.ARCSECPERPIXEL) + xoffset
    yshift = (yshift * c.ARCSECPERPIXEL) + yoffset
    return xshift.astype("f4"), yshift.astype("f4"), flags, ok_indices


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

    # noinspection PyTupleAssignmentBalance
    m, C = np.polyfit(avt, sep, 1)
    if verbose > 1:
        print("\nstim_coef0, stim_coef1 = " + str(C) + ", " + str(m))

    return stims, (C, m)


def unpack_data_chunk(data, chunkbeg, chunkend, copy=True):
    chunk = {}
    for variable_name, variable in data.items():
        if copy is True:
            chunk[variable_name] = variable[chunkbeg:chunkend].copy()
        else:
            chunk[variable_name] = variable[chunkbeg:chunkend]
    return chunk


def chunk_data(chunksz, data, copy=True):
    length = len(next(data.values()))
    chunk_slices = []
    for chunk_ix in range(int(length / chunksz) + 1):
        chunkbeg, chunkend = chunk_ix * chunksz, (chunk_ix + 1) * chunksz
        if chunkend > length:
            chunkend = None
        chunk_slices.append((chunkbeg, chunkend))
    return {
        chunk_ix: unpack_data_chunk(data, *indices, copy=copy)
        for chunk_ix, indices in enumerate(chunk_slices)
    }


def load_cal_data(stims, band, eclipse):
    cal_data = NestingDict()
    for cal_type in ("wiggle", "walk", "linearity"):
        print_inline(f"Loading {cal_type} files...")
        cal_data[cal_type]["x"], _ = getattr(cals, cal_type)(band, "x")
        cal_data[cal_type]["y"], _ = getattr(cals, cal_type)(band, "y")
    print_inline("Loading flat field...")
    cal_data["flat"]["array"], _ = cals.flat(band)
    print_inline("Loading mask...")
    print_inline("Loading mask file...")
    cal_data["mask"]["array"], _ = cals.mask(band)
    cal_data["mask"]["array"] = cal_data["mask"]["array"].astype(np.uint8)
    # This is for the post-CSP stim distortion corrections.
    # TODO: it gets applied elsewhere, too. change feedback?
    #  ...actually just document the data flow here better, as best we can,
    #  given our limited knowledge
    print_inline("Loading distortion files...")
    if eclipse > 37423:
        stimsep = compute_stimstats_2(stims, band)[-2]
    else:
        stimsep = c.STIMSEP
    print_inline(f" Using stim separation of : {stimsep}")
    cal_data["distortion"]["x"], distortion_header = cals.distortion(
        band, "x", eclipse, stimsep
    )
    cal_data["distortion"]["y"], _ = cals.distortion(
        band, "y", eclipse, stimsep
    )
    cal_data["distortion"]["header"] = np.array(
        [
            distortion_header[field]
            for field in (
                "DC_X0",
                "DC_DX",
                "DC_Y0",
                "DC_DY",
                "DC_D0",
                "DC_DD",
                "NAXIS3",
                "NAXIS1",
                "NAXIS2",
            )
        ]
    )
    return cal_data


def flag_ghosts(leg_data):
    """ Adds a flag for ghosts in post-CSP eclipses
    using the concentration of low YA values as a ghost
    indicator.

    Not very fast, could use some work to speed it up.
    """
    from quickbin import bin2d

    ra = leg_data["ra"]
    dec = leg_data["dec"]
    ya = leg_data["ya"]
    col = leg_data["col"]
    row = leg_data["row"]

    print("filtering")
    valid_indices = (col >= 0) & (col <= 800) & (row >= 0) & (row <= 800)
    # mask NaNs
    nan_free_indices = valid_indices & ~np.isnan(ya) & ~np.isnan(ra) & ~np.isnan(dec)
    ra_valid, dec_valid, ya_valid = ra[nan_free_indices],\
                                    dec[nan_free_indices],\
                                    ya[nan_free_indices]
    print("binning")
    n_bins = 1600
    binned = bin2d(
        ra_valid,
        dec_valid,
        ya_valid,
        ('mean', 'count'),
        n_bins=n_bins
        )
    filtered = (binned['mean'] < 6) & (binned['count'] > 15)

    # get the ra/dec edges
    ra_min, ra_max = np.min(ra_valid), np.max(ra_valid)
    dec_min, dec_max = np.min(dec_valid), np.max(dec_valid)
    ra_edges = np.linspace(ra_min, ra_max, n_bins + 1)
    dec_edges = np.linspace(dec_min, dec_max, n_bins + 1)

    # get flagged ra/dec
    flagged_pixels = np.array(np.nonzero(filtered)).T
    ra_flagged = (ra_edges[flagged_pixels[:, 0]] + ra_edges[flagged_pixels[:, 0] + 1]) / 2
    dec_flagged = (dec_edges[flagged_pixels[:, 1]] + dec_edges[flagged_pixels[:, 1] + 1]) / 2

    print("masking")
    ra_bin_width = (ra_max - ra_min) / n_bins
    dec_bin_width = (dec_max - dec_min) / n_bins

    mask = np.zeros_like(ra, dtype=bool)

    chunk_size = 100000
    for start in range(0, len(ra), chunk_size):
        end = min(start + chunk_size, len(ra))
        ra_chunk = ra[start:end]
        dec_chunk = dec[start:end]

        # done twice?
        valid_chunk_indices = ~np.isnan(ra_chunk) & ~np.isnan(dec_chunk)
        ra_chunk = ra_chunk[valid_chunk_indices]
        dec_chunk = dec_chunk[valid_chunk_indices]

        if len(ra_chunk) > 0:
            for r, d in zip(ra_flagged, dec_flagged):
                ra_mask = (ra_chunk >= r - ra_bin_width / 2) & (ra_chunk < r + ra_bin_width / 2)
                dec_mask = (dec_chunk >= d - dec_bin_width / 2) & (dec_chunk < d + dec_bin_width / 2)
                combo_mask = ra_mask & dec_mask
                mask[start:end][valid_chunk_indices] |= combo_mask

    leg_data["flags"][mask] = 120

    return leg_data
