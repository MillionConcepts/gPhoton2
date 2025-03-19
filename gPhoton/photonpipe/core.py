"""top-level handler module for gPhoton.execute_photonpipe"""

import time
import warnings
from multiprocessing import Pool
from sys import stdout
from typing import Optional

import numpy as np
import pyarrow
from pyarrow import parquet

from gPhoton.aspect import load_aspect_solution
from gPhoton.calibrate import find_fuv_offset
from gPhoton.io.mast import retrieve_raw6
from gPhoton.io.raw6 import load_raw6, get_eclipse_from_header
from gPhoton.photonpipe._steps import (
    create_ssd_from_decoded_data,
    perform_yac_correction,
    load_cal_data,
    process_chunk_in_shared_memory,
    chunk_data,
    process_chunk_in_unshared_memory,
    flag_ghosts
)
from gPhoton.pretty import print_inline
from gPhoton.reference import PipeContext
from gPhoton.sharing import (
    unlink_nested_block_dict,
    slice_into_shared_chunks,
    send_mapping_to_shared_memory,
    get_column_from_shared_memory,
)
from gPhoton.types import Pathlike


def execute_photonpipe(ctx: PipeContext, raw6file: Optional[Pathlike] = None):
    """
    Apply static and sky calibrations to -raw6 GALEX data, producing fully
        aspect_data-corrected and time-tagged photon list files.
    :param ctx: PipeContext object containing pipeline parameters

    :type ctx: PipeContext

    :param raw6file: Name of the raw6 file to use.

    :type raw6file: str
    """
    if ctx.share_memory is None:
        share_memory = ctx.threads is not None
    else:
        share_memory = ctx.share_memory

    if (share_memory is True) and (ctx.threads is None):
        warnings.warn(
            "Using shared memory without multithreading. "
            "This incurs a performance cost to no end."
        )
    startt = time.time()
    # download raw6 if local file is not passed
    if raw6file is None:
        print("downloading raw6file")
        raw6file = retrieve_raw6(ctx.eclipse, ctx.band, ctx.eclipse_path())
    # get / check eclipse # from raw6 header --
    eclipse = get_eclipse_from_header(raw6file, ctx.eclipse)
    print(f"Processing eclipse {eclipse}")
    if ctx.band == "FUV":
        xoffset, yoffset = find_fuv_offset(eclipse, aspect_dir=ctx.aspect_dir)
    else:
        xoffset, yoffset = 0.0, 0.0

    aspect = load_aspect_solution(eclipse, ctx.aspect, ctx.verbose,
                                  aspect_dir=ctx.aspect_dir)

    data, nphots = load_raw6(raw6file, ctx.verbose)
    # the stims are only actually used for post-CSP corrections, but we
    # temporarily retain them in both cases for brevity.
    # the use of a 90.001 separation angle and fixed stim coefficients
    # post-CSP is per original mission execute_pipeline; see rtaph.c #1391
    if eclipse > 37423:
        stims, _ = create_ssd_from_decoded_data(
            data, ctx.band, eclipse, ctx.verbose, margin=90.001
        )
        stim_coefficients = (5105.48, 0.0)
    else:
        stims, stim_coefficients = create_ssd_from_decoded_data(
            data, ctx.band, eclipse, ctx.verbose, margin=20
        )
    # Post-CSP 'yac' corrections.
    if eclipse > 37423:
        data = perform_yac_correction(ctx.band, eclipse, stims, data)
    cal_data = load_cal_data(stims, ctx.band, eclipse)
    if share_memory is True:
        cal_data = send_mapping_to_shared_memory(cal_data)
    legs = chunk_by_legs(aspect, ctx.chunksz, data, share_memory)
    del data
    # explode to dict of arrays for numba etc.
    aspect = {col: aspect[col].to_numpy() for col in aspect.columns}
    if share_memory is True:
        chunk_function = process_chunk_in_shared_memory
    else:
        chunk_function = process_chunk_in_unshared_memory
    pool = None if ctx.threads is None else Pool(ctx.threads)
    results, addresses = {}, []
    for leg_ix, leg in legs.items():
        addresses += [(leg_ix, chunk_ix) for chunk_ix in leg]
    for leg_ix, chunk_ix in addresses:
        infix = f" (leg {leg_ix}) " if len(legs) > 1 else ""
        title = f"{(chunk_ix + 1) * (leg_ix + 1)}{infix}/ {len(addresses)}: "
        chunk = legs[leg_ix].pop(chunk_ix)
        process_args = (
            aspect,
            ctx.band,
            cal_data,
            chunk,
            title,
            stim_coefficients,
            xoffset,
            yoffset,
            ctx.extended_photonlist
        )
        if pool is None:
            results[(leg_ix, chunk_ix)] = chunk_function(*process_args)
        else:
            results[(leg_ix, chunk_ix)] = pool.apply_async(
                chunk_function, process_args
            )
        del process_args
        del chunk
    if pool is not None:
        pool.close()
        # profiling code
        # while not all(res.ready() for res in results.values()):
        #     print(_ProcessMemoryInfoProc().rss / 1024 ** 3)
        #     time.sleep(0.1)
        # TODO, maybe: write per-leg photonlists aysnchronously
        pool.join()
        stdout.flush()
        print_inline("cleaning up processes")
        results = {task: result.get() for task, result in results.items()}
    if share_memory is True:
        print_inline("cleaning up cal data")
        unlink_nested_block_dict(cal_data)
    proc_count = 0
    outfiles = []
    for leg_ix, leg_ctx in enumerate(ctx.explode_legs()):
        leg_results = {}
        leg_addresses = tuple(filter(lambda k: k[0] == leg_ix, results.keys()))
        for address in leg_addresses:
            leg_results[address[1]] = results.pop(address)
        array_dict = retrieve_leg_results(leg_results, share_memory)
        # ghost flag for post CSP
        if eclipse > 37423:
            print("running ghost flag")
            array_dict = flag_ghosts(array_dict)
        proc_count += len(array_dict["t"])
        # noinspection PyArgumentList
        print(f"writing table to {leg_ctx['photonfile']}")
        parquet.write_table(
            pyarrow.Table.from_arrays(
                list(array_dict.values()), names=list(array_dict.keys())
            ),
            leg_ctx['photonfile'],
            row_group_size=5000000,
            use_dictionary=FIELDS_FOR_WHICH_DICTIONARY_COMPRESSION_IS_USEFUL,
            version="2.6",
        )
        outfiles.append(leg_ctx['photonfile'])
    stopt = time.time()
    print_inline("")
    print("")
    if ctx.verbose:
        seconds = stopt - startt
        rate = nphots / seconds
        print("Runtime statistics:")
        print(
            f" runtime		=	{round(seconds, 2)} sec. "
            f"= ({round(seconds/60, 2)} min.)"
        )
        print(f"  processed	   =   {str(proc_count)} of {str(nphots)} events.")
        print(f"rate		=	{round(rate, 2)} photons/sec.\n")
    return outfiles


def retrieve_leg_results(results, share_memory):
    array_dict = {}
    if share_memory is True:
        for name in results[0]:
            array_dict[name] = get_column_from_shared_memory(
                results, name, unlink=True
            )
    else:
        child_dicts = [results[ix] for ix in sorted(results.keys())]
        # TODO: this is memory-greedy.
        for name in child_dicts[0]:
            array_dict[name] = np.hstack(
                [child_dict[name] for child_dict in child_dicts]
            )
        del child_dicts
    return array_dict


def chunk_by_legs(aspect, chunksz, data, share_memory):
    chunks = {}
    bounds = []
    start = 0
    leg_groups = tuple(aspect.groupby('leg'))
    for leg_ix, leg in leg_groups:
        times = leg['time']
        start_ix = np.nonzero(data['t'][start:] >= times.iloc[0])[0][0]
        end_ix = np.nonzero(data['t'][start:] <= times.iloc[-1])[0][-1]
        bounds.append((start + start_ix, start + end_ix))
        leg_data = {
            field: value[start + start_ix:start + end_ix]
            for field, value in data.items()
        }
        start += end_ix
        if share_memory is True:
            chunks[leg_ix] = slice_into_shared_chunks(chunksz, leg_data)
        else:
            chunks[leg_ix] = chunk_data(chunksz, leg_data, copy=True)
    return chunks


# ------------------------------------------------------------------------------

FIELDS_FOR_WHICH_DICTIONARY_COMPRESSION_IS_USEFUL = [
    "t",
    "flags",
    "x",
    "y",
    "xa",
    "xb",
    "ya",
    "yb",
    "yamc",
    "xamc",
    "q",
]
