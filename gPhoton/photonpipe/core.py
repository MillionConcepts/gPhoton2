"""top-level handler module for gPhoton.execute_photonpipe"""

import time
import warnings
from multiprocessing import Pool
from pathlib import Path
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
)
from gPhoton.pretty import print_inline
from gPhoton.sharing import (
    unlink_nested_block_dict,
    slice_into_shared_chunks,
    send_mapping_to_shared_memory,
    get_column_from_shared_memory,
)
from gPhoton.types import GalexBand, Pathlike


def execute_photonpipe(
    outfile: Pathlike,
    band: GalexBand,
    raw6file: Optional[str] = None,
    verbose: int = 0,
    eclipse: Optional[int] = None,
    overwrite: int = True,
    chunksz: int = 1000000,
    threads: int = 4,
    share_memory: Optional[bool] = None,
    write_intermediate_variables: bool = False
):
    """
    Apply static and sky calibrations to -raw6 GALEX data, producing fully
        aspect_data-corrected and time-tagged photon list files.

    :param raw6file: Name of the raw6 file to use.

    :type raw6file: str

    :param scstfile: Spacecraft state file to use.

    :type scstfile: str

    :param band: Name of the band to use, either 'FUV' or 'NUV'.

    :type band: str

    :param outfile: Base of the output file names.

    :type outfile: str, pathlib.Path

    :param verbose: Verbosity level, to be detailed later.

    :type verbose: int

    """
    if share_memory is None:
        share_memory = threads is not None

    if (share_memory is True) and (threads is None):
        warnings.warn(
            "Using shared memory without multithreading. "
            "This incurs a performance cost to no end."
        )
    outfile = Path(outfile)
    if outfile.exists():
        if overwrite:
            print(f"{outfile} already exists... deleting")
            outfile.unlink()
        else:
            print(f"{outfile} already exists... aborting run")
            return outfile

    startt = time.time()
    # download raw6 if local file is not passed
    if raw6file is None:
        print(f"downloading raw6file")
        raw6file = retrieve_raw6(eclipse, band, outfile)
    # get / check eclipse # from raw6 header --
    eclipse = get_eclipse_from_header(raw6file, eclipse)
    print_inline("Processing eclipse {eclipse}".format(eclipse=eclipse))
    if band == "FUV":
        xoffset, yoffset = find_fuv_offset(eclipse)
    else:
        xoffset, yoffset = 0.0, 0.0

    aspect = load_aspect_solution(eclipse, verbose)

    data, nphots = load_raw6(raw6file, verbose)
    # the stims are only actually used for post-CSP corrections, but we
    # temporarily retain them in both cases for brevity.
    # the use of a 90.001 separation angle and fixed stim coefficients
    # post-CSP is per original mission execute_pipeline; see rtaph.c #1391
    if eclipse > 37460:
        stims, _ = create_ssd_from_decoded_data(
            data, band, eclipse, verbose, margin=90.001
        )
        stim_coefficients = (5105.48, 0.0)
    else:
        stims, stim_coefficients = create_ssd_from_decoded_data(
            data, band, eclipse, verbose, margin=20
        )
    # Post-CSP 'yac' corrections.
    if eclipse > 37460:
        data = perform_yac_correction(band, eclipse, stims, data)
    cal_data = load_cal_data(stims, band, eclipse)
    if share_memory is True:
        cal_data = send_mapping_to_shared_memory(cal_data)

    if share_memory is True:
        chunks = slice_into_shared_chunks(chunksz, data, nphots)
        chunk_function = process_chunk_in_shared_memory
    else:
        chunks = chunk_data(chunksz, data, nphots, copy=True)
        chunk_function = process_chunk_in_unshared_memory
    del data
    if threads is not None:
        pool = Pool(threads)
    else:
        pool = None
    results = {}
    chunk_indices = list(chunks.keys())
    for chunk_ix in chunk_indices:
        chunk = chunks.pop(chunk_ix)
        process_args = (
            aspect,
            band,
            cal_data,
            chunk,
            f"{str(chunk_ix + 1)} of {str(len(chunk_indices))}:",
            stim_coefficients,
            xoffset,
            yoffset,
            write_intermediate_variables
        )
        if pool is None:
            results[chunk_ix] = chunk_function(*process_args)
        else:
            results[chunk_ix] = pool.apply_async(chunk_function, process_args)
        del process_args
        del chunk
    if pool is not None:
        pool.close()
        # profiling code
        # while not all(res.ready() for res in results.values()):
        #     print(_ProcessMemoryInfoProc().rss / 1024 ** 3)
        #     time.sleep(0.1)
        pool.join()
        results = {task: result.get() for task, result in results.items()}
    # make sure this remains in order
    chunk_indices = sorted(results.keys())
    array_dict = {}
    if share_memory is True:
        unlink_nested_block_dict(cal_data)
        for name in results[0].keys():
            array_dict[name] = get_column_from_shared_memory(
                results, name, unlink=True
            )
    else:
        child_dicts = [results[ix] for ix in chunk_indices]
        # TODO: this is memory-greedy.
        for name in child_dicts[0].keys():
            array_dict[name] = np.hstack(
                [child_dict[name] for child_dict in child_dicts]
            )
        del child_dicts
    proc_count = len(array_dict["t"])

    dict_comp = VARIABLES_FOR_WHICH_DICTIONARY_COMPRESSION_IS_USEFUL.copy()
    if band == "FUV":
        dict_comp.append("x")

    # noinspection PyArgumentList
    parquet.write_table(
        pyarrow.Table.from_arrays(
            list(array_dict.values()), names=list(array_dict.keys())
        ),
        outfile,
        use_dictionary=dict_comp,
        version="2.6",
    )
    stopt = time.time()
    print_inline("")
    print("")
    if verbose:
        seconds = stopt - startt
        rate = nphots / seconds
        print("Runtime statistics:")
        print(f" runtime		=	{seconds} sec. = ({seconds/60} min.)")
        print(f"	processed	=	{str(proc_count)} of {str(nphots)} events.")
        if proc_count < nphots:
            print("		WARNING: MISSING EVENTS! ")
        print(f"rate		=	{rate} photons/sec.\n")
    return outfile


# ------------------------------------------------------------------------------

VARIABLES_FOR_WHICH_DICTIONARY_COMPRESSION_IS_USEFUL = [
    "t",
    "flags",
    "y",
    "xa",
    "xb",
    "ya",
    "yb",
    "yamc",
    "xamc",
    "q",
    "mask",
    "detrad",
]