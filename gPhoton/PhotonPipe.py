"""
.. module:: PhotonPipe
   :synopsis: A recreation / port of key functionality of the GALEX mission
       pipeline to generate calibrated and sky-projected photon-level data from
       raw spacecraft and detector telemetry. Generates time-tagged photon lists
       given mission-produced -raw6, -scst, and -asprta data.
"""
from multiprocessing import Pool
import os
import time
import warnings

# Core and Third Party imports.
import numpy as np
import pyarrow
from pyarrow import parquet

# gPhoton imports.

from gPhoton.CalUtils import find_fuv_offset
from gPhoton.MCUtils import print_inline
from gPhoton._pipe_components import (
    retrieve_aspect_solution,
    retrieve_raw6,
    create_ssd_from_decoded_data,
    retrieve_scstfile,
    get_eclipse_from_header,
    perform_yac_correction,
    load_cal_data,
    load_raw6,
    process_chunk_in_shared_memory,
    chunk_data,
    process_chunk_in_unshared_memory,
)


# ------------------------------------------------------------------------------

from gPhoton._shared_memory_pipe_components import (
    unlink_cal_blocks,
    slice_into_shared_chunks,
    send_cals_to_shared_memory,
    get_column_from_shared_memory,
)


def photonpipe(
    outbase,
    band,
    raw6file=None,
    scstfile=None,
    aspfile=None,
    verbose=0,
    retries=20,
    eclipse=None,
    overwrite=True,
    chunksz=1000000,
    threads=4,
    share_memory=None,
):
    """
    Apply static and sky calibrations to -raw6 GALEX data, producing fully
        aspect-corrected and time-tagged photon list files.

    :param raw6file: Name of the raw6 file to use.

    :type raw6file: str

    :param scstfile: Spacecraft state file to use.

    :type scstfile: str

    :param band: Name of the band to use, either 'FUV' or 'NUV'.

    :type band: str

    :param outbase: Base of the output file names.

    :type outbase: str, pathlib.Path

    :param aspfile: Name of aspect file to use.

    :type aspfile: int

    :param nullfile: Name of output file to record NULL lines.

    :type nullfile: int

    :param verbose: Verbosity level, to be detailed later.

    :type verbose: int

    :param retries: Number of query retries to attempt before giving up.

    :type retries: int
    """
    if share_memory is None:
        share_memory = threads is not None

    if (share_memory is True) and (threads is None):
        warnings.warn(
            "Using shared memory without multithreading. "
            "This incurs a performance cost to no end."
        )

    outfile = "{outbase}.parquet".format(outbase=outbase)
    if os.path.exists(outfile):
        if overwrite:
            print(f"{outfile} already exists... deleting")
            os.remove(outfile)
        else:
            print(f"{outfile} already exists... aborting run")
            return outfile

    startt = time.time()

    # download raw6 if local file is not passed
    if raw6file is None:
        raw6file = retrieve_raw6(eclipse, band, outbase)
    # get / check eclipse # from raw6 header --
    eclipse = get_eclipse_from_header(eclipse, raw6file)
    print_inline("Processing eclipse {eclipse}".format(eclipse=eclipse))

    if band == "FUV":
        scstfile = retrieve_scstfile(band, eclipse, outbase, scstfile)
        xoffset, yoffset = find_fuv_offset(scstfile)
    else:
        xoffset, yoffset = 0.0, 0.0

    aspect = retrieve_aspect_solution(aspfile, eclipse, retries, verbose)

    cal_data = load_cal_data(raw6file, band, eclipse)
    if share_memory is True:
        cal_data = send_cals_to_shared_memory(cal_data)

    data, nphots = load_raw6(band, eclipse, raw6file, verbose)
    if eclipse > 37460:
        stim_coefficients = (5105.48, 0.0) # post-CSP only use the default per rtaph.c #1391
    else:
        stims, stim_coefficients = create_ssd_from_decoded_data(
            data, band, eclipse, verbose, margin=20
        )
        del stims
    # Post-CSP 'yac' corrections.
    if eclipse > 37460:
        stims_for_yac, yac_coef = create_ssd_from_decoded_data(
            data, band, eclipse, verbose, margin=90.001
        )
        data = perform_yac_correction(band, eclipse, stims_for_yac, data)
        del stims_for_yac, yac_coef
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
        unlink_cal_blocks(cal_data)
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

    variables_for_which_dictionary_compression_is_useful = [
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
                         "detrad"
                     ]
    if band == "FUV":
        variables_for_which_dictionary_compression_is_useful.append("x")

    # noinspection PyArgumentList
    parquet.write_table(
        pyarrow.Table.from_arrays(
            list(array_dict.values()), names=list(array_dict.keys())
        ),
        outfile,
        use_dictionary=variables_for_which_dictionary_compression_is_useful,
        version="2.0",
    )
    stopt = time.time()
    # TODO: consider:  awswrangler.s3.to_parquet()
    print_inline("")
    print("")
    if verbose:
        print("Runtime statistics:")
        print(
            " runtime		=	{seconds} sec. = ({minutes} min.)".format(
                seconds=stopt - startt, minutes=(stopt - startt) / 60.0
            )
        )
        print(f"	processed	=	{str(proc_count)} of {str(nphots)} events.")
        if proc_count < nphots:
            print("		WARNING: MISSING EVENTS! ")
        print(f"rate		=	{str(nphots / (stopt - startt))} photons/sec.")
        print("")
    return outfile


# ------------------------------------------------------------------------------
