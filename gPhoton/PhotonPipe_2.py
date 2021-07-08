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

# Core and Third Party imports.
import pyarrow
import pyarrow.parquet

# gPhoton imports.
import sh
from pyarrow import Tensor, plasma, ipc, FixedSizeBufferWriter
from pympler.process import _ProcessMemoryInfoProc

import gPhoton.cal as cal
from gPhoton.CalUtils import find_fuv_offset
import gPhoton.constants as c
from gPhoton.MCUtils import print_inline
from gPhoton._pipe_components import (
    retrieve_aspect_solution,
    retrieve_raw6,
    create_ssd_from_decoded_data,
    retrieve_scstfile,
    get_eclipse_from_header,
    perform_yac_correction,
    load_cal_data, load_raw6, process_chunk2,
)


# ------------------------------------------------------------------------------


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

    :type outbase: str

    :param aspfile: Name of aspect file to use.

    :type aspfile: int

    :param nullfile: Name of output file to record NULL lines.

    :type nullfile: int

    :param verbose: Verbosity level, to be detailed later.

    :type verbose: int

    :param retries: Number of query retries to attempt before giving up.

    :type retries: int
    """

    outfile = "{outbase}.parquet".format(outbase=outbase)
    if os.path.exists(outfile):
        if overwrite:
            os.remove(outfile)
        else:
            print("{of} already exists... aborting run".format(of=outfile))

    startt = time.time()

    # Scale factor for the time column in the output csv so that it
    # can be recorded as an int in the database.
    dbscale = 1000

    # download raw6 if local file is not passed
    if raw6file is None:
        raw6file = retrieve_raw6(eclipse, band, outbase)
    # get / check eclipse # from raw6 header --
    eclipse = get_eclipse_from_header(eclipse, raw6file)
    print_inline("Processing eclipse {eclipse}".format(eclipse=eclipse))

    cal_data, distortion_cube = load_cal_data(band, eclipse)

    if band == "FUV":
        scstfile = retrieve_scstfile(band, eclipse, outbase, scstfile)
        xoffset, yoffset = find_fuv_offset(scstfile)
    else:
        xoffset, yoffset = 0.0, 0.0

    print_inline("Loading mask file...")
    mask, maskinfo = cal.mask(band)
    maskfill = c.DETSIZE / (mask.shape[0] * maskinfo["CDELT2"])

    aspect = retrieve_aspect_solution(aspfile, eclipse, retries, verbose)

    data, nphots = load_raw6(band, eclipse, raw6file, verbose)
    stims, stim_coefficients = create_ssd_from_decoded_data(
        data, band, eclipse, verbose, margin=20
    )
    del stims
    # Post-CSP 'yac' corrections.
    if eclipse > 37460:
        stims_for_yac, yac_coef = create_ssd_from_decoded_data(
            data, band, eclipse, verbose, margin=90.001
        )
        # impure function, modifies data inplace
        perform_yac_correction(band, eclipse, stims_for_yac, data)
        del stims_for_yac
        del yac_coef
    results = {}
    variable_names = [
        key for key in data.keys()
    ]
    table_indices = []
    total_chunks = range(int(nphots / chunksz) + 1)
    for chunk_ix in total_chunks:
        chunkbeg, chunkend = chunk_ix * chunksz, (chunk_ix + 1) * chunksz
        if chunkend > nphots:
            chunkend = None
        table_indices.append((chunkbeg, chunkend))
    total_chunks = len(table_indices)
    client = plasma.connect(f"/tmp/plasma1")
    object_ids = {}
    for chunk_ix in range(total_chunks):
        tensors = [
            Tensor.from_numpy(array[slice(*table_indices[chunk_ix])]) for array
            in data.values()
        ]
        object_ids[chunk_ix] = [
            plasma.ObjectID.from_random()
            for _ in tensors
        ]
        tensor_sizes = [
            ipc.get_tensor_size(tensor)
            for tensor in tensors
        ]
        buffers = [
            client.create(object_id, tensor_size)
            for object_id, tensor_size in
            zip(object_ids[chunk_ix], tensor_sizes)
        ]
        streams = [
            FixedSizeBufferWriter(buffer) for buffer in buffers
        ]
        for tensor, stream, object_id in zip(tensors, streams,
                                             object_ids[chunk_ix]):
            ipc.write_tensor(tensor, stream)
            client.seal(object_id)
            del tensor
        del tensors
    client.disconnect()  # drop references from main thread to plasma objects
    if threads is not None:
        pool = Pool(threads)
    else:
        pool = None
    for chunk_ix in reversed(range(total_chunks)):  # popping from end of list
        chunk_title = f"{str(chunk_ix + 1)} of {str(total_chunks)}:"
        index_data = {
            'object_ids': object_ids[chunk_ix],
            'variables': variable_names
        }
        process_args = (
            aspect,
            band,
            cal_data,
            distortion_cube,
            chunk_title,
            dbscale,
            index_data,
            mask,
            maskfill,
            stim_coefficients,
            xoffset,
            yoffset,
        )
        if pool is None:
            results[chunk_ix] = process_chunk2(*process_args)
        else:
            results[chunk_ix] = pool.apply_async(process_chunk2, process_args)
        del process_args
    del cal_data
    if pool is not None:
        pool.close()
        pool.join()
        results = {task: result.get() for task, result in results.items()}
    chunk_indices = sorted(results.keys())
    proc_count = sum([len(table) for table in results.values()])
    pyarrow.parquet.write_table(
        pyarrow.concat_tables([results[ix] for ix in chunk_indices]), outfile
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
            print("		WARNING: MISSING EVENTS!")
        print(f"rate		=	{str(nphots / (stopt - startt))} photons/sec.")
        print("")
    return

# ------------------------------------------------------------------------------
