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
import gPhoton.cal as cal
from gPhoton.CalUtils import find_fuv_offset
import gPhoton.constants as c
from gPhoton.MCUtils import print_inline
from gPhoton._pipe_components import (
    retrieve_aspect_solution,
    process_chunk_in_unshared_memory,
    retrieve_raw6,
    create_ssd_from_decoded_data,
    retrieve_scstfile,
    get_eclipse_from_header,
    perform_yac_correction,
    chunk_data,
    load_cal_data, load_raw6,
)


# ------------------------------------------------------------------------------
# import line_profiler
# lp = line_profiler.LineProfiler()
# @lp
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
    chunks = chunk_data(chunksz, data, nphots, copy=True)
    del data
    total_chunks = len(chunks)
    if threads is not None:
        pool = Pool(threads)
    else:
        pool = None
    for chunk_ix in reversed(range(total_chunks)):  # popping from end of list
        chunk_title = f"{str(chunk_ix + 1)} of {str(total_chunks)}:"
        chunk = chunks.pop()
        process_args = (
            aspect,
            band,
            cal_data,
            distortion_cube,
            chunk,
            chunk_title,
            dbscale,
            mask,
            maskfill,
            stim_coefficients,
            xoffset,
            yoffset,
        )
        if pool is None:
            results[chunk_ix] = process_chunk_in_unshared_memory(*process_args)
        else:
            results[chunk_ix] = pool.apply_async(process_chunk_in_unshared_memory, process_args)
        del chunk
        del process_args
    if pool is not None:
        pool.close()
        # while not all(res.ready() for res in results.values()):
        #     a = _ProcessMemoryInfoProc().rss / 1024 ** 3
        #     print(a)
        #     time.sleep(0.1)
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
    # lp.print_stats()
    return

# ------------------------------------------------------------------------------
