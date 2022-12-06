"""
primary handler module for the 'full' gPhoton pipeline. can:
1. retrieve data from MAST or another specified location
2. create calibrated photonlists
3. create full-depth FITS images and movies of any number of specified depths
4. generate lightcurves and photonlists using detected or specified sources
These steps can also be performed separately using methods from the
gPhoton.io, gPhoton.photonpipe, gPhoton.moviemaker, and gPhoton.lightcurve
modules. This module is intended to perform them with optimized transitions
and endpoints / output suitable for remote automation.
"""
import re
from pathlib import Path
import shutil
from time import time
from types import MappingProxyType
from typing import Optional, Sequence, Mapping, Literal
import warnings

from cytoolz import identity, keyfilter
from gPhoton.reference import eclipse_to_paths, Stopwatch
from gPhoton.types import GalexBand, Pathlike
from more_itertools import chunked

# oh no! divide by zero! i am very distracting!
warnings.filterwarnings(action="ignore", category=RuntimeWarning)


def get_photonlist(
    eclipse: int,
    band: GalexBand,
    roots: dict[str],
    download: bool = True,
    recreate: bool = False,
    verbose: int = 1,
    threads: Optional[int] = None,
    raise_errors: bool = True,
    **_unused_options
):
    """
    Args:
        eclipse: GALEX eclipse number to process
        band: GALEX band to process: 'FUV' or 'NUV'?
        roots: dict with keys 'local', remote:
            Root of directory tree (on the local system, or at least
            somewhere with write permissions), to write and/or look for files
            'remote': 'Root of another directory tree (perhaps a mounted S3
            bucket or something) to check for input files.
            Unlike local root, there is no assumption that write access is
            available to these paths.
        download: If raw6 (L0 telemetry) files cannot be found in the expected
            paths from local_root or remote_root, shall we try to download
            them from MAST?
        recreate: If a photonlist already exists at the expected path from
            local_root or remote_root, shall we recreate it (overwriting it if
            it's in local_root)?
        verbose: How chatty shall photonpipe be, 0-2? Higher is chattier. Full
            documentation forthcoming.
        threads: how many threads shall the multithreaded portions of
          photonpipe use? Passing None turns multithreading off entirely.
        raise_errors: if we encounter errors during pipeline execution,
            shall we raise an exception (True) or return structured text
            describing that error and its context (False)? Typically this
            should be True for ad-hoc local execution (and especially for
            debugging), and False for managed remote execution.
    """
    # TODO, maybe: parameter to run only certain legs
    files, photonpaths, temp_directory = _set_up_paths(eclipse, band, roots)
    if (roots.get("remote") is not None) and (recreate is False):
        if all([Path(file).exists() for file in roots["remote"]["photonfiles"]]):
            print(
                f"making temp local copies of photon file(s) from remote: "
                f"{roots['remote']['photonfiles']}"
            )
            photonpaths = []
            for file in roots['remote']['photonfiles']:
                photonpaths.append(
                    Path(shutil.copy(Path(file), temp_directory))
                )
    if recreate or not all([path.exists() for path in photonpaths]):
        if not photonpaths[0].parent.exists():
            photonpaths[0].parent.mkdir(parents=True)
        raw6path = _look_for_raw6(
            eclipse, band, download, files, temp_directory
        )
        if not raw6path.exists():
            print("couldn't find raw6 file.")
            if raise_errors is True:
                raise OSError("couldn't find raw6 file.")
            return "return code: couldn't find raw6 file."
        from gPhoton.photonpipe import execute_photonpipe

        try:
            execute_photonpipe(
                photonpaths[0].parent,
                band,
                raw6file=str(raw6path),
                verbose=verbose,
                chunksz=1000000,
                threads=threads,
            )
        except ValueError as value_error:
            if raise_errors:
                raise value_error
            return parse_photonpipe_error(value_error)
    else:
        print(
            f"using existing photon list(s): "
            f"{[str(path) for path in photonpaths]}"
        )
    return photonpaths


def execute_photometry_only(
    eclipse,
    band,
    roots,
    depth,
    compression,
    lil,
    aperture_sizes,
    source_catalog_file,
    threads,
    stopwatch,
    **_unused_options
):
    if stopwatch is None:
        stopwatch = Stopwatch()
        stopwatch.start()
    from gPhoton.aspect import aspect_tables

    metadata = aspect_tables(eclipse, ("metadata",))[0]
    legs = metadata["legs"][0].as_py()
    legs = [0] if legs == 0 else range(legs)
    errors = []
    for leg in legs:
        loaded_results = load_moviemaker_results(
            eclipse, band,  leg, roots, depth,  compression, lil
        )
        # this is an error code
        if isinstance(loaded_results, str):
            errors.append(loaded_results)
            continue
        output_filenames, results = loaded_results
        from gPhoton.lightcurve import make_lightcurves

        result = make_lightcurves(
            results,
            output_filenames,
            eclipse,
            band,
            leg,
            aperture_sizes,
            source_catalog_file,
            threads,
            stopwatch,
        )
        if result != "successful":
            errors.append(result)
    print(
        f"{round(time() - stopwatch.start_time, 2)} seconds for execution"
    )
    if len(errors) > 0:
        return "return code: " + ";".join(errors)
    return "return code: successful"


# TODO, maybe: add a check somewhere for: do we have information regarding
#  this eclipse at all?
def execute_pipeline(
    eclipse: int,
    band: GalexBand,
    depth: Optional[int] = None,
    threads: Optional[int] = None,
    local_root: str = "test_data",
    remote_root: Optional[str] = None,
    download: bool = True,
    recreate: bool = False,
    verbose: int = 2,
    maxsize: Optional[int] = None,
    source_catalog_file: Optional[str] = None,
    write: Mapping = MappingProxyType({"image": True, "movie": True}),
    aperture_sizes: Sequence[float] = tuple([12.8]),
    lil: bool = True,
    coregister_lightcurves: bool = False,
    stop_after: Optional[Literal["photonpipe", "moviemaker"]] = None,
    compression: Literal["none", "gzip", "rice"] = "gzip",
    hdu_constructor_kwargs: Optional[Mapping] = None,
    min_exptime: Optional[float] = None,
    photometry_only: bool = False,
) -> str:
    """
    Args:
        eclipse: GALEX eclipse number to process
        band: GALEX band to process: 'FUV' or 'NUV'?
        depth: how many seconds of events to use when integrating each
            movie frame. in a sense: inverse FPS. None means "make a
            full-depth image only".
        threads: how many threads to use for parallel processing. Passing None
            turns off parallel processing entirely. Passing 1 will process in a
            single parallel thread (not recommended except for test
            purposes). Increasing thread count increases execution speed but
            also increases memory pressure, particularly for movie creation.
            Multithreading currently works best on Linux and is not
            guaranteed to be at all performant on other operating systems.
        local_root: Root of directory tree (on the local system, or at least
            somewhere with write permissions), to write and/or look for files
        remote_root: Root of another directory tree (perhaps a mounted S3
            bucket or something) to check for preexisting raw6 and
            photonlist files. Unlike local_root, there is no assumption that
            write access is available to these paths.
        download: if raw6 files aren't available, should we download them
            from MAST, or quit?
        recreate: if photonlist file is already present, should we recreate
            (and possibly overwrite) it?
        verbose: how many messages do you want to see? 0 turns almost all
            output off; levels up to 4 are meaningful.
        maxsize: maximum working memory size in bytes. None deactivates
            estimated size checking. if estimated memory cost of generating
            image or movie frames exceeds this threshold, the pipeline will
            stop. Note that estimates are not 100% reliable!
        source_catalog_file: by default, the pipeline performs photometry on
            automatically-detected sources. passing the path to a CSV file as
            source_catalog_file specifies positions, preempting automated
            source detection.
        write: save images and/or movies to disk, or discard them after using
            them for photometry?
        aperture_sizes: what aperture size(s) (in arcseconds) should we
            use to compute photometry. passing multiple sizes may be useful
            for background estimation or related processes.
        lil: should we use matrix sparsification techniques whe movies?
            introduces some CPU overhead but can significantly reduce memory
            usage, especially for large numbers of frames, and especially
            during the frame integration and photometry steps.
        coregister_lightcurves: should we pin the start time of the first
            movie frame / lightcurve bin to the start time of the other
            band's first movie frame / lightcurve bin (if it exists)?
        stop_after: should we bail out after a particular phase? options are
            "photonpipe" (make photonlist only), "moviemaker" (make and write
            images and movies but don't perform photometry on them)
        compression: what sort of compression should we apply to movies and
            images? "gzip" is monolithic gzip; "rice" is RICE_1 (for the
            cntmap, lossy) tile compression; "none" is no compression at all.
        hdu_constructor_kwargs: optional mapping of kwargs to pass to the
            astropy.io.fits HDU constructor (for instance, tile compression
            parameters)
        min_exptime: minimum effective exposure time to run image/movie
            and lightcurve generation. None means no lower bound.
        photometry_only: attempt to perform photometry on already-existing
            images/movies, doing nothing else

    Returns:
        str: `"return code: successful"` for fully successful execution;
            `"return code: {other_thing}"` for various known failure states
            (many of which produce a subset of valid output products)
    """
    # set up parameter structure from input options that are never changed by
    # the pipeline. this structure is passed to child functions for brevity.
    # TODO: bring filter kwargs wrapper in to replace variadic sigs in children
    opt = {
        'eclipse': eclipse,
        'band': band,
        'depth': depth,
        'threads': threads,
        'download': download,
        'recreate': recreate,
        'verbose': verbose,
        'roots': {'local': local_root, 'remote': remote_root},
        'compression': compression,
        'lil': lil,
        'aperture_sizes': aperture_sizes,
        'source_catalog_file': source_catalog_file,
        'min_exptime': min_exptime,
        'maxsize': maxsize,
        'stopwatch': Stopwatch(),
        'hdu_constructor_kwargs': hdu_constructor_kwargs,
        'write': write
    }

    # SETUP AND FILE RETRIEVAL
    opt['stopwatch'].click()
    if eclipse > 47000:
        print("CAUSE data w/eclipse>47000 are not yet supported.")
        return "return code: CAUSE data w/eclipse>47000 are not yet supported."
    if verbose > 1:
        from gPhoton.aspect import aspect_tables

        metadata = aspect_tables(eclipse, ("metadata",))[0]
        print(
            f"eclipse {eclipse} {band}  -- {metadata['obstype'][0].as_py()}; "
            f"{metadata['legs'][0].as_py()} leg(s)"
        )
    if photometry_only is True:
        return execute_photometry_only(**opt)

    # fetch, locate, or create photonlist, as requested
    get_photonlist_result = get_photonlist(**opt, raise_errors=False)
    # we received an error return code
    if isinstance(get_photonlist_result, str):
        return get_photonlist_result  # this is an error return code
    else:
        photonpaths = get_photonlist_result  # strictly explanatory renaming
    if stop_after == "photonpipe":
        print(
            f"stop_after='photonpipe' passed, halting; "
            f"{round(time() - opt['stopwatch'].start_time, 2)} "
            f"seconds for execution"
        )
        return "return code: successful (planned stop after photonpipe)"
    files, _, _ = _set_up_paths(**opt)
    opt['stopwatch'].click()
    from gPhoton.parquet_utils import get_parquet_stats

    ok_paths = []
    for path in photonpaths:
        file_stats = get_parquet_stats(str(path), ["flags", "ra"])
        if (file_stats["flags"]["min"] > 6) or (file_stats["ra"]["max"] is None):
            print(f"no unflagged data in {path}, not processing")
        ok_paths.append(path)
    if len(ok_paths) == 0:
        print("no usable legs, bailing out.")
        return "return code: no unflagged data (stopped after photon list)"
    # MOVIE-RENDERING SECTION
    from gPhoton.moviemaker import (
        create_images_and_movies,
        write_moviemaker_results,
    )

    errors = []
    for leg, path in enumerate(photonpaths):
        # check to see if we're pinning our frame / lightcurve time series to
        # the time series of existing analysis for the other band
        fixed_start_time = check_fixed_start_time(
            eclipse,
            depth,
            local_root,
            remote_root,
            band,
            coregister_lightcurves,
            leg
        )
        if len(photonpaths) > 0:
            print(f"processing leg {leg + 1} of {len(photonpaths)}")
        results = create_images_and_movies(
            path, fixed_start_time=fixed_start_time, **opt
        )
        opt['stopwatch'].click()
        if not (results["status"].startswith("successful")):
            message = (
                f"Moviemaker pipeline unsuccessful on leg {leg} "
                f"{(results['status'])}"
            )
            print(message)
            errors.append(message)
            continue
        if not stop_after == "moviemaker":
            from gPhoton.lightcurve import make_lightcurves

            photometry_result = make_lightcurves(
                results, files['local'], leg=leg, **opt
            )
        else:
            photometry_result = 'successful'
        write_result = write_moviemaker_results(
            results, files['local'], leg=leg, **opt
        )
        if photometry_result != 'successful':
            errors.append(photometry_result)
        if write_result != 'successful':
            errors.append(write_result)
    print(
        f"{round(time() - opt['stopwatch'].start_time, 2)} "
        f"seconds for execution"
    )
    if len(errors) > 0:
        return "return code: " + ";".join(errors)
    return "return code: successful"


def _look_for_raw6(
    eclipse: int,
    band: GalexBand,
    download: bool,
    files,
    temp_directory: Path,
) -> Path:
    """
    :param eclipse: GALEX eclipse number to run
    :param band: "NUV" or "FUV" -- near or far ultraviolet
    :param files: mapping of paths to local and remote_files generated in
    _set_up_paths(); 'remote' key may be absent if there is no remote root
    :param temp_directory: path to scratch directory for temp copy of raw6
    from remote_root
    :return: tuple of primary filename dict, path to photon list we'll be
    using, remote filename dict, name of temp/scratch directory
    :return: path to raw6 file
    """

    raw6path = Path(files['local']["raw6"])
    if not raw6path.exists() and (files.get('remote') is not None):
        if Path(files['remote']["raw6"]).exists():
            print(
                f"making temp local copy of raw6 file from remote: "
                f"{files['remote']['raw6']}"
            )
            raw6path = Path(
                shutil.copy(files['remote']["raw6"], temp_directory)
            )
    if not raw6path.exists() and (download is True):
        from gPhoton.io.mast import retrieve_raw6

        print("downloading raw6file")
        raw6file = retrieve_raw6(eclipse, band, raw6path)
        if raw6file is not None:
            raw6path = Path(raw6file)
    return raw6path


def load_moviemaker_results(
    eclipse, band, leg, roots, depth, compression, lil=True
):
    files, _, temp_directory = _set_up_paths(
        eclipse, band, roots, depth, compression
    )
    image = pick_and_copy_array(files, temp_directory, "images", leg)
    if image is None:
        print("Photometry-only run, but image not found. Skipping.")
        return f"leg {leg}: image not found"
    if depth is not None:
        movie = pick_and_copy_array(files, temp_directory, "movies", leg)
        if movie is None:
            print("Photometry-only run, but movie not found. Skipping.")
            return f"leg {leg}: movie not found"
    image_result = unpack_image(image, compression)
    results = {'wcs': image_result['wcs'], 'image_dict': image_result}
    if depth is not None:
        # noinspection PyUnboundLocalVariable
        results |= {'movie_dict': unpack_movie(movie, compression, lil)}
    else:
        results['movie_dict'] = {}
    return files["local"], results


def pick_and_copy_array(files, temp_directory, which="images", leg=0):
    if Path(files["local"][which][leg]).exists():
        return files["local"][which][leg]
    if files.get("remote") is None:
        return None
    if not Path(files["remote"][which][leg]).exists():
        return None
    print(f"making temp copy of {which} from remote")
    shutil.copy(files["remote"][which][leg], temp_directory)
    return Path(temp_directory, Path(files["local"][which][leg]).name)


def _set_up_paths(
    eclipse: int,
    band: GalexBand,
    roots: dict[str],
    depth: Optional[int] = None,
    compression: Literal["none", "rice", "gzip"] = "gzip",
    **_unused_options
) -> tuple[dict, list[Path], Path]:
    """
    initial path setup & file retrieval step for execute_pipeline.
    :param eclipse: GALEX eclipse number to run
    :param band: "NUV" or "FUV" -- near or far ultraviolet
    :param depth: how many seconds of events to use when integrating each
    movie frame. in a sense: inverse FPS.
    :param roots: 'local': path to write pipeline results to, 'remote':
    additional path to check for preexisting raw6 and photonlist files
    (intended primarily for multiple servers referencing a shared set
    of remote resources)
    :param compression: planned type of compression for movie/image files

    :return: tuple of: primary filename dict, path to photon list we'll be
    using, remote filename dict, name of temp/scratch directory
    """
    local_files = eclipse_to_paths(
        eclipse, roots.get('local'), depth, compression
    )[band]
    eclipse_dir = Path(list(local_files.values())[0]).parent
    if not eclipse_dir.exists():
        eclipse_dir.mkdir(parents=True)
    if roots.get('remote') is not None:
        # we're only ever looking for raw6 & photonlist, don't need depth etc.
        remote_files = eclipse_to_paths(
            eclipse, roots.get('remote'), depth, compression
        )[band]
    else:
        remote_files = None
    temp_directory = Path(
        roots.get('local', 'data'), "temp", str(eclipse).zfill(5)
    )
    if not temp_directory.exists():
        temp_directory.mkdir(parents=True)
    photonpaths = [Path(file) for file in local_files["photonfiles"]]
    files = {'local': local_files, 'remote': remote_files}
    return files, photonpaths, temp_directory


def parse_photonpipe_error(value_error: ValueError) -> str:
    if str(value_error).startswith("bad distortion correction"):
        print(str(value_error))
        return "return code: bad distortion correction solution"
    if "probably not a valid FUV observation" in str(value_error):
        print(str(value_error))
        return "return code: not a valid FUV observation"
    if "FUV temperature out of range" in str(value_error):
        print(str(value_error))
        return "return code: FUV temperature value out of range"
    raise value_error


def check_fixed_start_time(
    eclipse,
    depth,
    local: Pathlike,
    remote: Optional[Pathlike],
    band: GalexBand,
    coregister: bool,
    leg: int
) -> Optional[str]:
    if (coregister is not True) or (depth is None):
        return None
    other = "NUV" if band == "FUV" else "FUV"
    expfile = None
    for location in (remote, local):
        if location is None:
            continue
        # TODO: must fix this
        try:
            exp_fn = eclipse_to_paths(
                eclipse, location, depth
            )[other]["expfiles"][leg]
        except IndexError:
            print(
                f"Disturbingly, the cross-band eclipse does not appear to "
                f"possess an equivalent leg."
            )
            break
        if Path(exp_fn).exists():
            expfile = exp_fn
            break
    if expfile is None:
        print(
            f"Cross-band frame coregistration requested, but exposure "
            f"time table at this depth for {other} was not found."
        )
        return None
    import pandas as pd

    print(f"pinning first bin to first bin from {expfile}")
    # these files are small enough that we do not need to bother scratching
    # them to disk, even from a remote / fake filesystem
    coreg_exptime = pd.read_csv(expfile)
    return coreg_exptime["t0"].iloc[0]


def load_array_file(array_file, compression):
    import astropy.wcs
    import fitsio
    from gPhoton.io.fits_utils import AgnosticHDUL, pyfits_open_igzip
    if compression == 'gzip':
        hdul = AgnosticHDUL(pyfits_open_igzip(array_file))
    else:
        hdul = AgnosticHDUL(fitsio.FITS(array_file))
    offset = 0 if compression != "rice" else 1
    cnt_hdu, flag_hdu, edge_hdu = (hdul[i + offset] for i in range(3))
    headerdict = dict(cnt_hdu.header)
    tranges = keyfilter(lambda k: re.match(r"T[01]", k), headerdict)
    tranges = tuple(chunked(tranges.values(), 2))
    exptimes = tuple(
        keyfilter(lambda k: re.match(r"EXPT_", k), headerdict).values()
    )
    wcs = astropy.wcs.WCS(cnt_hdu.header)
    results = {"exptimes": exptimes, "tranges": tranges, "wcs": wcs}
    return (cnt_hdu, edge_hdu, flag_hdu), results


def unpack_movie(movie_file, compression, lil):
    hdus, results = load_array_file(movie_file, compression)
    planes = ([], [], [])
    if lil is True:
        import scipy.sparse

        # noinspection PyUnresolvedReferences
        constructor = scipy.sparse.coo_matrix
    else:
        constructor = identity
    for hdu, plane in zip(hdus, planes):
        for frame_ix in range(len(results['exptimes'])):
            plane.append(constructor(hdu[frame_ix, :, :][0]))
    return results | {"cnt": planes[0], "flag": planes[1], "edge": planes[2]}


def unpack_image(image_file, compression):
    hdus, results = load_array_file(image_file, compression)
    planes = {
        "cnt": hdus[0].data, "flag": hdus[1].data, "edge": hdus[2].data
    }
    return results | planes
