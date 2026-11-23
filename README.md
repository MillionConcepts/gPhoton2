# gPhoton2

## overview

The gPhoton project aims to produce GALEX derived data products that are both
quantitatively more correct and qualitatively more capable than
original mission-produced products, with a special emphasis on enabling
high-precision short-time-domain science.

gPhoton2 is a substantial rewrite of [the original gPhoton library](https://github.com/cmillion/gphoton), intended
to improve deployment flexibility and help support complex, full-catalog
surveys on short timelines. It is 1-3 orders of magnitude faster than
gPhoton1 and features several significant calibration and quality of life improvements.

## installation

gPhoton 2 is not yet distributed via any package manager. Install it
from a clone of this repository. We recommend installing and running within a `conda` environment created from
the provided [environment.yml](environment.yml) file,
and installing the package into that environment with `pip`.
If you have `git` and `mamba` installed, this looks like:
```
git clone https://github.com/MillionConcepts/gPhoton2.git
cd gPhoton2
mamba env create -f environment.yml
mamba activate gphoton
pip install -e .
```

Some optional features require extra dependencies. See [optional dependencies](#dependencies)
below.

### aspect tables

gPhoton 2 relies on several large aspect metadata tables (~3.2 GB total)
that are not distributed along with this repository: `aspect.parquet`,
`boresight.parquet`, `leg-aperture.parquet`, and `metadata.parquet`. They
are hosted at [MAST](https://archive.stsci.edu/hlsps/gphoton/aspect_files/),
and gPhoton downloads them automatically the first time each one is needed. To skip
the download, place copies of these files in the aspect directory yourself.

By default, the aspect directory is `gPhoton/aspect/` inside the installed
package. With an editable install, that is the `gPhoton/aspect/` directory
of your clone. If you installed gPhoton 2 somewhere you can't or don't want
to write (such as a shared `site-packages`), pass `aspect_dir` to
`execute_pipeline()` and to the search functions to put the tables somewhere
else.

A small table of MAST download URLs for raw6 and scst files,
`raw_data_urls.parquet`, ships with the repository.

## basic usage

While many of the components of gPhoton2 can be used piecemeal, the
`gPhoton.pipeline.execute_pipeline` function is the primary interface to its
integrated processing pipelines. It can be called either within a script
or using CLI hooks. [run_pipeline.py](run_pipeline.py), in the root of
this repository, is an example script;

[pipeline_cli.py](pipeline_cli.py) is a CLI hook (and requires the
optional `fire` dependency). To use the CLI hook, pass arguments on the
command line, using `--` for keyword arguments. For instance, running:

`python pipeline_cli.py 23456 NUV --depth=30`

will call `execute_pipeline()` with positional arguments 23456
(eclipse number) and NUV (band), and the keyword argument depth=30. This will:

1. look up the raw GALEX telemetry (raw6) file for eclipse 23456 and download
   it from MAST, unless it is already present;
2. reduce the raw data to a calibrated 'photonlist' file;
3. use that photonlist to make a full-depth image and a 30-second-depth
   movie, both as gzip-compressed FITS files;
4. detect sources in the full-depth image, then perform aperture photometry
   at those positions (i.e. "forced photometry") in every movie frame. The result is one photometry
   table with a row per source: `aperture_sum_N`, `artifact_flag_N`,
   `expt_N`, `t0_N`, and `t1_N` columns for each frame `N`, plus full-depth
   `aperture_sum`, `artifact_flag`, and `exptime` columns.

By default, output files will be written in `test_data/e[NNNNN]/`, relative to the
directory you ran it from. (Where NNNNN is the eclipse number.) Pass `local_root` to write somewhere else.
Omitting `depth` produces only full-depth (i.e. visit-depth) image (and photometry on it).

To perform forced photometry on known positions---for example, from another catalog---instead of detected sources,
pass `source_catalog_file`: the path to a CSV or parquet file with these
columns:

| column    | units           | definition                               |
|-----------|-----------------|------------------------------------------|
| `eclipse` | integer         | GALEX orbital eclipse number             |
| `ra`      | decimal degrees | source right ascension (J2000)           |
| `dec`     | decimal degrees | source declination (J2000)               |

Any other columns will be ignored in the input source catalog file. If the file has no `eclipse` column, aperture photometry will be performed for all source positions within the requested eclipse. Duplicate
positions are dropped. The gPhoton2 photometry table is a valid source catalog input file.

Forced photometry (via input source catalog) causes the pipeline to skip source detection completely, so it cannot be combined with the automated extended source flagging feature
(`extended_flagging=True`), which relies on detection and spatial clustering of sources in each image.

Please refer to the docstring of `execute_pipeline()` in
[gPhoton/pipeline.py](gPhoton/pipeline.py) for a complete description of all
accepted arguments.

### finding eclipses

gPhoton 2 always requires an eclipse number to be specified. ("Eclipse" refers to the period of time in which the spacecraft was in Earth's shadow, and could safely observe. This happened once per orbit. Therefore "eclipse"~="orbit".)

To find eclipses that
cover a position, use the functions in `gPhoton.search.core`.
* The function
`galex_cone_search(ra, dec)` returns metadata for every eclipse whose
boresight passed within 0.625 degrees of that position. This is the approximate radius of the detector field-of-view, making this a good heuristic for whether a particular location on the sky was observed.
* `galex_sky_box(ra, dec, arcseconds)` matches a square box instead of a cone.
* `eclipses_near_object(name, arcseconds)` looks up a named object via SIMBAD, and then does a sky-box search. This requires the optional `astroquery` dependency.
* `filter_galex_eclipses()` selects eclipses by
predefined observation type (`"mislike"` or `"complex"`) or by pyarrow
`filters`.

For many worked examples of the main processing modes, see [COOKBOOK.md](COOKBOOK.md).

### tutorial

[The tutorial notebook](tutorial.ipynb) walks step-by-step through the process of finding eclipses for a given set of
sources, building an input source catalog, running the pipeline to generate images and photometry of those sources, and then plotting the
resulting images, movies, and lightcurves.

FIXME: The tutorial currently expects local copies
of the GFCAT catalog tables. Bundle these tables with the distro.

FIXME: The tutorial requires the optional dependencies [`pdr`](https://github.com/MillionConcepts/pdr) and
`imageio`. These need to be defined as optional dependencies in the environment.yml and pyproject.toml

### output files

Output files are named for the eclipse, band, depth, leg, and product type.
For instance, `python pipeline_cli.py 23456 NUV --depth=30` writes files
including:

* `e23456-nd-b00.parquet` -- photonlist for leg 0
* `e23456-nd-ffull-b00-image-g.fits.gz` -- full-depth image
* `e23456-nd-f0030-b00-movie-g.fits.gz` -- 30-second-depth movie
* `e23456-nd-f0030-b00-movie-photom-12_8.csv` -- photometry (with per-frame
  exposure times) using a 12.8-arcsecond aperture
* `e23456-nd-raw6.fits.gz` -- downloaded raw telemetry (a scratch directory,
  `test_data/temp/e23456`, is also created)

`nd`/`fd` denote NUV and FUV direct-mode data; `bNN` is the leg number within each eclipse. The
image/movie suffix reflects compression (`g` for gzip, `r` for RICE, `n` for
none); only gzip output ends in `.fits.gz`. Gzip compression shells out to
none); only gzip output ends in `.fits.gz`. Gzip compression shells out to `igzip`, `libdeflate_gzip`, or `gzip`, so one of them must be installed. Burst mode (`burst=True`) writes each movie frame as its own file, with `-tNNNN-` in
the name corresponding to frame number. Pass `ftype="parquet"` to write photometry tables as parquet instead
of CSV.

### supported eclipses

The pipeline will decline to process:

* eclipses above 47000, which are CAUSE-era data, collected after detector safety limits were relaxed, and frequently in a new "scan mode"; these are not yet supported;
* eclipses 37424 through 38149, which fall after the FUV detector's "Course Sun Point"
  event but before the TAC switch that (largely) mitigated serious data quality problems.

Passing `override_eclipse_limits=True` will force (attempted) processing of these eclipses; the pipeline will
most likely fail, though.

Note that an additional "ghost" flagging step is applied automatically to eclipses after 38149, which attempts (mostly successfully) to identify the "post-CSP ghost" artifacts that remained after the CSP event and TAC switch.

## requirements

### os
gPhoton 2 development to date has been focused on performance and stability in our
primary cloud deployment environments -- AWS EC2 instances running Ubuntu
Linux. As such, Linux is the preferred OS for gPhoton2. It is also known to be
compatible with MacOS, but multithreading may not be as performant; we suggest running gPhoton2 in single-threaded mode on MacOS. The software will probably run on Windows Subsystem for Linux (WSL), although it has not been extensively tested.

### resources
gPhoton 2 does not have hard-and-fast system requirements. It is highly
configurable and not all of the portions of its pipelines are equally
expensive. Also, the GALEX raw data archive is extremely diverse, and some
visits / eclipses are much cheaper to process than others. The resource needs scale basically linearly with the number of _photon events_ being simultaneously processed; therefore, processing a
brief visit on a sparse field could easily take less than 20% of the
resources as processing a long visit on a dense field.

However, here are some general notes:

* It should be possible to process any MIS-like eclipse to 30-second depth
  (bin size) in ~6 GB of free memory.
* gPhoton 2 processes each "leg" (nominal boresight position) of an eclipse
  separately, so eclipses with many "legs", like much of the All-Sky
  Imaging Survey (AIS) will often be very memory-cheap to process relative
  to overall visit time.
* We do not recommend executing the full gPhoton pipeline in an environment
  with less than ~8 GB of free memory, although some features of the pipeline
  may function well with considerably less, particularly on smaller raw data
  files.
* The most memory-expensive portion of the pipeline tends to be writing
  video files; if you are not writing video files to disk, you can generally
  get away with less memory.
* Decreasing the depth / increasing temporal
  resolution will tend to sharply increase the required memory for both
  producing and writing movies. If you only want to produce
  full-depth images with gPhoton 2 and neither want to write movies or generate
  photometry, time and memory requirements will tend to go sharply down.
* If you pass the `--burst=True` parameter, gPhoton 2 will write each
  frame of a movie as a separate file, which reduces memory pressure
  significantly, but is somewhat slower, and can produce a very large number of files.
* There is no particular minimum CPU requirement; gPhoton 2 will run on
  quite slow processors; but it can use all the processing power you can throw
  at it. Integrating video tends to be the most CPU-bound portion of the
  pipeline (especially at very high temporal resolution), followed by counting
  photometry on video frames, followed by producing photonlists.
* Running gPhoton 2 in multithreaded mode will tend to increase average
  (although not necessarily peak) working memory. Even in multithreaded mode,
  some portions of gPhoton 2 are thread-bound, and single-thread performance
  remains important.
* More than 8 parallel processes do not tend to be
  useful, except for high-resolution photometry on very dense fields.
* An x86_64 processor architecture is recommended but not required.
* If you are downloading raw6 files, a fast network connection will of course
  be useful. The first run on a fresh install also downloads the (~3.2GB) aspect
  tables, which will take as long as it takes.
* Some output files may be large enough that portions of the pipeline become
  I/O bound, and we do not recommend using a HDD or networked storage as
  working space for gPhoton 2.

### storage

Because of the diversity of the GALEX archive, the sizes of gPhoton's input
and output files vary greatly, but typical sizes for files associated with
NUV data from a MIS-like eclipse are:
* photonlist (as Snappy-compressed .parquet): 1 GB
* raw6 (as gzip-compressed FITS): 200 MB
* 30-second-depth movie (as gzip-compressed FITS): 300 MB
* full-depth image (as gzip-compressed FITS): 60 MB
* photometry table (as flat CSV): 4.5 MB

An FUV visit tends to have ~20% of the recorded events as its paired NUV visit, so FUV data files will tend to be much smaller.

Making different choices about temporal resolution and compression can
alter file size significantly. Compression is extremely efficient on these image data, and compression
ratio increases almost linearly with temporal resolution due to increasing
array sparsity.

The aspect tables described [above](#aspect-tables) need ~3.2 GB.

### dependencies

gPhoton 2 requires Python >= 3.11.

It also depends on the following Python libraries:
* astropy
* cytoolz
* dustgoggles
* fast-histogram
* fitsio
* matplotlib
* more-itertools
* numba
* numpy >= 2.2
* pandas
* photutils >= 3.0
* pyarrow
* quickbin (available from conda-forge)
* rasterio
* requests
* rich
* scikit-image >= 0.20
* scikit-learn
* scipy
* sh
* shapely

The following dependencies are optional (`pip` extra in parentheses):
* astroquery (`search`): SIMBAD object search
* fire (`cli`): pipeline CLI hook
* icc_rt (`x86_64`): performance improvements to numba; x86_64 only
* isal (`gzip`; `python-isal` on conda): faster gzip handling for some
  features
* matplotlib (`hulls`): extended-source hull plotting
* pandas-stubs, types-requests (`typecheck`): mypy type stubs
* pytest (`tests`): test suite
* pdr: tutorial notebook
* imagio: tutorial notebook

## tests

gPhoton 2's regression tests live in `gPhoton/tests/`. Unit tests run by default; end-to-end regression tests are deselected by default and can be run with `pytest gPhoton/tests -m e2e`; see
[gPhoton/tests/README.md](gPhoton/tests/README.md).

## caveats

gPhoton 2 is an active component of multiple ongoing research projects. In
particular, continued development is a core component of the [GALEX Legacy
Catalog (GLCAT) project.](https://www.millionconcepts.com/documents/glcat_adap_trimmed.pdf) The GLCAT project is nearing its conclusion, so the feature set and interface are somewhat stable. However, stability is not guaranteed. Whenever you use gPhoton2 for your research, it would be wise to note the specific git hash of the version of the software that you used, in order to guarantee reproducibility.

[FIXME] gPhoton 2 does not currently have comprehensive software tests, and its existing
regression tests are disabled. It has been primarily (although not
exclusively) deployed and tested in our preferred deployment environments.
Some features or components may be unstable in other environments.

## feedback and contributions

We are happy to receive bug reports or to clarify points of
confusion or consider feature requests. Please initiate a discussion through GitHub Issues before making a pull request.

## citations
If you use this software to support your research, please cite:
M. St. Clair, C. Million, R. Albach, S.W. Fleming (2022) _gPhoton2_. DOI:10.5281/zenodo.7472061

[![DOI](https://zenodo.org/badge/383023797.svg)](https://zenodo.org/badge/latestdoi/383023797)

## acknowledgements
This project has been supported by NASA Grants 80NSSC18K0084 and 80NSSC21K1421. Development of gPhoton1 was supported by the Mikulski Archive for Space Telescopes (MAST).