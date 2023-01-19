# gPhoton 2

## overview

The gPhoton project aims to produce derived data products that are both 
quantitatively more correct and qualitatively more capable than 
original GALEX mission products, with a special emphasis on enabling 
high-precision short-time-domain science. 

gPhoton 2 is a substantial rewrite of the original gPhoton library intended 
to improve deployment flexibility and help support complex, full-catalog
surveys on short timelines. It is 1-3 orders of magnitude faster than
original gPhoton and features several significant calibration improvements. 
It is currently in early beta.

## installation

gPhoton 2 is not currently distributed via any package manager. The easiest 
way to install it is simply to clone this repository. We recommend installing 
its requirements by feeding the provided [environment.yml](environment.yml) 
file to `conda` or `mamba`. If you have `git` and `mamba` 
installed, this might be as easy as (modulo environment differences):
```
git clone https://github.com/MillionConcepts/gPhoton2.git
cd gphoton2
mamba env create -f environment.yml
```

gPhoton 2 also relies on several large aspect metadata files that are not 
distributed along with this repository. 
[They are currently available here.](https://drive.google.com/drive/u/1/folders/1aPfLKsZM8x5Pxji0Lh3dUblpo9dyt1IW)
Place these files in the `gPhoton/aspect/` directory after downloading
them.

## basic usage

While many of the components of gPhoton 2 can be used piecemeal, the 
`gPhoton.pipeline.execute_pipeline` function is the primary interface to its
integrated processing pipelines. It can be called either within a script
or using simple CLI hooks. `gPhoton/run_pipeline.py` is a very simple 
example of calling it from a script; `gPhoton/pipeline_cli.py` is a simple
CLI hook. To use the CLI hook, pass arguments on the command line, using `--`
for keyword arguments. For instance, running:

`python pipeline_cli.py 23456 NUV --depth=30`

will call `execute_pipeline()` with positional arguments 23456 
(eclipse number) and NUV (band), and the keyword argument depth=30. This will
fetch raw GALEX telemetry (if it's not in your path already) over HTTP from 
[MAST](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html)
for eclipse 23456, reduce it to a 'photonlist' file, then use that photonlist 
to make a gzip-compressed FITS image, 30-second-bin lightcurves (based on 
automatically-detected source positions), and a table of 
exposure times for each bin of that lightcurve. It will write all of these
files to the `gPhoton/test_data` subdirectory.
 
Please refer to docstrings in `gPhoton/pipeline.py` for a complete description
of all accepted arguments to this function.

More comprehensive documentation for gPhoton 2 is forthcoming.

## requirements

### os
gPhoton 2 development to date has been focused on performance and stability in our 
primary cloud deployment environments -- AWS EC2 instances running Ubuntu
Linux. As such, Linux is the preferred OS for gPhoton 2, although it is also
compatible with MacOS. Multithreading is poorly performant on MacOS. Please 
run gPhoton 2 in single-threaded mode if you are using it on MacOS. gPhoton 2
should also run on Windows Subsystem for Linux, although it has not been 
extensively tested.

### resources
gPhoton 2 does not have hard-and-fast system requirements. It is highly 
configurable and not all of the portions of its pipelines are equally 
expensive. Also, the GALEX raw data archive is extremely diverse, and some
visits / eclipses are much cheaper to process than others -- processing a 
brief visit on a sparse field could easily take less than 20% as many 
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
  significantly but is somewhat slower.
* There is no particular minimum CPU requirement; gPhoton 2 will run on
  quite slow processors, but can use all the processing power you can throw
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
    be useful. 
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
* lightcurves (as flat CSV): 4.5 MB
* exposure time table (as flat CSV): 3 KB

FUV data will tend to be significantly smaller due to the relatively lower
density of FUV events in almost all GALEX visits.

Making different choices about temporal resolution and compression can
alter these figures significantly. These files tend to be much larger 
uncompressed, particularly high-temporal-resolution videos -- compression
ratio increases almost linearly with temporal resolution due to increasing
array sparsity.

### dependencies

gPhoton 2 requires Python 3.9 or 3.10. 3.10 is recommended. (3.11 support 
pends upstream changes in `numba` and is anticipated sometime in Q1 2023.)

It also depends on the following Python libraries:
* astropy
* dustgoggles
* fast-histogram
* fitsio
* more-itertools
* numba
* numpy
* pandas
* photutils
* pip
* pyarrow
* requests
* rich
* scipy
* sh

The following dependencies are optional:
* astroquery (sky object search functions)
* fire (pipeline CLI)
* icc_rt (performance improvements to numba)
* igzip (some gzip-handling features)

## caveats

gPhoton 2 is an active component of multiple ongoing research projects. In 
particular, continued development is a core component of the [GALEX Legacy 
Catalog (GLCAT) project.](https://www.millionconcepts.com/documents/glcat_adap_trimmed.pdf)
It is almost certain that it will receive meaningful new features and 
improvements, so interface stability is not guaranteed.

gPhoton 2 currently lacks comprehensive software tests and has been primarily 
(although not exclusively) deployed and tested in our preferred deployment
environments. Some features or components may be unstable in other 
environments.

## feedback and contributions

We would love to receive bug reports. We are also happy to clarify points of 
confusion or consider feature requests. Please file GitHub Issues. We 
also invite contributions; if you are interested in adding work to gPhoton 2, 
please email the repository maintainers and let us know what you might be 
interested in fixing, tweaking, or implementing.

## citations
If you use this software to support your research, please cite:
M. St. Clair, C. Million, R. Albach, S.W. Fleming (2022) _gPhoton2_. DOI:10.5281/zenodo.7472061

[![DOI](https://zenodo.org/badge/383023797.svg)](https://zenodo.org/badge/latestdoi/383023797)

## acknowledgements
This project has been supported by NASA Grants 80NSSC18K0084 and 80NSSC21K1421.
