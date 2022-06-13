# gPhoton 2

## overview
The gPhoton project aims to produce derived data products that are both quantitatively 
more correct and qualitatively more capable than 
original GALEX mission products, with a special emphasis on enabling high-precision 
short-time-domain science. 

gPhoton 2 is a substantial rewrite of the original gPhoton library intended to 
improve deployment flexibility and help support complex, full-catalog
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
git clone https://github.com/MillionConcepts/gphoton_working.git
cd gphoton_working
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
to make a gzip-compressed FITS image, a 30-second-bin lightcurve, and a table of 
exposure times for each bin of that lightcurve. It will write all of these
files to the `gPhoton/test_data` subdirectory.
 
Please refer to docstrings in `gPhoton/pipeline.py` for a complete explanation
of function arguments.

More comprehensive documentation for gPhoton 2 is forthcoming.

## caveats

gPhoton 2 is an active component of multiple ongoing research projects. In 
particular, continued development is a core component of the [GALEX Legacy 
Catalog (GLCAT) project.](https://www.millionconcepts.com/documents/glcat_adap_trimmed.pdf)
It is almost certain that it will receive meaningful new features and 
improvements, so interface stability is not guaranteed.

gPhoton 2 currently lacks comprehensive software tests and has been primarily 
(although not exclusively) deployed and tested in Ubuntu environments on AWS 
EC2 instances. It is possible that some features or components may be 
unstable on other platforms.

## feedback and contributions

We would love to receive bug reports. We are also happy to clarify points of 
confusion or consider feature requests. Please file GitHub Issues. We 
also invite contributions; if you are interested in adding work to gPhoton 2, 
please email the repository maintainers and let us know what you might be 
interested in fixing, tweaking, or implementing.

## citations
