# gPhoton 2 cookbook

Short recipes for the main processing modes. See the [README](README.md) for
installation and the docstring of `execute_pipeline()` in
[gPhoton/pipeline.py](gPhoton/pipeline.py) for every argument. All recipes
assume you are running in the `gphoton` environment, from a directory where it
is fine to write `test_data/`. Output file names are examples for eclipse
23456 (NUV, leg 0).

Note: the first run downloads raw telemetry from MAST and, if needed, ~3.2 GB
of aspect tables.

## 1. Run one eclipse and read the photometry as a lightcurve

```python
from gPhoton.pipeline import execute_pipeline

execute_pipeline(23456, "NUV", depth=30)
```

Writes `e23456-nd-b00.parquet` (photonlist), `e23456-nd-ffull-b00-image-g.fits.gz`,
`e23456-nd-f0030-b00-movie-g.fits.gz`, and
`e23456-nd-f0030-b00-movie-photom-12_8.csv` in `test_data/e23456/`.
Omit `depth` for a full-depth image and photometry only.

Sources are detected once, on the full-depth image; photometry is then
measured at those positions in every frame. The photometry table has one row
per source. For frame `N` it has `aperture_sum_N` (counts), `expt_N` (exposure
seconds), `t0_N` and `t1_N` (frame start and stop times), and `artifact_flag_N`.
The unsuffixed `aperture_sum`, `exptime`, and `artifact_flag` columns are for
the full-depth image.

Restructure the table to be more conveniently viewed as a lightcurve with
something like the following.

```python
import pandas as pd
from math import sqrt

phot = pd.read_csv("test_data/e23456/e23456-nd-f0030-b00-movie-photom-12_8.csv")
n = [c.removeprefix("aperture_sum_") for c in phot if c.startswith("aperture_sum_")]
src = phot.iloc[0] # the first source in the catalog
lightcurve = pd.DataFrame(
    {
        "t0": [src[f"t0_{i}"] for i in n],
        "cps": [src[f"aperture_sum_{i}"] / src[f"expt_{i}"] for i in n],
        "cps_err": [
          sqrt(src[f"aperture_sum_{i}"]) / src[f"expt_{i}"] for i in n
        ],
        "flag": [int(src[f"artifact_flag_{i}"]) for i in n],
    }
)
```

(Frames with zero exposure time give inf or NaN count rates.)

## 2. Photometry at known positions

Make a CSV or parquet file with `ra` and `dec` (decimal degrees) and, if it
covers several eclipses, an `eclipse` column. Without an `eclipse` column all
rows are used. Duplicates are dropped.

```python
pd.DataFrame({"eclipse": [23456], "ra": 323.93, "dec": -2.165}).to_csv("cat.csv", index=False) # write a valid catalog file
execute_pipeline(23456, "NUV", depth=30, source_catalog_file="cat.csv")
```

## 3. Several apertures
Generate separate measurements for a set of aperture radii.

```python
execute_pipeline(23456, "NUV", depth=30, aperture_sizes=[6.0, 12.8, 25.6])
```

Radii are in arcseconds. Each produces its own photometry file:
`...-movie-photom-6_0.csv`, `...-photom-12_8.csv`, `...-photom-25_6.csv`.

## 4. Frame depth, outputs, and NUV/FUV coregistration
Generate 30-second photometry but only store the full-depth image (not the movie).
```python
# image only, no movie file kept
execute_pipeline(23456, "NUV", depth=30, write={"image": True, "movie": False})
```

Force the time bins for NUV and FUV to perfectly align, making cross-band comparisons much easier. (Note: Always run the NUV first.)

```python
execute_pipeline(23456, "NUV", depth=30, coregister_lightcurves=True)
execute_pipeline(23456, "FUV", depth=30, coregister_lightcurves=True)
```

## 5. Find eclipses

```python
from gPhoton.search.core import (
    galex_cone_search, galex_sky_box, eclipses_near_object, filter_galex_eclipses
)

galex_cone_search(323.93, -2.165)          # boresight within 2250" (0.625 deg) of [ra, dec]
galex_sky_box(323.93, -2.165, 1968.75)     # box around boresight positions
eclipses_near_object("M 81", 1968.75)      # SIMBAD lookup; requires astroquery to be installed
filter_galex_eclipses("mislike")           # or "complex"; or filters=[...]
```

Each returns metadata, including eclipse numbers, that you can loop over.

## 6. Burst mode and compression

Running in burst mode generates one image per movie frame, rather than a single movie (cube) file. Warning: This can create hundreds or thousands of files if the depth is small!

```python
execute_pipeline(23456, "NUV", depth=120, burst=True, compression="rice")
```

`burst=True` writes every movie frame as its own FITS file (`-tNNNN-` in the
name) instead of one big movie. `compression` is `"gzip"` (default; needs a
`gzip`/`igzip`/`libdeflate_gzip` binary, output `.fits.gz`), `"rice"`
(tile-compressed, `.fits`), or `"none"`.

## 7. Extended sources and photon information

```python
execute_pipeline(23456, "NUV", depth=30, extended_flagging=True)
execute_pipeline(23456, "NUV", depth=30, extended_photonlist=True)
```

- `extended_flagging` runs a DBSCAN-based extended-source finding heuristic to determine whether apertures lie within an extended / crowded source. It writes an
  `...-extended-sources.csv` and fills an `extended_source` column in the photometry file. This cannot be used with a catalog input file.
- `extended_photonlist` keeps extra photon columns (q, ya, row, col) and adds
  photometry columns on images made from them (`dose_col`, `dose_row`,
  `ya_aperture_sum`, `stdcolrow_aperture_sum`, `q_aperture_sum`). Those
  images are not written to disk.

## 8. Partial and repeat runs

```python
execute_pipeline(23456, "NUV", depth=30, stop_after="photonpipe")   # produce photonlist only
execute_pipeline(23456, "NUV", depth=30, stop_after="moviemaker")   # + images/movies
execute_pipeline(23456, "NUV", depth=30, photometry_only=True,
                 source_catalog_file="cat.csv") # reuse/reread existing images; requires isal package; cat.csv created above
execute_pipeline(23456, "NUV", depth=30, recreate=True, suffix="v2") # redo with new filenames
```

## 9. Flags

Per-pixel flag backplane in image/movie FITS files (`FLAG` extension):

| bit | meaning                          |
|-----|----------------------------------|
| 1   | mission hotspot mask             |
| 2   | post-CSP ghost                   |
| 4   | wide edge                        |
| 8   | narrow edge                      |

The photometry table's `artifact_flag` columns hold the same four bits,
set when any flagged pixel falls inside the aperture:

```python
good = phot[phot["artifact_flag"] == 0]
no_edge = phot[(phot["artifact_flag"] & (4 | 8)) == 0]
```

FITS files also have `COUNT` and `COVERAGE` extensions (images additionally
`DOSE`).

## 10. Coadds

```python
from gPhoton.coadd import get_image_fns, coadd_image_files

execute_pipeline(23456, "NUV")
execute_pipeline(23453, "NUV") # known to overlap e23456 from the cone search above

files = get_image_fns(23456, 23453, band="NUV", root="test_data") # construct standard image filepaths
coadd = coadd_image_files(files)   # counts summed on a shared WCS
```

Note that if the individual images don't overlap, it will still produce a coadd. These might have absurd dimensions!

## 11. Convert to and from magnitudes or fluxes

```python
from gPhoton.analysis.utils import counts2mag, mag2counts

counts2mag(cps=1.5, band="NUV")    # AB magnitude
mag2counts(20.0, "NUV") # convert from mag to cps
```

Substitute `flux` for `mag` if flux units are needed.

## 12. Command line

Install for command-line access.

```
pip install -e ".[cli]"
python pipeline_cli.py 23456 NUV --depth=30 --aperture_sizes="[6.0,12.8]"
```

Every `execute_pipeline()` argument becomes a CLI flag.
