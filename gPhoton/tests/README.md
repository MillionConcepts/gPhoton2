# gPhoton 2 tests

The suite has two layers:

- **Unit tests** (`test_units.py`, `test_units_.py`). These are fast and
  run by default. They cover the science-critical calculations: the
  gnomonic projection, exposure and dead time, frame binning and artifact
  flagging. They also cover the pipeline's contracts (output file names,
  catalog loading, eclipse gating, aspect table downloads) and the
  regression comparison code itself.
- **End-to-end regression tests** (`test_e2e_regression.py`). These are the
  main protection for the stability of scientific output. Each test runs
  `execute_pipeline` on a real eclipse and fingerprints every output
  product. It then compares the fingerprints against a committed baseline
  in `golden/`.

## running the tests

Install the test dependencies with `pip install -e ".[tests]"` (or use the
provided `environment.yml`), then run:

```
pytest gPhoton/tests              # unit tests only (e2e is deselected)
pytest gPhoton/tests -m e2e       # end-to-end regression tests
```

The end-to-end tests take about 2.5 minutes in total with 4 threads. They
need the raw6 files for e23456 (NUV and FUV) and e43337 (NUV). These are
cached in `<repo>/test_data/eNNNNN/`, or in the directory named by the
`GPHOTON_TEST_DATA` environment variable. Missing files are downloaded from
MAST on first use, and so are any missing aspect tables. Each test runs in
its own temporary directory, so the cache is never written to except by
these downloads.

A few unit tests download the small `metadata` aspect table on first use.
Everything else in the unit suite is network-free.

## end-to-end scenarios

Each scenario exercises distinct critical branches of the pipeline:

| test | eclipse | covers |
|---|---|---|
| E1 `test_e23456_nuv` | e23456 NUV, 120 s movie + image, RICE | raw6 decoding, NUV calibration, aspect, flagging, images / movies, WCS, exposure time, coverage backplane, source finding, aperture photometry, RICE FITS output |
| E2 `test_e23456_nuv_forced_photometry` | photometry-only rerun on E1's products | reading FITS products back, source catalog loading, forced photometry |
| E3 `test_e23456_fuv` | e23456 FUV, same settings as E1 | FUV-specific calibration (detector temperature, FUV offset, FUV cal files) |
| E4 `test_e43337_nuv_post_csp_multileg` | e43337 NUV, 12-leg AIS, images only, gzip | post-CSP calibration, leg splitting, per-leg boresight / coverage, gzip output |

Not yet covered by any scenario:

- `threads=None` (the unshared-memory and unthreaded code paths)
- `coregister_lightcurves` and fixed start times
- `extended_flagging` and `extended_photonlist`
- `aspect2`
- `compression="none"`, `burst`, `lil=False`
- multiple aperture sizes
- `coadd.py`

## fingerprints and tolerances

`fingerprint.py` reduces each product to a small JSON summary:

- **photonlists**: row count, column names, per-column null/NaN counts,
  mean/std/min/max (plus sum), flag value counts, and 16 evenly spaced
  rows recorded verbatim.
- **FITS images and movies**: per HDU, the array shape, dtype, sum, nonzero
  count and max, plus per-frame sums for movies. Also selected header keys:
  WCS, exposure times, time ranges and photon count.
- **photometry tables**: per-column NaN count, sum and median, and 16
  sources spaced evenly in declination, recorded verbatim.

The verbatim rows catch values that move between rows, such as flux
attached to the wrong source or photon times misaligned with positions.
Aggregate statistics cannot see these.

Every product is bit-for-bit reproducible. For RICE-compressed output, this
relies on the fixed `dither_seed` that the tests pass via
`hdu_constructor_kwargs`. The tolerances therefore only absorb
floating-point differences between platforms and library versions:

- by default, floats are compared with `rtol=1e-6` and `atol=1e-6`.
- absolute times (`t`, `t0_*`, `t1_*`, `T0_*`, `T1_*`) are compared to
  within 1e-4 s, and sky coordinates (`ra`, `dec`, `CRVAL*`) to within
  0.01 arcsec. These quantities carry large offsets, so a relative
  tolerance would be far too loose for them. For example, 1e-6 of a GALEX
  time is about 870 s. They are not summed, for the same reason.
- integers and strings must match exactly.

When a comparison fails, the differences are listed by pipeline stage,
earliest first: photonlist, image, movie, then photometry. The first
diverging stage usually points to the cause.

## updating the baselines

After an *intentional* change in output:

1. Run `pytest gPhoton/tests -m e2e --regen-golden` to rewrite `golden/`.
2. Review the diff in the golden JSON files and confirm that the changes
   are the ones you expect and are scientifically correct.
3. Run `pytest gPhoton/tests -m e2e` again. This confirms that the new
   baselines are reproducible.
4. Commit the golden files together with the change that caused them.

Never regenerate baselines to make an unexplained failure go away.

## legacy tests

`test_pipeline_regression.py` and `test_pipeline_regression_quick.py`,
along with `photometry_regression_stats.csv`, predate the current output
schema. They checked only the row and column counts of the photometry
tables, and they are permanently skipped. They are kept as a record of the
scenarios they covered: the DIS, AIS, CAI and GII survey types, and several
parameter combinations. They can be deleted once the e2e suite covers
those scenarios.
