# gPhoton 2 tests

**Status: every test in this directory is currently disabled** with
`@pytest.mark.skip`. The benchmarks they check against predate recent
changes to the pipeline and its outputs, so the tests no longer pass as
written. They remain here as a record of the intended coverage and as a
starting point for regenerated benchmarks. Running `pytest` will report them
as skipped.

## what the tests cover

`test_pipeline_regression.py` is the thorough end-to-end test. It runs
`execute_pipeline` on several eclipses chosen to represent different survey
types (DIS e08330, AIS e18669, CAI e20334, GII e25125) with many combinations
of parameters, including movie depth, aperture sizes, compression, and forced
photometry from a source catalog. A final test checks the shapes (row and
column counts) of the resulting photometry tables against the benchmarks in
`photometry_regression_stats.csv`. That file is stale: it still lists
`*-exptime.csv` files, which the pipeline no longer writes, and photometry
shapes from the old output schema. This is a cursory
regression check: it does not compare photometric values. Expect a full run
to take about half an hour.

`test_pipeline_regression_quick.py` runs the pipeline once on the standard
test eclipse (e23456, NUV, 120-second depth) and checks the shape of the
resulting photometry table. Expect it to take about a minute.

Known issue: `test_pipeline_regression_06` passes a file path as
`extended_photonlist`, which is a boolean argument.

Fast, network-free unit tests live in `test_units.py` and run by default.

Both tests write their output to `regr_test_data/` relative to the directory
pytest runs from, and paths in `photometry_regression_stats.csv` assume
pytest runs from the repository root. On a fresh install, the first run will
also download raw6 files and the aspect tables from MAST.

## running the tests

Install the test dependencies with `pip install -e ".[tests]"` (or use the
provided `environment.yml`, which includes `pytest`), then run from the
repository root:

```
pytest gPhoton/tests/test_pipeline_regression_quick.py
```

While the tests are disabled, this command only reports them as skipped. To
run one locally, for instance while regenerating benchmarks, comment out the
`@pytest.mark.skip` decorator on that test.

## re-enabling the tests

To bring a test back into service:

1. Run its pipeline call on a known-good build.
2. Inspect the outputs and confirm they are scientifically correct.
3. Update `photometry_regression_stats.csv` (for the full suite) or the
   expected row/column counts in `test_pipeline_regression_quick.py` to match.
4. Remove the `@pytest.mark.skip` decorator.

Once re-enabled, run the quick test frequently during development and the
full suite before large merges.