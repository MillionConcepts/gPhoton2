import pytest
import csv
import pandas as pd

# I just want the dumbest possible regression tests against refactoring.
class TestExecutionRegression():
    from gPhoton.pipeline import execute_pipeline
    execute_pipeline(
        35688,
        "NUV",
        depth=120,
        threads=4,
        local_root="test_data",
        recreate=True,
        aperture_sizes=[12.8],
        write={"movie": True, "image": True},
        coregister_lightcurves=True,
        photometry_only=False,
        compression="rice",
        lil=True,
    )

    def test_source_count(self):
        # Check that the number of extracted sources remains fixed
        sourcelist = pd.read_csv('test_data/e35688/e35688-nd-f0120-b00-movie-photom-12_8.csv')
        assert len(sourcelist) == 3934

    def test_photometry_spotcheck(self):
        # Check the stability of a random photometric measurement
        sourcelist = pd.read_csv('test_data/e35688/e35688-nd-f0120-b00-movie-photom-12_8.csv')
        assert sourcelist.iloc[1]['aperture_sum_9'] == pytest.approx(35.10006000550969)

    def test_exptime_bins(self):
        # Check that the number of exptime bins remains fixed
        exptimes = pd.read_csv('test_data/e35688/e35688-nd-f0120-b00-movie-exptime.csv')
        assert len(exptimes) == 14

    def test_exptime_cumulative(self):
        exptimes = pd.read_csv('test_data/e35688/e35688-nd-f0120-b00-movie-exptime.csv')
        assert exptimes.expt.sum() == pytest.approx(1372.091507045843)