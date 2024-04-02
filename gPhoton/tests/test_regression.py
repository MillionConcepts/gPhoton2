import pytest
import csv
import pandas as pd

# I just want the dumbest possible regression tests against refactoring.
class TestExecutionRegressionMovie():
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
        sourcelist = pd.read_csv('test_data/e35688/e35688-nd-b00-f0120-movie-photom-12_8.csv')
        assert len(sourcelist) == 5217

    def test_photometry_spotcheck(self):
        # Check the stability of a random photometric measurement
        sourcelist = pd.read_csv('test_data/e35688/e35688-nd-b00-f0120-movie-photom-12_8.csv')
        assert sourcelist.iloc[1]['aperture_sum_9'] == pytest.approx(111.13503807931043)

    def test_exptime_bins(self):
        # Check that the number of exptime bins remains fixed
        exptimes = pd.read_csv('test_data/e35688/e35688-nd-b00-f0120-movie-exptime.csv')
        assert len(exptimes) == 14

    def test_exptime_cumulative(self):
        exptimes = pd.read_csv('test_data/e35688/e35688-nd-b00-f0120-movie-exptime.csv')
        assert exptimes.expt.sum() == pytest.approx(1372.091507045843)
    
    def test_extended_shapes(self):
        shapes = pd.read_csv('test_data/e35688/e35688-nd-b00-extended-shapes.csv')
        assert len(shapes) == 54

    def test_aperture_sums(self):
        sourcelist = pd.read_csv('test_data/e35688/e35688-nd-b00-f0120-movie-photom-12_8.csv')
        assert sum([sourcelist[f'aperture_sum_{k}'][0] for k in range(14)]) == pytest.approx(sourcelist['aperture_sum'][0])

class TestExecutionRegressionFullframe():
    from gPhoton.pipeline import execute_pipeline
    execute_pipeline(
        35688,
        "NUV",
        #depth=120,
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

    def test_extended_shapes(self):
        shapes = pd.read_csv('test_data/e35688/e35688-nd-b00-extended-shapes.csv')
        assert len(shapes) == 54
    
    def test_source_count(self):
        # Check that the number of extracted sources remains fixed
        sourcelist = pd.read_csv('test_data/e35688/e35688-nd-b00-ffull-image-photom-12_8.csv')
        assert len(sourcelist) == 5217

