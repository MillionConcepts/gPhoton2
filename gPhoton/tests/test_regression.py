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

    def test_lightcurve_parse(self):
        from gPhoton.io.lc_utils import parse_lightcurve
        lcs = parse_lightcurve('test_data/e35688/e35688-nd-b00-f0120-movie-photom-12_8.csv',
                               'test_data/e35688/e35688-nd-b00-f0120-movie-exptime.csv')
        assert len(lcs) == 5217

    def test_exposure_parse(self):
        from gPhoton.io.lc_utils import parse_exposure_time
        expt = parse_exposure_time('test_data/e35688/e35688-nd-b00-f0120-movie-exptime.csv')
        assert len(expt) == 14

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

class TestExecutionRegressionFullframeGzip():
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
        #compression="rice",
        lil=True,
    )

    def test_image_open(self):
        from gPhoton.io.fits_utils import get_fits_data
        cnt_img = get_fits_data([FILENAME],dim=1)
        assert cnt_img.shape == (3099, 3065)
        flag_img = get_fits_data(FILENAME,dim=2)
        assert flag_img.shape == (3099, 3065)
        edge_img = get_fits_data(FILENAME,dim=3)
        assert edge_img.shape == (3099, 3065)
    
    def test_wcs_read(self):
        from gPhoton.io.fits_utils import read_wcs_from_fits
        headers,wcs = read_wcs_from_fits(FILENAME)
        assert wcs.wcs.crval[0] == pytest.approx(323.49279419)
        assert wcs.wcs.crval[1] == pytest.approx(-2.06396934)        

