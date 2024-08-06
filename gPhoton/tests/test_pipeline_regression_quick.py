import pytest
import pandas as pd

from gPhoton.pipeline import execute_pipeline
import pandas as pd

eclipse = 23456

class TestPipelineRegressionQuick01():
    execute_pipeline(
        eclipse,
        "NUV",
        depth=120,
        threads=4,
        local_root="regr_test_data",
        recreate=True,
        aperture_sizes=[12.8],
        write={"movie": False, "image": False},
        coregister_lightcurves=False,
        #stop_after='photonpipe',
        photometry_only=False,
        compression="rice",
        lil=True,
        burst=False,
        extended_photonlist=False,
    )

class TestPhotometryRegression01():
    data = pd.read_csv('regr_test_data/e23456/e23456-nd-f0120-b00-movie-photom-12_8.csv',index_col=None)
    assert data.shape[0]==2938 # photometry file rows
    assert data.shape[1]==63 # photometry file columns