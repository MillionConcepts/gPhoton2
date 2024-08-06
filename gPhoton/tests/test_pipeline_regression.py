import pytest
import pandas as pd

from gPhoton.pipeline import execute_pipeline
import pandas as pd

dis_eclipse = 8330
ais_eclipse = 18669
cai_eclipse = 20334
gii_eclipse = 25125

class TestPipelineRegression01():
    execute_pipeline(
        dis_eclipse, # DIS eclipse
        "NUV", # band
        compression="rice",
        aperture_sizes=[6.0, 12.8,],
        threads=4,
        depth=120,
        local_root="regr_test_data",
        write={"movie": True, "image": True},
        recreate=True,
        coregister_lightcurves=False,
        photometry_only=False,
        lil=True,
        burst=False,
        #aspect="aspect2"
    )

class TestPipelineRegression02():
    execute_pipeline(
        ais_eclipse, # AIS eclipse
        "NUV", # band
        compression="gzip",
        aperture_sizes=[6.0, 12.8,],
        threads=4,
        depth=120,
        local_root="regr_test_data",
        write={"movie": False, "image": True},
        recreate=True,
        coregister_lightcurves=False,
        photometry_only=False,
        lil=True,
        burst=False,
        #aspect="aspect2"
    )

class TestPipelineRegression03():
    execute_pipeline(
        cai_eclipse, # CAI eclipse
        "FUV", # band
        compression="rice",
        aperture_sizes=[6.0, 12.8,],
        threads=None,
        depth=120,
        local_root="regr_test_data",
        write={"movie": True, "image": True},
        recreate=True,
        coregister_lightcurves=False,
        photometry_only=False,
        lil=True,
        burst=False,
        #aspect="aspect2"
    )

class TestPipelineRegression04():
    execute_pipeline(
        gii_eclipse, # GII eclipse
        "FUV", # band
        compression="none",
        aperture_sizes=[6.0, 12.8,],
        threads=None,
        depth=120,
        local_root="regr_test_data",
        write={"movie": False, "image": False},
        recreate=True,
        coregister_lightcurves=False,
        photometry_only=False,
        lil=True,
        burst=False,
        #aspect="aspect2"
    )

class TestPipelineRegression05():
    phot = pd.read_csv(f'regr_test_data/e{dis_eclipse:05}/e{dis_eclipse:05}-nd-f0120-b00-movie-photom-6_0.csv',
                    index_col=None)
    cat = phot[['ra','dec']]
    cat['eclipse']=dis_eclipse
    cat.to_csv(f'regr_test_data/e{dis_eclipse:05}/e{dis_eclipse:05}-nd-cat.csv',index=None)
    execute_pipeline(
        dis_eclipse, # DIS eclipse
        "FUV", # band
        compression="rice",
        aperture_sizes=[6.0, 12.8,],
        threads=4,
        depth=120,
        local_root="regr_test_data",
        write={"movie": True, "image": True},
        recreate=True,
        coregister_lightcurves=True,
        photometry_only=False,
        lil=True,
        burst=False,
        #aspect="aspect2",
        source_catalog_file=f'regr_test_data/e{dis_eclipse:05}/e{dis_eclipse:05}-nd-cat.csv' # change to accept an in-memory object as well as a file path
    )

class TestPipelineRegression06():
    execute_pipeline(
        ais_eclipse, # AIS eclipse
        "FUV", # band
        compression="rice",
        aperture_sizes=[6.0, 12.8,],
        threads=4,
        depth=120,
        local_root="regr_test_data",
        write={"movie": True, "image": True},
        recreate=True,
        coregister_lightcurves=False,
        photometry_only=False,
        lil=True,
        burst=False,
        extended_photonlist="regr_test_data/e14488/e14488-nd-b00.parquet",
    )

class TestPipelineRegression07():
    execute_pipeline(
        cai_eclipse, # CAI eclipse
        "NUV", # band
        compression="rice",
        aperture_sizes=[6.0, 12.8,],
        threads=4,
        depth=120,
        local_root="regr_test_data",
        write={"movie": True, "image": True},
        recreate=True,
        coregister_lightcurves=True,
        photometry_only=False,
        lil=True,
        burst=False,
        aspect="aspect", # change this to accept a file path, not a code name
    )

class TestPipelineRegression08():
    phot = pd.read_csv(f'regr_test_data/e{gii_eclipse:05}/e{gii_eclipse:05}-fd-f0120-b00-movie-photom-6_0.csv',
                    index_col=None)
    cat = phot[['ra','dec']]
    cat['eclipse']=gii_eclipse
    cat.to_csv(f'regr_test_data/e{gii_eclipse:05}/e{gii_eclipse:05}-fd-cat.csv',index=None)

    execute_pipeline(
        gii_eclipse, # GII eclipse
        "NUV", # band
        compression="none",
        aperture_sizes=[6.0, 12.8,],
        threads=4,
        depth=120,
        local_root="regr_test_data",
        write={"movie": True, "image": True},
        recreate=True,
        coregister_lightcurves=False,
        photometry_only=False,
        lil=True,
        burst=False,
        #aspect="aspect2",
        source_catalog_file=f'regr_test_data/e{gii_eclipse:05}/e{gii_eclipse:05}-fd-cat.csv',
    )
