import pytest

source_catalog_file = (
    "/home/michael/Desktop/galex_swarm/wdcat_eclipse_list.csv"
)

PIPELINE_CASES = [
    {
        "eclipse": 12160,
        "band": "FUV",
        "depth": 60,
        "threads": None,
        "data_root": "test_data",
        "recreate": True,
        "source_catalog_file": source_catalog_file,
        "aperture_sizes": [12.8, 25.6, 51.2],
        "write": {"movie": True, "image": True}
    },
    {
        "eclipse": 39971,
        "band": "NUV",
        "depth": 60,
        "threads": 4,
        "data_root": "test_data",
        "recreate": True,
        "aperture_sizes": [12.8],
        "write": {"movie": True, "image": True}
    }
]



@pytest.mark.parametrize("pipeline_kwargs", PIPELINE_CASES)
def test_wrapper_for_coverage(pipeline_kwargs):
    from gPhoton.pipeline import pipeline
    pipeline(**pipeline_kwargs)