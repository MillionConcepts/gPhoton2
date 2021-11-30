from gPhoton.pipeline import execute_pipeline

source_catalog_file = (
    "/home/michael/Desktop/galex_swarm/wdcat_eclipse_list.csv"
)

if __name__ == "__main__":
    execute_pipeline(
        12160,
        "FUV",
        depth=60,
        threads=None,
        local_root="test_data",
        recreate=True,
        maxsize=20*1024**3,
        source_catalog_file=None,
        aperture_sizes=[12.8],
        write={"movie": True, "image": True},
        coregister_lightcurves=True
    )
