from gPhoton.pipeline import execute_pipeline

source_catalog_file = (
    "/home/michael/Desktop/galex_swarm/wdcat_eclipse_list.csv"
)

if __name__ == "__main__":
    execute_pipeline(
        39971,
        "NUV",
        depth=60,
        threads=None,
        data_root="test_data",
        recreate=False,
        maxsize=16*1024**3,
        source_catalog_file=source_catalog_file,
        aperture_sizes=[12.8, 25.6, 51.2],
        write={"movie": True, "image": True},
    )

