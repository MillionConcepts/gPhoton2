from gPhoton.pipeline import pipeline

source_catalog_file = (
    "/home/michael/Desktop/galex_swarm/" "wdcat_eclipse_list.csv"
)

if __name__ == "__main__":
    pipeline(
        12843,
        "NUV",
        depth=5,
        threads=2,
        data_root="test_data",
        recreate=False,
        maxsize=25*1024**3,
        source_catalog_file=source_catalog_file,
        aperture_sizes=[12.8, 25.6, 51.2],
        write={"movie": False, "image": True},
    )

