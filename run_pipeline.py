from gPhoton.pipeline import pipeline

source_catalog_file = "/home/michael/Desktop/galex_swarm/" \
                      "wdcat_eclipse_list.csv"

if __name__ == "__main__":
    pipeline(
        18051,
        "NUV",
        depth=5,
        threads=4,
        data_root="test_data",
        recreate=False,
        maxsize=30*1024**3,
        source_catalog_file=source_catalog_file
    )
