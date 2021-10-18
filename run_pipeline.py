from gPhoton.pipeline import pipeline

source_catalog_file = "/home/michael/Desktop/galex_swarm/" \
                      "wdcat_eclipse_list.csv"

if __name__ == "__main__":
    pipeline(
        29097,
        "NUV",
        depth=480,
        threads=None,
        data_root="test_data",
        recreate=False,
        source_catalog_file=source_catalog_file
    )
