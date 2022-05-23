from gPhoton.pipeline import execute_pipeline

source_catalog_file = (
    "/home/michael/Desktop/galex_swarm/wdcat_eclipse_list.csv"
)

if __name__ == "__main__":
    execute_pipeline(
        16578,
        "NUV",
        depth=None,
        threads=4,
        local_root="test_data",
        recreate=False,
        maxsize=22*1024**3,
        source_catalog_file=None,
        aperture_sizes=[12.8, 25.6, 51.2],
        write={"movie": False, "image": True},
        coregister_lightcurves=True,
        stop_after="moviemaker",
        compression="gzip",
        lil=True
        # hdu_constructor_kwargs={'quantize_level': 16}
    )
