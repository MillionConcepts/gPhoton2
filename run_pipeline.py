from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        23456,
        "NUV",
        depth=30,
        threads=4,
        local_root="test_data",
        recreate=True,
        aperture_sizes=[12.8],
        write={"movie": True, "image": True},
        coregister_lightcurves=False,
        stop_after="photonpipe",
        compression="rice",
        lil=True,
    )
