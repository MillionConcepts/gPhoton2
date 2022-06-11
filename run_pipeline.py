from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        23456,
        "NUV",
        depth=120,
        threads=4,
        local_root="test_data",
        recreate=False,
        aperture_sizes=[12.8, 51.2],
        write={"movie": True, "image": True},
        coregister_lightcurves=True,
        stop_after="moviemaker",
        compression="rice",
        lil=True
    )
