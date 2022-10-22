from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        15943,
        "NUV",
        depth=30,
        threads=None,
        local_root="test_data",
        recreate=False,
        aperture_sizes=[12.8],
        write={"movie": True, "image": True},
        coregister_lightcurves=False,
        stop_after="moviemaker",
        compression="rice",
        lil=True,
    )
