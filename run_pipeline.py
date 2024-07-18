from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        12148,
        "NUV",
        depth=None,
        threads=4,
        local_root="test_data",
        recreate=True,
        aperture_sizes=[12.8],
        write={"movie": True, "image": True},
        coregister_lightcurves=False,
        # stop_after="moviemaker",
        photometry_only=False,
        compression="rice",
        lil=True,
    )
