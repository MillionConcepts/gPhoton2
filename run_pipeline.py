from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        23456,
        "NUV",
        depth=None,
        threads=4,
        local_root="test_data",
        remote_root="/mnt/s3",
        recreate=False,
        aperture_sizes=[12.8, 51.2],
        write={"movie": True, "image": True},
        coregister_lightcurves=True,
        compression="rice",
        lil=True,
        photometry_only=False,
        verbose=4
    )
