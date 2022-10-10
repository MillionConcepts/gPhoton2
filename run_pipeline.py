from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        781,
        "NUV",
        depth=120,
        threads=4,
        local_root="test_data",
        recreate=False,
        aperture_sizes=[12.8, 51.2],
        write={"movie": False, "image": True},
        coregister_lightcurves=True,
        compression="rice",
        lil=True
    )
