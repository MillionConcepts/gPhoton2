from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        33795,
        "NUV",
        depth=120,
        # integer; None to deactivate (default None)
        threads=4,
        # where to both write output data and look for input data
        local_root="test_data",
        # auxiliary remote location for input data
        # remote_root="/mnt/s3",
        recreate=False,
        # list of floats; relevant only to lightcurve / photometry portion
        aperture_sizes=[12.8],
        # actually write image/movie products? otherwise hold in memory but
        # discard (possibly after performing photometry).
        write={"movie": False, "image": True},
        coregister_lightcurves=False,
        # photonpipe, moviemaker, None (default None)
        stop_after=None,
        photometry_only=False,
        # "none", "gzip", "rice"
        compression="rice",
        # use array sparsification on movie frames?
        lil=True,
        # write movie frames as separate files
        burst=False,
        extended_photonlist=True,
        extended_flagging=False,
        verbose=2
        #source_catalog_file="test_data/e00775/e00775-nd-f0120-b01-movie-photom-12_8.csv"
    )
