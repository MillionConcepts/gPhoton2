from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        775,
        "NUV",
        depth=120,
        # integer; None to deactivate (default None)
        threads=4,
        # where to both write output data and look for input data
        local_root="test_data",
        # auxiliary remote location for input data
        # remote_root="/mnt/s3",
        recreate=True,
        # list of floats; relevant only to lightcurve / photometry portion
        aperture_sizes=[12.8],
        # actually write image/movie products? otherwise hold in memory but
        # discard (possibly after performing photometry).
        write={"movie": False, "image": True},
        coregister_lightcurves=False,
        # photonpipe, moviemaker, None (default None)
        stop_after=None,
        photometry_only=False,
        # None, "gzip", "rice"
        compression="rice",
        # use array sparsification on movie frames?
        lil=True,
        # write movie frames as separate files
        burst=False,
        extended_photonlist=True,
        extended_flagging=True,
        # aspect file, don't need to set unless need to use alt
        # file, 'aspect2.parquet'
        source_catalog_file="test_data/e00775/e00775-nd-f0120-b01-movie-photom-12_8.csv"
    )
