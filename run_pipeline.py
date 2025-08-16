from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        36690,
        "NUV",
        depth=250,
        # integer; None to deactivate (default None)
        threads=4,
        # where to both write output data and look for input data
        local_root="/home/bekah/eclipse",
        # auxiliary remote location for input data
        # remote_root="/mnt/s3",
        recreate=True,
        #source_catalog_file="/home/bekah/eclipse/e23456/e23456-fd-f0120-b00-movie-photom-12_8.csv",
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
        verbose=2,
        #single_leg=1,
        photonlist_cols=['roll']
    )
