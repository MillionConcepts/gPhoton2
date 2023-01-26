from gPhoton.pipeline import execute_pipeline

if __name__ == "__main__":
    execute_pipeline(
        10982,
        "NUV",
        depth=None,
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
        write={"movie": True, "image": True},
        coregister_lightcurves=False,
        # photonpipe, moviemaker, None (default None)
        # stop_after="moviemaker",
        photometry_only=False,
        # None, "gzip", "rice"
        compression="gzip",
        # use array sparsification on movie frames?
        lil=True,
        # write movie frames as separate files
        burst=False,
        extended_photonlist=False,
        daophot_params={'threshold': 0.008, 'fwhm': 6, 'sharplo': 0.05},
        bg_params={
            'threshold': 0.004,
            'fwhm': 16,
            'min_radius': 2,
            'peak_col': 'peak',
            'browse': True
        },
        do_background=True
    )
