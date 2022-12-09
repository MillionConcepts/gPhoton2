def emojified(compression, depth, leg, band, eclipse_base, frame):
    band_emoji = {"NUV": "🌠", "FUV": "👻"}[band]
    ext = {"gzip": ".fits.gz", "none": ".fits", "rice": ".fits"}[compression]
    comp = {"gzip": "🤐", "none": "🍵", "rice": "🍚"}[compression]
    frame = "🎥" if frame is None else f"f{str(frame).zfill(4)}"
    prefix = f"{eclipse_base}-{band_emoji}"
    file_dict = {
        "raw6": f"{prefix}-raw6.fits.gz",
        "photonfile": f"{prefix}-🦿{leg}.parquet",
        "image": f"{prefix}️-🦿{leg}-🖼-{comp}{ext}",
        "extended_catalogs": f"{prefix}-🦿{leg}-extended-sources.csv"
    }
    if depth is not None:
        depth = str(depth).zfill(4)
        file_dict |= {
            "movie": f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-{comp}{ext}",
            # stem -- multiple aperture sizes possible
            "photomfile": f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-photom-",
            "expfiles": f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-exptime.csv"
        }
    else:
        file_dict |= {"photomfiles": [f"{prefix}-⏱️full-{leg}-🖼-photom-"]}
    return file_dict
