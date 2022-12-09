def emojified(compression, depth, legs, eclipse_base, frame):
    bands, band_emoji = ("NUV", "FUV"), ("🌠", "👻")
    file_dict = {}
    ext = {"gzip": ".fits.gz", "none": ".fits", "rice": ".fits"}[compression]
    comp = {"gzip": "🤐", "none": "🍵", "rice": "🍚"}[compression]
    frame = "🎥" if frame is None else f"f{str(frame).zfill(4)}"
    for band, emoji in zip(bands, band_emoji):
        prefix = f"{eclipse_base}-{emoji}"
        band_dict = {
            "raw6": f"{prefix}-raw6.fits.gz",
            "photonfiles": [f"{prefix}-🦿{leg}.parquet" for leg in legs],
            "images": [f"{prefix}-️-🦿{leg}-🖼-{comp}-{ext}" for leg in legs],
            "extended_catalogs": [
                f"{prefix}-🦿{leg}-extended-sources.csv" for leg in legs
            ]
        }
        if depth is not None:
            depth = str(depth).zfill(4)
            band_dict |= {
                "movies": [
                    f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-{comp}{ext}"
                    for leg in legs
                ],
                # stem -- multiple aperture sizes possible
                "photomfiles": [
                    f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-photom-"
                    for leg in legs
                ],
                "expfiles": [
                    f"{prefix}-⏱️{depth}-🦿{leg}-{frame}-exptim.csv"
                    for leg in legs
                ]
            }
        else:
            band_dict |= {
                "photomfiles": [
                    f"{prefix}-⏱️full-{leg}-🖼-photom-"
                    for leg in legs
                ]
            }
        file_dict[band] = band_dict
    return file_dict
