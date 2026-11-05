This directory contains -- or should contain -- reference tables 
generated from mission .scst files and MAST's catalog databases, formatted as
Snappy-compressed .parquet files: a table of aspect solution data 
(aspect.parquet), a table of per-pointing boresight data (boresight.parquet), 
a table of visit/eclipse-level metadata (metadata.parquet), a table of
per-leg coverage apertures (leg-aperture.parquet), and a table of MAST
download URLs for raw6 and scst files (raw_data_urls.parquet). They are 
automatically referenced by components of the gPhoton 2 pipeline, but are 
also useful for eclipse selection, data fusion, coverage surveys, etc.

raw_data_urls.parquet is distributed in this GitHub repository. The other
tables are not; they are hosted at
[MAST](https://archive.stsci.edu/hlsps/gphoton/aspect_files/) and are
downloaded automatically into this directory (or into a user-specified
aspect directory) the first time each one is needed.
