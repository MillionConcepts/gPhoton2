# 1. we're not concerned about the effects of slewing etc. on one-leg
# observations, so long as we turn everything on the edge to 0.
# note: single-leg observations with enormously huge WCS do not appear to have
# big, non-circular full-depth images -- most of the grid is blank.
# TODO: completely blank?

# 2. how do we select a spatial grid/WCS for the resultant image? these are
# all nominally in shared sky coordinates, whatever the deficiencies of
# gnomonic projection, and at shared _scales_ -- but not on a shared grid.
# so...we interpolate them to some shared space? create a bounding WCS and bin
# them? something like that?
# yes. let's just do that. make a bounding wcs and interpolate all coadd
# contributors to that grid.

def coadd_galex_images(image_files: Sequence[str, Path]):
