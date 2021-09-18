# import pathlib
# import re
#
# from gfcat.gfcat_utils import read_image
#
#
# def optimize_make_photometry(listfile: str, cntfile: str):
#     # TODO: specify depth in filename by default?
#     photomfile = re.sub(r"-\w\w\.parquet", "-photom.csv", listfile)
#     cntmap, flagmap, edgemap, wcs, tranges, exptimes = read_image(cntfile)
#     if not cntmap.max():
#         print('Image contains nothing.')
#         pathlib.Path(os.path.dirname(listfile) + "/No{band}".format(
#         band=band)
#                                 ).touch()
#         return
#
#     trange, exptime = tranges[0], exptimes[0]
#     if exptime < 600:
#         print("Skipping low exposure time visit.")
#         pathlib.Path("{path}/LowExpt".format(path=os.path.dirname(
#         photonfile))).touch()
#         return
#     movmap, _, _, _, tranges, exptimes = read_image(cntfilename.replace(
#     "-cnt", "-mov"))
#     daofind = DAOStarFinder(fwhm=5, threshold=0.01)
#     sources = daofind(cntmap / exptime)
#     try:
#         print(f'Located {len(sources)} sources.')
#     except TypeError:
#         print('Image contains no sources.')
#         pathlib.Path(os.path.dirname(photonfile) + "/No{band}".format(
#         band=band)
#                                         ).touch()
#         return
#     positions = (sources["xcentroid"], sources["ycentroid"])
#     apertures = CircularAperture(positions, r=8.533333333333326)
#     phot_table = aperture_photometry(cntmap, apertures)
#     flag_table = aperture_photometry(flagmap, apertures)
#     edge_table = aperture_photometry(edgemap, apertures)
#
#     phot_visit = sources.to_pandas()
#     phot_visit["xcenter"] = phot_table.to_pandas().xcenter.tolist()
#     phot_visit["ycenter"] = phot_table.to_pandas().ycenter.tolist()
#     phot_visit["aperture_sum"] = phot_table.to_pandas().aperture_sum.tolist()
#     phot_visit["aperture_sum_mask"] = flag_table.to_pandas(
#     ).aperture_sum.tolist()
#     phot_visit["aperture_sum_edge"] = edge_table.to_pandas(
#     ).aperture_sum.tolist()
#     phot_visit["ra"] = [
#         wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()[0]
#         for pos in apertures.positions
#     ]
#     phot_visit["dec"] = [
#         wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()[1]
#         for pos in apertures.positions
#     ]
#
#     for i, frame in enumerate(movmap):
#         mc.print_inline("Extracting photometry from frame #{i}".format(i=i))
#         phot_visit["aperture_sum_{i}".format(i=i)] = (
#             aperture_photometry(frame, apertures).to_pandas()[
#             "aperture_sum"].tolist()
#         )
#     print("Writing data to {f}".format(f=photomfile))
#     phot_visit.to_csv(photomfile, index=False)
#     pd.DataFrame(
#         {
#             "expt": exptimes,
#             "t0": np.array(tranges)[:, 0].tolist(),
#             "t1": np.array(tranges)[:, 1].tolist(),
#         }
#     ).to_csv(cntfilename.replace("-cnt.fits.gz", "-exptime.csv"),
#     index=False)
#
# return