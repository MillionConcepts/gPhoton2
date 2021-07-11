import pandas as pd
import numpy as np
from gPhoton import cal, curvetools as ct, galextools as gt, FileUtils as fu, \
    MCUtils as mc, gAperture, gQuery as gq
from astropy.io import fits as pyfits
from astropy import wcs as pywcs
from photutils import DAOStarFinder
from photutils import aperture_photometry
from photutils import CircularAperture
import requests
import pathlib
import os
import shutil
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import DBSCAN
import warnings
import sqlalchemy as sql

def obstype_from_eclipse(eclipse):
    try:
        t = fu.web_query_aspect(eclipse,quiet=True)[3]
        obsdata = gq.getArray(gq.obstype_from_t(t.mean()))[0]
        obstype = obsdata[0]
        nlegs = obsdata[4]
        print(
            "e{eclipse} is an {obstype} mode observation w/ {n} legs.".format(
                eclipse=eclipse, obstype=obstype, n=nlegs
            )
        )
        return obstype, len(t), nlegs
    except (IndexError, TypeError, ValueError):
        return "NoData", 0, 0


def eclipse_to_urls(eclipse, mast_url="http://galex.stsci.edu/gPhoton/RAW6/"):
    eclipse_range = "e{lower}_{upper}".format(
        lower=int(np.floor(eclipse / 100) * 100),
        upper=int(np.floor(eclipse / 100) * 100 + 99),
    )
    base_url = "{mast_url}{eclipse_range}/{eclipse}/e{eclipse}".format(
        mast_url=mast_url, eclipse_range=eclipse_range, eclipse=str(eclipse).zfill(5)
    )
    return {
        "scst_url": "{base_url}-scst.fits.gz".format(base_url=base_url),
        "NUV": {"raw6_url": "{base_url}-nd-raw6.fits.gz".format(base_url=base_url)},
        "FUV": {"raw6_url": "{base_url}-fd-raw6.fits.gz".format(base_url=base_url)},
    }


def eclipse_to_files(eclipse, data_directory="/Volumes/BigDataDisk/gPhotonData/GTDS"):
    eclipse_path = "{d}/e{e}/".format(d=data_directory, e=eclipse)
    eclipse_base = "{ep}e{e}".format(ep=eclipse_path, e=eclipse)
    return {
        "NUV": {"raw6": "{eb}-nd-raw6.fits.gz".format(eb=eclipse_base)},
        "FUV": {"raw6": "{eb}-fd-raw6.fits.gz".format(eb=eclipse_base)},
    }


def download_raw6(
    eclipse, band, force=False, data_directory="/Volumes/BigDataDisk/gPhotonData/GTDS"
):
    url = eclipse_to_urls(eclipse)[band]["raw6_url"]
    filepath = eclipse_to_files(eclipse, data_directory=data_directory)[band]["raw6"]
    if os.path.exists(filepath) and not force:
        print("{fn} already exists.".format(fn=filepath))
        print("\tUse keyword `force` to re-download.")
        return filepath
    print(f"Writing data to {filepath}")
    if not os.path.exists(os.path.dirname(filepath)):
        print("Creating {dirname}".format(dirname=os.path.dirname(filepath)))
        os.makedirs(os.path.dirname(filepath))
    mc.print_inline("Querying {url}".format(url=url))
    r = requests.get(url)
    if r.status_code == 404:
        mc.print_inline(
            "Querying {url} (retry)".format(url=url.replace("RAW6", "RAW6.2"))
        )
        # Higher number visits have a different URL, so try that one...
        r = requests.get(url.replace("RAW6", "RAW6.2"))
        if r.status_code == 404:
            mc.print_inline(
                "No raw6 file available for {band} e{eclipse}.".format(
                    band=band, eclipse=eclipse
                )
            )
            print("")
            return False
    mc.print_inline(
        "Downloading {url} \n\t to {filepath}".format(url=url, filepath=filepath)
    )
    with open(filepath, "wb") as fd:
        for chunk in r.iter_content(chunk_size=128):
            fd.write(chunk)
    print("Downloaded {filepath}".format(filepath=filepath))
    return filepath



    return image


def compute_shutter(events, trange, shutgap=0.05):
    ix = np.where(
        (events["t"] >= trange[0]) & (events["t"] < trange[1]) & (events["flags"] == 0)
    )
    t = np.sort([trange[0]] + list(np.unique(events["t"].iloc[ix])) + [trange[1]])
    ix = np.where(t[1:] - t[:-1] >= shutgap)
    shutter = np.array(t[1:] - t[:-1])[ix].sum()
    return shutter


def compute_exptime(events, band, trange):
    rawexpt = trange[1] - trange[0]
    times = np.array(events["t"])

    tix = np.where((times >= trange[0]) & (times < trange[1]))

    shutter = compute_shutter(events, trange)

    # Calculate deadtime
    model = {
        "NUV": [-0.000434730599193, 77.217817988],
        "FUV": [-0.000408075976406, 76.3000943221],
    }

    rawexpt -= shutter  # THIS IS A CORRECTION THAT NEEDS TO BE
    # IMPLEMENTED IN gPhoton!!!
    if rawexpt == 0:
        return rawexpt
    gcr = len(times[tix]) / rawexpt
    feeclkratio = 0.966
    refrate = model[band][1] / feeclkratio
    scr = model[band][0] * gcr + model[band][1]
    deadtime = 1 - scr / feeclkratio / refrate

    return rawexpt * (1.0 - deadtime)


def fits_header(band, wcs, tranges, exptimes, hdu=None):

    hdu = hdu if hdu else pyfits.PrimaryHDU()

    hdu.header["CDELT1"], hdu.header["CDELT2"] = wcs.wcs.cdelt
    hdu.header["CTYPE1"], hdu.header["CTYPE2"] = wcs.wcs.ctype
    hdu.header["CRPIX1"], hdu.header["CRPIX2"] = wcs.wcs.crpix
    hdu.header["CRVAL1"], hdu.header["CRVAL2"] = wcs.wcs.crval
    hdu.header["EQUINOX"], hdu.header["EPOCH"] = 2000.0, 2000.0
    hdu.header["BAND"] = 1 if band == "NUV" else 2
    hdu.header["VERSION"] = "v{v}".format(v="imcaldev")
    hdu.header["N_FRAME"] = len(tranges)
    for i, trange in enumerate(tranges):
        hdu.header["T0_{i}".format(i=i)] = trange[0]
        hdu.header["T1_{i}".format(i=i)] = trange[1]
        hdu.header["EXPT_{i}".format(i=i)] = exptimes[i]
    return hdu


def make_wcs(
    skypos,
    pixsz=0.000416666666666667,  # Same as the GALEX intensity maps
    imsz=[3200, 3200],  # Same as the GALEX intensity maps):
):
    wcs = pywcs.WCS(naxis=2)
    wcs.wcs.cdelt = np.array([-pixsz, pixsz])
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [(imsz[1] / 2.0) + 0.5, (imsz[0] / 2.0) + 0.5]
    wcs.wcs.crval = skypos
    return wcs


def read_image(fn):
    hdu = pyfits.open(fn)
    print("Opened {fn}".format(fn=fn))
    image = hdu[0].data
    exptimes, tranges = [], []
    for i in range(hdu[0].header["N_FRAME"]):
        exptimes += [hdu[0].header["EXPT_{i}".format(i=i)]]
        tranges += [
            [hdu[0].header["T0_{i}".format(i=i)], hdu[0].header["T1_{i}".format(i=i)]]
        ]
    skypos = (hdu[0].header["CRVAL1"], hdu[0].header["CRVAL2"])
    nphots = hdu[0].header["NPHOTs"]
    wcs = make_wcs(skypos)
    print("\tParsed file header.")
    try:
        flagmap = hdu[1].data
        edgemap = hdu[2].data
        print("\tRetrieved flag and edge maps.")
    except IndexError:
        flagmap = None
        edgemap = None
    return image, flagmap, edgemap, wcs, tranges, exptimes


def make_qa_plots(fn):
    if not "cnt" in fn:
        return
    image, flagmap, edgemap, _, _, _ = read_image(fn)
    plt.ioff()
    plt.figure(figsize=(5, 5.3))
    plt.title(fn.split("/")[-1].split(".")[0])
    plt.imshow(np.log10(image), cmap="Greys_r", origin="lower")
    plt.imshow(np.log10(edgemap), origin="lower", cmap="cool")
    plt.imshow(np.log10(flagmap), origin="lower", cmap="Wistia")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(fn.replace(".fits.gz", ".png"), dpi=200)
    plt.close()
    plt.ion()
    return


def make_gap_qa_plots(eclipse, sourceid, imfile=None,
    data_directory="/Volumes/BigDataDisk/gPhotonData/GTDS",rerun=False):
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    photonfile = "{data_directory}/e{eclipse}/e{eclipse}-{b}d.h5".format(
        data_directory=data_directory, eclipse=eclipse, b="n")
    try:
        exptime = pd.read_csv('{d}/e{e}-nd-exptime.csv'.format(d=os.path.dirname(photonfile),e=eclipse),index_col=None)
        photom = pd.read_csv('{d}/e{e}-nd-photom.csv'.format(d=os.path.dirname(photonfile),e=eclipse),index_col='id')
    except:
        print('\tCannot find photometry files.')
        return
    cnt = photom.loc[sourceid][
        ["aperture_sum_{i}".format(i=i) for i in np.arange(len(exptime))]
                            ].values
    cps = cnt / exptime.expt.values.flatten()
    cps_err = np.sqrt(cnt) / exptime.expt.values.flatten()
    gapdir = os.path.dirname(photonfile) + "/gap/"
    if not os.path.exists(gapdir):
        print("\tCreating {dirname}".format(dirname=gapdir))
        os.makedirs(gapdir)
    gapfile = gapdir+'e{e}-nd-{i}-30s.csv'.format(e=eclipse,i=sourceid)
    if os.path.exists(gapfile) and not rerun:
        print('\tReading gAperture file.')
        gap = pd.read_csv(gapfile,index_col=None)
    else:
        gap = gAperture('NUV',
                        (photom.loc[sourceid].ra,photom.loc[sourceid].dec),
                        gt.aper2deg(6), trange=[exptime['t0'].min(),exptime['t1'].max()],
                        verbose=2, stepsz=30, csvfile=gapfile, detsize=1.25)
    
    # Screen again for real variability, given more data points
    ad = stats.anderson(gap["cps"][1:-1])
    if ad.statistic<ad.critical_values.min(): # 15% significance level
        print('\tNot a significantly variable source.')
        return
    sr = stats.spearmanr(gap['cps'][1:-1],gap['detrad'][1:-1],nan_policy='omit')
    if (np.abs(sr.correlation)>0.3):
        print('\tLikely artifact: strongly correlated to detrad')
        return
    # 960 pixels == 0.4 degree from det center
    if ((np.abs(sr.correlation)>0.15) and
        (np.sqrt((photom.loc[sourceid].xcenter-1600)**2 +
                 (photom.loc[sourceid].ycenter-1600)**2)>960)):
        print('\tikely artifact: at det edge with correlation to detrad')
        return
#    if (np.array(gap['flags'].values,dtype='int16') & 16).any():
#        print('Skipping variable bright source in nonlinear regime.')
#        return
    plt.ion()
    nfigs = 4
    if imfile:
        nfigs += 1
    t = gap["t0"] - gap["t0"].min()+15
    fig = plt.figure(figsize=(3, 7))
    fig.patch.set_visible(False)
    gs = fig.add_gridspec(7, 3)

    # if obstuple:
    #    fig.title('e{e} #{i}'.format(e=obstuple[0],i=obstuple[1]))
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, gap["cps"], "k-")
    ax1.plot(exptime['t0']-exptime['t0'].min()+60,cps,'r-',alpha=0.5)
    ax1.fill_between(
        t, gap["cps"] - 3 * gap["cps_err"], gap["cps"] + 3 * gap["cps_err"]
    )
    ax1.axhline(y=0,color='k',linestyle=':',alpha=0.2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylabel("gAp")
    ax1.set_xlim([min(t), max(t)])
    ax1.set_ylim([0.98*np.nanmin((gap["cps"]-3*gap["cps_err"])[1:-1]),
                  1.02*np.nanmax((gap["cps"]+3*gap["cps_err"])[1:-1])])
    ax1.patch.set_visible(False)
    ax1.axis('off') # remove bounding box

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, gap["detrad"], "k-",
        label='Corr: {sr}'.format(
            sr=np.round(sr.correlation,2)))
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylabel("rad")
    ax2.set_xlim([min(t), max(t)])
    ax2.patch.set_visible(False)
    ax2.legend()

    ax3 = fig.add_subplot(gs[2,:])
    ax3.plot(t,np.log10(gap['exptime']),'k-')#,
    #    label='Corr: {sr}'.format(
    #        sr=np.round(stats.spearmanr(
    #            gap['cps'],gap['exptime']).correlation,2)))
    ax3.axhline(y=0.5,color='k',linestyle=':',alpha=0.2)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_ylabel('expt')
    ax3.set_xlim([min(t),max(t)])
    ax3.patch.set_visible(False)
    #ax3.legend()

    ax4 = fig.add_subplot(gs[3,:])
    ax4.plot(t,gap['racent']-np.median(gap['racent']),'b-')
    ax4.plot(t,gap['deccent']-np.median(gap['deccent']),'r-')
    #ax4.fill_between(t,-gt.aper2deg(6),gt.aper2deg(6))
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_ylabel('RA Dec')
    ax4.set_xlim([min(t),max(t)])
    ax4.set_ylim([-gt.aper2deg(6)/3,gt.aper2deg(6)/3])
    ax4.patch.set_visible(False)

    cntfilename = photonfile.replace(".h5", "-cnt.fits.gz")
    if os.path.exists(cntfilename):
        image, flagmap, edgemap, _, _, _ = read_image(cntfilename) 
        stampsz = 100
        x = int(np.round(photom.loc[sourceid].xcentroid))
        y = int(np.round(photom.loc[sourceid].ycentroid))
        ax5 = fig.add_subplot(gs[4:,:])
        ax5.imshow(np.log10(image[y - stampsz : y + stampsz, x - stampsz : x + stampsz]),origin='lower',cmap='Greys_r')
        ax5.imshow(np.log10(edgemap[y - stampsz : y + stampsz, x - stampsz : x + stampsz]),origin='lower',cmap='cool')
        ax5.imshow(np.log10(flagmap[y - stampsz : y + stampsz, x - stampsz : x + stampsz]),origin='lower',cmap='Wistia')
        ax5.set_xticks([])
        ax5.set_yticks([])

    plt.tight_layout()
    plt.savefig(gapfile.replace('30s.csv','30s-qa.png'),
                transparent=True,dpi=200)
    plt.close()
    return

def make_images(
    eclipse,
    band,
    pixsz=0.000416666666666667,
    imsz=[3200, 3200],
    data_directory="/Volumes/BigDataDisk/gPhotonData/GTDS",
    detrad_pix=350,
    bins=["", 120],
):
    photonfile = "{data_directory}/e{eclipse}/e{eclipse}-{b}d.h5".format(
        data_directory=data_directory, eclipse=eclipse, b="n" if band=="NUV" else "f"
    )

    if os.path.exists(os.path.dirname(photonfile) + "/No{band}".format(band=band)):
        print("No data.")
        return

 #   if (
 #       os.path.exists(photonfile.replace(".h5", "-cnt.fits"))
 #       or os.path.exists(photonfile.replace(".h5", "-cnt.fits.gz"))
 #   ) & (
 #       os.path.exists(photonfile.replace(".h5", "-mov.fits"))
 #       or os.path.exists(photonfile.replace(".h5", "-mov.fits"))
 #   ):
 #       print(
 #           "{band} images already exist for e{eclipse}.".format(
 #               band=band, eclipse=eclipse
 #           )
 #       )
 #       return

    xcalfilename = photonfile.replace(".h5", "-xcal.h5")
    print("Reading data from {xcalfilename}".format(xcalfilename=xcalfilename))
    events = pd.read_hdf(xcalfilename, "events")
    if (
        not (0 in np.unique(events['flags'].values)) or not np.isfinite(events["ra"]).any()
    ):
        print("No unflagged data.")
        pathlib.Path(
            os.path.dirname(photonfile) + "/No{band}".format(band=band)
        ).touch()
        return

    for binsz in bins:
        if not binsz:
            cntfilename = photonfile.replace(".h5", "-cnt.fits")
        else:
            cntfilename = photonfile.replace(".h5", f"-mov-{int(binsz)}s.fits")
        print("Integrating {cntfilename}".format(cntfilename=cntfilename))
        tranges, exptimes = [], []
        image, flagmap, edgemap = [], [], []  # to try to recover some memory
        trange = [events["t"].min(), events["t"].max()]
        print(f'\t[{trange[0]},{trange[1]}]')
        t0s = np.arange(trange[0], trange[1], binsz if binsz else trange[1] - trange[0])
        center_skypos = (
            events["ra"].min() + (events["ra"].max() - events["ra"].min()) / 2,
            events["dec"].min() + (events["dec"].max() - events["dec"].min()) / 2,
        )
        wcs = make_wcs(center_skypos)

        for i, t0 in enumerate(t0s):
            mc.print_inline("\tProcessing frame {i} of {n}".format(i=i + 1, n=len(t0s)))
            t1 = t0 + (binsz if binsz else trange[1] - trange[0])
            ix = np.where(
                (events["t"] >= t0) & (events["t"] < t1) & (events["flags"] == 0)
            )
            if not len(ix[0]):
                continue
            tranges += [[t0, t1]]
            coo = list(zip(events["ra"].iloc[ix[0]], events["dec"].iloc[ix[0]]))
            foc = wcs.sip_pix2foc(wcs.wcs_world2pix(coo, 1), 1)
            weights = 1.0 / events["response"][ix[0]]
            H, xedges, yedges = np.histogram2d(
                foc[:, 1] - 0.5,
                foc[:, 0] - 0.5,
                bins=imsz,
                range=([[0, imsz[0]], [0, imsz[1]]]),
                weights=weights,
            )
            foc = []
            if len(t0s) == 1:
                image = H
            else:
                image += [H]
            exptimes += [compute_exptime(events, band, tranges[-1])]
        if not binsz:
            ix = np.where(
                (events.col.values > 0.0)
                & (events.col.values < 799.0)
                & (events.row.values > 0.0)
                & (events.row.values < 799.0)
                & (events.t.values >= trange[0])
                & (events.t.values < trange[1])
            )
            mask, maskinfo = cal.mask(band)
            masked_ix = np.where(
                mask[
                    np.array(events.col.values[ix], dtype="int64"),
                    np.array(events.row.values[ix], dtype="int64"),
                ]
                == 0
            )
            try:
                coo = list(
                    zip(
                        events.ra.values[ix][masked_ix],
                        events.dec.values[ix][masked_ix],
                    )
                )
                foc = wcs.sip_pix2foc(wcs.wcs_world2pix(coo, 1), 1)
                flagmap, _, _ = np.histogram2d(
                    foc[:, 1] - 0.5,
                    foc[:, 0] - 0.5,
                    bins=imsz,
                    range=([[0, imsz[0]], [0, imsz[1]]]),
                )
                print("\tGenerated flag map.")
            except:
                print("\tProducing empty flag map.")
                flagmap = np.zeros(imsz)
            edge_ix = np.where(
                np.sqrt(
                    (np.array(events.col.values[ix], dtype="int64") - 400) ** 2
                    + (np.array(events.row.values[ix], dtype="int64") - 400) ** 2
                )
                > detrad_pix
            )
            try:
                coo = list(
                    zip(events.ra.values[ix][edge_ix], events.dec.values[ix][edge_ix])
                )
                foc = wcs.sip_pix2foc(wcs.wcs_world2pix(coo, 1), 1)
                edgemap, _, _ = np.histogram2d(
                    foc[:, 1] - 0.5,
                    foc[:, 0] - 0.5,
                    bins=imsz,
                    range=([[0, imsz[0]], [0, imsz[1]]]),
                )
                print("\tGenerated edge map.")
            except:
                print("\tProducing empty edge map.")
                edgemap = np.zeros(imsz)

        hdu = pyfits.PrimaryHDU(image)
        hdu = fits_header(band, wcs, tranges, exptimes, hdu=hdu)
        hdu.header["NPHOTS"] = len(events["dec"][ix[0]])
        if not binsz:
            hdulist = pyfits.HDUList(
                [hdu, pyfits.ImageHDU(flagmap), pyfits.ImageHDU(edgemap)]
            )
        else:
            hdulist = pyfits.HDUList([hdu])
        print("\t\tWriting {cntfilename}".format(cntfilename=cntfilename))
        hdulist.writeto(cntfilename, overwrite=True)
        print("\t\tCompressing {cntfilename}.gz".format(cntfilename=cntfilename))
        os.system("gzip -f {cntfilename}".format(cntfilename=cntfilename))
        make_qa_plots(cntfilename + ".gz")
    return


def make_photometry(
    eclipse, band, rerun=False, data_directory="/Volumes/BigDataDisk/gPhotonData/GTDS"
):
    photonfile = "{data_directory}/e{eclipse}/e{eclipse}-{b}d.h5".format(
        data_directory=data_directory, eclipse=eclipse, b="n" if band=="NUV" else "f"
    )
    if os.path.exists(os.path.dirname(photonfile) + "/No{band}".format(band=band)):
        print("No data.")
        return
    cntfilename = photonfile.replace(".h5", "-cnt.fits.gz")
    photomfile = cntfilename.replace("-cnt.fits.gz", "-photom.csv")
    if os.path.exists(photomfile) and not rerun:
        print("{f} already exists.".format(f=photomfile))
        return
    cntmap, flagmap, edgemap, wcs, tranges, exptimes = read_image(cntfilename)
    if not cntmap.max():
        print('Image contains nothing.')
        pathlib.Path(os.path.dirname(photonfile) + "/No{band}".format(band=band)
                                ).touch()
        return

    trange, exptime = tranges[0], exptimes[0]
    if exptime < 600:
        print("Skipping low exposure time visit.")
        pathlib.Path("{path}/LowExpt".format(path=os.path.dirname(photonfile))).touch()
        return
    movmap, _, _, _, tranges, exptimes = read_image(cntfilename.replace("-cnt", "-mov"))
    daofind = DAOStarFinder(fwhm=5, threshold=0.01)
    sources = daofind(cntmap / exptime)
    try:
        print(f'Located {len(sources)} sources.')
    except TypeError:
        print('Image contains no sources.')
        pathlib.Path(os.path.dirname(photonfile) + "/No{band}".format(band=band)
                                        ).touch()
        return
    positions = (sources["xcentroid"], sources["ycentroid"])
    apertures = CircularAperture(positions, r=8.533333333333326)
    phot_table = aperture_photometry(cntmap, apertures)
    flag_table = aperture_photometry(flagmap, apertures)
    edge_table = aperture_photometry(edgemap, apertures)

    phot_visit = sources.to_pandas()
    phot_visit["xcenter"] = phot_table.to_pandas().xcenter.tolist()
    phot_visit["ycenter"] = phot_table.to_pandas().ycenter.tolist()
    phot_visit["aperture_sum"] = phot_table.to_pandas().aperture_sum.tolist()
    phot_visit["aperture_sum_mask"] = flag_table.to_pandas().aperture_sum.tolist()
    phot_visit["aperture_sum_edge"] = edge_table.to_pandas().aperture_sum.tolist()
    phot_visit["ra"] = [
        wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()[0]
        for pos in apertures.positions
    ]
    phot_visit["dec"] = [
        wcs.wcs_pix2world([pos], 1, ra_dec_order=True)[0].tolist()[1]
        for pos in apertures.positions
    ]

    for i, frame in enumerate(movmap):
        mc.print_inline("Extracting photometry from frame #{i}".format(i=i))
        phot_visit["aperture_sum_{i}".format(i=i)] = (
            aperture_photometry(frame, apertures).to_pandas()["aperture_sum"].tolist()
        )
    print("Writing data to {f}".format(f=photomfile))
    phot_visit.to_csv(photomfile, index=False)
    pd.DataFrame(
        {
            "expt": exptimes,
            "t0": np.array(tranges)[:, 0].tolist(),
            "t1": np.array(tranges)[:, 1].tolist(),
        }
    ).to_csv(cntfilename.replace("-cnt.fits.gz", "-exptime.csv"), index=False)

    return


def isvar(cps, cps_err, exptime, binsz=120, band = 'NUV'):
    cps_10p_rolloff = {'NUV':311, 'FUV':109}
    # AD eliminates a lot of true negatives, but includes a lot of
    #  false positives, maybe due to the low number of data points
    ad = stats.anderson(cps)
    tix = np.where(exptime[1:-1]/120>0.5) # avoid weirdly low expt bins
    return ((max(cps[1:-1][tix] - 3 * cps_err[1:-1][tix]) > 
             min(cps[1:-1][tix] + 3 * cps_err[1:-1][tix])) &
            (ad.statistic > ad.critical_values[2]) & # 5% significance level
            (cps < 2 * cps_10p_rolloff[band]).all()) # avoid very bright stars

def screen_variables(
    eclipse,
    band,
    skip_static=True,
    data_directory="/Volumes/BigDataDisk/gPhotonData/GTDS",
    rerun=False,plot=False,
    run_gaperture=False,refine=True,
):
    obstype,_,nlegs = obstype_from_eclipse(eclipse)
    if ((obstype in ['AIS', 'NoData', 'GII', 'CAI', 'Unknown']) or (nlegs>0)):
        return []
    photonfile = "{data_directory}/e{eclipse}/e{eclipse}-{b}d.h5".format(
        data_directory=data_directory, eclipse=eclipse, b="n" if band=="NUV" else "f"
    )
    if not os.path.exists(os.path.dirname(photonfile)):
        print("No data directory for e{eclipse}.".format(eclipse=eclipse))
        return []
    cntfilename = photonfile.replace(".h5", "-cnt.fits.gz")
    photomfile = cntfilename.replace("-cnt.fits.gz", "-photom.csv")
    try:
        phot = pd.read_csv(photomfile,index_col='id')
        expt = pd.read_csv(photomfile.replace("photom", "exptime"))
    except FileNotFoundError:
        print("\tPhotometry files not available.")
        return []
    if (expt.expt.values/120).mean()<0.5 or (expt.expt.values<0).any():
        print("\tSkipping visit for exposure time weirdness.")
        return []
    varix = []
    for index in phot.index.values:
        if ((phot.loc[index].aperture_sum_mask!=0) |
            (phot.loc[index].aperture_sum_edge!=0)):
            continue
        cnt = phot.loc[index][
            ["aperture_sum_{i}".format(i=i) for i in np.arange(len(expt))]
        ].values
        cps = cnt / expt.expt.values.flatten()
        cps_err = np.sqrt(cnt) / expt.expt.values.flatten()
        if not isvar(cps, cps_err, expt.expt.values, band=band) and skip_static:
            continue
        varix+=[index]
    if not len(varix):
        return varix
    x_,y_ = (phot.loc[varix].xcenter.values,
             phot.loc[varix].ycenter.values)
    X = np.stack((x_,y_),axis=1)
    db = DBSCAN(eps=40,min_samples=1).fit(X) # 40 pixels = 1 arcmin
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    if plot:
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(unique_labels))]
        plt.figure()
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=14)
            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor='k', markersize=6)
        plt.xlim([phot.xcenter.min(),phot.xcenter.max()])
        plt.ylim([phot.ycenter.min(),phot.ycenter.max()])
        plt.show()
        plt.title('e{e}: {d} clusters ({m} valid, {n} noise)'.format(
                             e=eclipse,d=n_clusters_,m=len(varix),n=n_noise_))
    if refine:
        varix_=[]
        for lbl in set(labels):
            dbix = np.where(labels==lbl)[0]
            if len(dbix)>2: # cluster variables are presumed artifacts
                continue
            elif len(dbix)>1: # small cluster -- use only the brightest
                mix = np.argmax(
                    phot.loc[np.array(varix)[dbix]].aperture_sum.max())
                varix_+=[np.array(varix)[dbix][mix]]
            else:
                varix_+=np.array(varix)[dbix].tolist()
    #print('{n} reduced to {m}'.format(n=len(varix),m=len(varix_)))
        if plot:
            plt.title('e{e}: {d} clusters ({m} valid, {n} noise)'.format(
                        e=eclipse,d=n_clusters_,m=len(varix_),n=n_noise_))
        return varix_
    return varix

def mine_variables(eclipse,band='NUV',
        data_directory = '/home/ubuntu/datadir'):
    photonfile = f'{data_directory}/e{eclipse}/e{eclipse}-{band.lower()[0]}d.h5'
    cmd = f"aws s3 cp s3://gfcat-test/data/e{eclipse}/e{eclipse}-{band.lower()[0]}d-photom.csv {data_directory}/e{eclipse}/"
    #print(cmd)
    os.system(cmd)
    if not os.path.exists(f'{data_directory}/e{eclipse}/e{eclipse}-{band.lower()[0]}d-photom.csv'):
        return
    os.system(cmd.replace('photom','exptime'))
    varix = screen_variables(eclipse,'NUV',data_directory=data_directory)
    print(f'Variables: {varix}')

    if len(varix):
        cmd = f"aws s3 cp s3://gfcat-test/data/e{eclipse}/e{eclipse}-{band.lower()[0]}d-cnt.fits.gz {data_directory}/e{eclipse}/"
        #print(cmd)
        os.system(cmd)
        gapdir = f"{data_directory}/e{eclipse}/gap/"
        if not os.path.exists(gapdir):
            print(f"Creating {gapdir}")
            os.makedirs(gapdir)
        cmd = f"aws s3 cp s3://gfcat-test/data/e{eclipse}/gap/ {data_directory}/e{eclipse}/gap/. --recursive"
        #print(cmd)
        os.system(cmd)
    
    for sourceid in varix:
        print(f"\nProcessing {sourceid}")
        _ = make_gap_qa_plots(eclipse, sourceid, imfile=None,
                                        data_directory=data_directory)
    print("")
    ### CLEANUP
    if len(varix):
        cmd = f"aws s3 sync {data_directory} s3://gfcat-test/data/ --quiet"
        print('Moving everything in {data_directory} to s3 bucket.')
        print(f'\tmove: {cmd}')
        os.system(cmd)
    print(f'Emptying {data_directory}/e{eclipse}/')
    shutil.rmtree(f'{data_directory}/e{eclipse}/')
    return 

def test_photometry(
    eclipse,
    band,
    nsamples,
    random_state=None,
    data_directory="/Volumes/BigDataDisk/gPhotonData/GTDS",
):
    photonfile = "{data_directory}/e{eclipse}/e{eclipse}-{b}d.h5".format(
        data_directory=data_directory, eclipse=eclipse, b="n" if band=="NUV" else "f"
    )
    cntfilename = photonfile.replace(".h5", "-cnt.fits.gz")
    photomfile = cntfilename.replace("-cnt.fits.gz", "-photom.csv")
    try:
        phot = pd.read_csv(photomfile,index_col='id')
        expt = pd.read_csv(photomfile.replace("photom", "exptime"))
    except FileNotFoundError:
        print("Photometry files not available.")
        return
    qadir = os.path.dirname(photonfile) + "/" + "qa"
    if not os.path.exists(qadir):
        os.makedirs(qadir)
    samples = (
        phot.loc[phot.aperture_sum_mask == 0]
        .loc[phot.aperture_sum_edge == 0]
        .loc[phot.aperture_sum > 100]
        .sample(nsamples, random_state=random_state)
    )
    plt.ioff()
    for n in samples.id.values:
        cnt = phot.iloc[n][
            ["aperture_sum_{i}".format(i=i) for i in np.arange(len(expt))]
        ].values
        cps = cnt / expt.expt.values.flatten()
        cps_err = np.sqrt(cnt) / expt.expt.values.flatten()
        skypos = (phot.iloc[n].ra, phot.iloc[n].dec)
        trange = [expt.t0.min(), expt.t1.max()]
        csvfile = (
            qadir + "/" + "{n}_{b}d.csv".format(n=n, b="n" if band=="NUV" else "f")
        )
        out = gAperture(
            band,
            skypos,
            gt.aper2deg(6),
            stepsz=120,
            trange=trange,
            detsize=1.25,
            verbose=2,
            csvfile=csvfile,
        )
        plt.figure()
        plt.errorbar(expt.t0.values, cps, yerr=3 * cps_err, fmt="kx-", label="img")
        plt.errorbar(
            expt.t0.values, out["cps"], yerr=3 * out["cps_err"], fmt="bx-", label="gAp"
        )
        plt.xticks([])
        plt.ylabel("cps")
        plt.title("e{e} ({b})".format(e=eclipse, b=band))
        plt.tight_layout()
        plt.legend()
        plt.savefig(csvfile.replace(".csv", ".png"))
        plt.close("all")
    plt.ion()
    return


def varplot(eclipse, band):
    exptime = pd.read_csv(
        "/Volumes/BigDataDisk/gPhotonData/GTDS/e{e}/e{e}-{b}d-exptime.csv".format(
            e=eclipse, b="n" if band=="NUV" else "f"
        )
    )
    photom = pd.read_csv(
        "/Volumes/BigDataDisk/gPhotonData/GTDS/e{e}/e{e}-{b}d-photom.csv".format(
            e=eclipse, b="n" if band=="NUV" else "f"
        ),
        index_col="id",
    )
    expt = exptime.expt.values
    t0 = np.arange(len(expt))
    plt.figure()
    n, max_cps = 0, 0
    for i in np.arange(len(photom)):
        if photom.iloc[i].aperture_sum_mask or photom.iloc[i].aperture_sum_edge:
            continue
        cnt = np.array(
            [
                photom.iloc[i]["aperture_sum_{j}".format(j=j)]
                for j in np.arange(len(expt))
            ]
        )
        cps = cnt / expt
        if max(cps) > max_cps:
            max_cps = max(cps)
        cps_err = np.sqrt(cnt) / expt
        plt.errorbar(t0, cps, yerr=3 * cps_err, fmt="kx-", alpha=0.1)
        if isvar(cps, cps_err, expt):
            plt.errorbar(t0, cps, yerr=3 * cps_err, fmt="bx-", alpha=0.25)
            n += 1
    print("{n} variables".format(n=n))
    plt.ylim([0.5, max_cps * 1.1])
    plt.semilogy()
    plt.title("{e} (n={n})".format(e=eclipse, n=n))
    return

def query(query,catdbfile='catalog.db'):
    # This will just run any SQL query that you feed it. The table is named "gfcat"
    engine = sql.create_engine(f'sqlite:///{catdbfile}', echo=False)
    out = engine.execute(query).fetchall()
    engine.dispose()
    return out

def conesearch(skypos,match_radius=0.005,catdbfile='catalog.db'):
    # This runs a box search in SQLite and then refines it into a cone
    out = np.array(query(f"SELECT eclipse, id, ra, dec, xcenter, ycenter FROM gfcat WHERE ra >= {skypos[0]-match_radius} AND ra <={skypos[0]+match_radius} AND dec>= {skypos[1]-match_radius} AND dec<={skypos[1]+match_radius}"))
    dist_ix = np.where(angularSeparation(skypos[0],skypos[1],
                                         out[:,2],out[:,3])<=match_radius)
    return pd.DataFrame({'eclipse':np.array(out[:,0][dist_ix],dtype='int16'),
                         'id':np.array(out[:,1][dist_ix],dtype='int16'),
                         'ra':out[:,2][dist_ix],
                         'dec':out[:,3][dist_ix],
                         'xcenter':out[:,4][dist_ix],
                         'ycenter':out[:,5][dist_ix]})
