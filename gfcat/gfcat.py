import os
import pathlib
import shutil
import sys
import warnings

import pandas as pd

from gfcat.gfcat_utils import (
    obstype_from_eclipse,
    mc,
    calibrate_photons,
    make_images,
    make_photometry,
    download_raw6,
)
from gPhoton.PhotonPipe_3 import photonpipe


# WARNING: TURNING OFF WARNINGS
if not sys.warnoptions:
    warnings.simplefilter("ignore")

bucketname = "gfcat-test"


def recalibrate(
    eclipse,
    band,
    data_directory='data',
    rerun=False,
    retain=False,
    ext = 'parquet'
):
    photonfile = f"{data_directory}/e{eclipse}/e{eclipse}-{band[0].lower()}d.{ext}"

    if not os.path.exists(os.path.dirname(photonfile)):
        print(f"Creating {os.path.dirname(photonfile)}")
        os.makedirs(os.path.dirname(photonfile))

    # AWS: Check for a photometry file, indicating doneness.
    if not rerun:
        cmd = f"aws s3 mv s3://{bucketname}/data/e{eclipse}/e{eclipse}-{band[0].lower()}d-photom.csv {data_directory}/e{eclipse}/"
        print("Checking for doneness...")
        print("\ttry: {cmd}".format(cmd=cmd))
        os.system(cmd)
        if os.path.exists(raw6file.replace("-raw6.fits.gz", "-photom.csv")):
            print("\tPhotometry already exists for this visit / band.")
            if not retain:
                shutil.rmtree("{d}/e{e}/".format(d=data_directory, e=eclipse))
            return  # already processed

    cmd = "aws s3 sync s3://{bn}/data/e{e}/ {d}/e{e}/ --exclude '*{nb}d*'".format(
        bn=bucketname,
        d=data_directory,
        e=eclipse,
        nb="f" if band is "NUV" else "n",
    )
    print("Retrieving existing data from s3 bucket.")
    print("\ttry: {cmd}".format(cmd=cmd))
    os.system(cmd)

    if os.path.exists(
        os.path.dirname(photonfile) + "/No{band}".format(band=band)
    ):
        print(
            "No {band} data for eclipse {eclipse}".format(
                band=band, eclipse=eclipse
            )
        )
        if not retain:
            shutil.rmtree("{d}/e{e}/".format(d=data_directory, e=eclipse))
        return

    if not os.path.exists(os.path.dirname(photonfile)):
        print("Creating {dirname}".format(dirname=os.path.dirname(photonfile)))
        os.makedirs(os.path.dirname(photonfile))

    # First check whether this is an observation type that we care about ---
    #   Currently limiting processing to MIS and DIS visits that have more
    #   than 600 seconds of observation time (e.g. 5x120s bins). This might
    #   miss some dither-mode GII visits, but makes the accounting easy.
    if not os.path.exists(photonfile):
        obstype, rawexpt, nlegs = obstype_from_eclipse(eclipse)
        if (obstype in ["AIS", "NoData", "GII", "CAI", "Unknown"]) or (
            nlegs > 0
        ):
            print("\tSkipping e{eclipse}.".format(eclipse=eclipse))
            pathlib.Path(os.path.dirname(photonfile) + "/NoNUV").touch()
            pathlib.Path(os.path.dirname(photonfile) + "/NoFUV").touch()
            if obstype in ["AIS"]:
                pathlib.Path(
                    os.path.dirname(photonfile)
                    + "/{obstype}".format(obstype=obstype)
                ).touch()
            if not retain:
                shutil.rmtree("{d}/e{e}/".format(d=data_directory, e=eclipse))
            return
        if rawexpt <= 600:
            pathlib.Path(
                os.path.dirname(photonfile)
                + "/{obstype}".format(obstype="LowExpt")
            ).touch()
            if not retain:
                shutil.rmtree("{d}/e{e}/".format(d=data_directory, e=eclipse))
            return

    raw6file = download_raw6(eclipse, band, data_directory=data_directory)
    if not raw6file:
        pathlib.Path(
            os.path.dirname(photonfile) + "/No{band}".format(band=band)
        ).touch()
        if not retain:
            shutil.rmtree("{d}/e{e}/".format(d=data_directory, e=eclipse))
        return

    if not os.path.exists(photonfile):
        try:
            photonpipe(
                raw6file.split(".")[0][:-5], band, raw6file=raw6file, verbose=2
            )
        except:
            print(
                "Error during data processing. (Probably a in the SSD calculation!)"
            )
            pathlib.Path(
                os.path.dirname(photonfile) + "/No{band}".format(band=band)
            ).touch()
            pathlib.Path(
                os.path.dirname(photonfile)
                + "/SSDlinalg_ERROR".format(band=band)
            ).touch()
            ### UGLY dupe of code below
            # Remove the very large calibrated photon files to avoid incurring s3 storage costs.
            print("Deleting the calibrated photon (*.{ext}) files.")
            try:
                os.remove(photonfile)
                os.remove(xcalfilename)
            except:
                pass
            cmd = "aws s3 sync {d} s3://{bn}/data/".format(
                bn=bucketname,
                d=data_directory,
                e=eclipse,
                b="n" if band is "NUV" else "f",
            )
            print("Moving data to s3 bucket.")
            print("\ttry: {cmd}".format(cmd=cmd))
            os.system(cmd)
            # cmd = "rm -rf {d}/.".format(d=data_directory)
            if not retain:
                print("Emptying {d}/e{e}/".format(d=data_directory, e=eclipse))
                shutil.rmtree("{d}/e{e}/".format(d=data_directory, e=eclipse))
            return
    xcalfilename = photonfile.replace(f".{ext}", f"-xcal.{ext}")

    if not os.path.exists(xcalfilename):
        mc.print_inline("Calibrating...")
        try:
            event_data = calibrate_photons(pd.read_parquet(photonfile), band)
            print("Calibrated.    ")
            if len(event_data):
                print(
                    "Writing {xcalfilename}".format(xcalfilename=xcalfilename)
                )
                with pd.HDFStore(xcalfilename) as store:
                    store.append("events", event_data)
                trange = [event_data["t"].min(), event_data["t"].max()]
                if trange[1] - trange[0] == 0:
                    print("\t\tZero exposure time.")
                    pathlib.Path(
                        os.path.dirname(photonfile)
                        + "/No{band}".format(band=band)
                    ).touch()
            else:
                print("\tNo valid photons in observation.")
                pathlib.Path(
                    os.path.dirname(photonfile) + "/No{band}".format(band=band)
                ).touch()
        except MemoryError:
            print(
                "Data file too big to be processed on this machine. Flagging for reprocessing."
            )
            pathlib.Path(
                os.path.dirname(photonfile) + "/No{band}".format(band=band)
            ).touch()
            pathlib.Path(
                os.path.dirname(photonfile) + "/TooBig".format(band=band)
            ).touch()
    else:
        print(
            "{xcalfilename} already exists".format(xcalfilename=xcalfilename)
        )

    make_images(eclipse, band, data_directory=data_directory)

    make_photometry(eclipse, band, rerun=rerun, data_directory=data_directory)

    # Remove the very large calibrated photon files to avoid incurring s3 storage costs.
    print("Deleting the calibrated photon (*.{ext}) files.")
    try:
        os.remove(photonfile)
        os.remove(xcalfilename)
    except:
        pass

    # Can run this locally to limit the amount of data to retrieve from AWS.
    # make_lightcurves(eclipse,band,data_directory=data_directory)

    cmd = "aws s3 sync {d} s3://{bn}/data/".format(
        bn=bucketname,
        d=data_directory,
        e=eclipse,
        b="n" if band is "NUV" else "f",
    )
    print("Moving data to s3 bucket.")
    print("\ttry: {cmd}".format(cmd=cmd))
    os.system(cmd)
    if not retain:
        # cmd = "rm -rf {d}/.".format(d=data_directory)
        print("Emptying {d}/e{e}/".format(d=data_directory, e=eclipse))
        shutil.rmtree("{d}/e{e}/".format(d=data_directory, e=eclipse))

    return
