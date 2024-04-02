import csv
from gPhoton.types import Pathlike
import numpy as np
import pandas as pd
import sys

def parse_exposure_time(exposure_file: Pathlike):
    # parse the exposure time files... quickly...
    with open(exposure_file) as data:
        expt_data = csv.DictReader(data) # way faster to parse the file like this
        expt_rows = []
        for row in expt_data:
            expt_rows.append(
                {
                    't0':row['t0'],
                    't1':row['t1'],
                    'expt_eff':float(row['expt'])
                }
            )
    return pd.DataFrame(expt_rows).astype({'t0':'float64','t1':'float64'})

def parse_lightcurve(photometry_file: Pathlike, exposure_file: Pathlike):
    # This is a blunt force way to suppress divide by zero warnings.
    # It's dangerous to suppress warnings. Don't do this.
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    expt = parse_exposure_time(exposure_file)
    with open(photometry_file) as data:
        obs = csv.DictReader(data)
        lightcurves = []
        for row in obs:
            lc = {}
            try:
                lc['counts'] = np.array([float(row[f'aperture_sum_{n}']) for n in range(len(expt['t0']))])
                lc['mask_flags'] = np.array([float(row[f'aperture_sum_flag_{n}']) for n in range(len(expt['t0']))],dtype=bool)
                lc['edge_flags'] = np.array([float(row[f'aperture_sum_edge_{n}']) for n in range(len(expt['t0']))],dtype=bool)
            except ValueError:
                continue # this light curve contains no valid data
            lc['cps'] = np.array(lc['counts'] / expt['expt_eff'])
            lc['cps_err'] = np.array(np.sqrt(lc['counts']) / expt['expt_eff'])
            lc['xcenter'] = float(row['xcenter'])
            lc['ycenter'] = float(row['ycenter'])
            lc['ra'] = float(row['ra'])
            lc['dec'] = float(row['dec'])
            lightcurves+=[lc]
    return lightcurves