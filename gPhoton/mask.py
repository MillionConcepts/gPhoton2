import sys
import os
import shutil
import gc
from quickbin import bin2d
import pandas as pd
import numpy as np
import pyarrow.parquet as parquet

def make_mask_savepaths(ctx):
    e = str(ctx.eclipse).zfill(5)
    b = 'n' if ctx.band == "NUV" else 'f'
    savepaths = {'cmask': f"e{e}-{b}d-cmask.bin", 'hmask': f"e{e}-{b}d-hmask.bin" }
    return savepaths

def make_masks_per_eclipse(photon_file, nbins, savepaths):
    # use existing photonlists to make individual hotspot and coldspot masks
    # per eclipse & band
    # just using first photonpath / leg
    if os.path.exists(photon_file):
        try:
            print("reading photonlist")
            nf = photonlist.read_table(photon_file, columns=['col', 'row', 'ra', 'dec', 't']).to_pandas()
            # for reading row groups
            #photonlist = parquet.ParquetFile(photon_file)
            #nf = photonlist.read_row_groups([0],columns=['col', 'row', 'ra', 'dec', 't']).to_pandas()
            nf['row_rnd'] = nf['row'].round().astype(int)
            nf['col_rnd'] = nf['col'].round().astype(int)
            print("filtering photonlist")
            # filtering photonlist for on detector because read_row_groups can't
            nf = nf[(nf['col_rnd'] <= 800) & (nf['row_rnd'] <= 800)
                    & (nf['ra'] != 0) & (nf['dec'] != 0)
                    & (nf['col_rnd'] >= 0) & (nf['row_rnd'] >= 0)]
            mask = pd.notna(nf['ra'])
            nf = nf[mask]
            print("calculating expt & adding edge points")
            # rough approx not accounting for dead time
            expt = nf.iloc[len(nf) - 1]['t'] - nf.iloc[0]['t']
            # adding edge points, have 'real' values from photonlist to
            # not mess with the stats too much
            ra = nf.iloc[0]['ra']
            dec = nf.iloc[0]['dec']
            t = nf.iloc[0]['t']
            edge_points = pd.DataFrame({
                'col': [0, 0, 800, 800],
                'row': [0, 800, 0, 800],
                'ra': [ra,ra,ra,ra],
                'dec': [dec,dec,dec,dec],
                't': [t,t,t,t]
            })
            nf = pd.concat([nf, edge_points], ignore_index=True)

            print("quickbinning")
            ra_dict = bin2d(nf['col'], nf['row'], nf['ra'], nbins, op=['std', 'count'])
            dec_stdev = bin2d(nf['col'], nf['row'], nf['dec'], 'std', nbins)
            count = ra_dict['count'] / expt

            print("masking binned data")
            density_mask = ra_dict['count'] >= .9
            disp_mask = ra_dict['std'] + dec_stdev > .014
            dark_mask = count <= .008

            print("making & saving new masks")
            hmask = np.ones(count.shape, dtype=bool)
            cmask = np.ones(count.shape, dtype=bool)
            hmask[density_mask & disp_mask] = 0
            hmask.tofile(savepaths['hmask'])
            cmask[dark_mask] = 0
            cmask.tofile(savepaths['cmask'])
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            print(ex)
            print(f"failed {eclipse}")
    else:
        print("fail!")

        print("cleaning up")
        gc.collect()

    return
