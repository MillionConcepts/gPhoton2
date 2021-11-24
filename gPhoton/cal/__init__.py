from io import BytesIO
import os
import sys

import numpy as np
import requests

from gPhoton import cal_dir
# gPhoton imports.
from gPhoton.MCUtils import get_fits_data, get_fits_header, get_tbl_data

#from gPhoton.download import download_with_progress_bar

# Remote repository for GALEX calibration files.
cal_url = 'https://archive.stsci.edu/prepds/gphoton/cal/cal/'



def check_band(band):
    if band not in ['NUV', 'FUV']:
        raise ValueError('Band must be NUV or FUV')
    return band

def check_xy(xy):
    if not xy in ['x', 'y']:
        raise ValueError('xy must be x or y.')
    return xy

def read_data(fn, dim=0):
    path = _os.path.join(cal_dir, fn)
    # Download the file if it doesn't exist locally.
    if not _os.path.exists(path):
        data_url='{b}/{f}'.format(b=cal_url,f=fn)
        fitsdata = download_with_progress_bar(data_url, path)
    if '.fits' in fn:
        data = get_fits_data(path, dim=dim)
        header = get_fits_header(path)
        if isinstance(data, np.recarray):
            for name in data.names:
                data[name] = data[name].byteswap().newbyteorder()
        else:
            data = data.byteswap().newbyteorder()
        return data, header
    elif '.tbl' in fn:
        return get_tbl_data(path)
    else:
        raise ValueError('Unrecognized data type: {ext}'.format(ext=fn[-4:]))

def wiggle(band, xy):
    fn = '{b}_wiggle_{d}.fits'.format(b=check_band(band), d=check_xy(xy))
    return read_data(fn)

def wiggle2():
    """The post-CSP wiggle file."""
    return read_data('WIG2_Sep2010.fits', dim=1)

def avgwalk(band, xy):
    fn = '{b}_avgwalk_{d}.fits'.format(
        b=check_band(band), d=check_xy(xy))
    return read_data(fn)

def walk(band, xy):
    fn = '{b}_walk_{d}.fits'.format(
        b=check_band(band), d=check_xy(xy))
    return read_data(fn)

def walk2():
    """The post-CSP walk file."""
    return read_data('WLK2_Sep2010.fits', dim=1)

def clock2():
    """The post-CSP clock file."""
    return read_data('CLK2_Sep2010.fits', dim=1)

def linearity(band, xy):
    fn = '{b}_NLC_{d}_det2sky.fits'.format(
        b=check_band(band), d=check_xy(xy))
    return read_data(fn)

def addbuffer(fn):
# Adds a 1 pixel buffer around all masked (==0) regions of a map.
    m, h = read_data(fn)
    ix = np.where(m == 0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            try:
                m[ix[0]+i, ix[1]+j] = 0
            except IndexError:
                continue
    return m, h

def flat(band, buffer=False):
    fn = '{b}_flat.fits'.format(b=check_band(band))
    return addbuffer(fn) if buffer else read_data(fn)

def distortion(band, xy, eclipse, raw_stimsep):
    index = ''
    if band == 'NUV':
        if eclipse > 37460:
            if raw_stimsep < 5136.3:
                index = 'a'
            elif raw_stimsep < 5137.25:
                index = 'b'
            else:
                index = 'c'
    fn = '{b}_distortion_cube_d{d}{i}.fits'.format(
        b=check_band(band).lower(), d=check_xy(xy), i=index)
    return read_data(fn)

def offset(xy):
    fn = 'fuv_d{d}_fdttdc_coef_0.tbl'.format(d=check_xy(xy))
    return read_data(fn)

def mask(band, buffer=False):
    fn = '{b}_mask.fits'.format(b=check_band(band))
    return addbuffer(fn) if buffer else read_data(fn)
