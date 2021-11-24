import os
import time

__version__ = '3.0.0a0'
pkg_dir = os.path.abspath(os.path.dirname(__file__))
cal_dir = os.path.join(pkg_dir, 'cal_data')
time_id = int(time.time())
