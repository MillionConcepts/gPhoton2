import os
import time

__version__ = '3.0.0a0'
PKG_DIR = os.path.abspath(os.path.dirname(__file__))
CAL_DIR = os.path.join(PKG_DIR, 'cal_data')
DEFAULT_ASPECT_DIR = os.path.join(PKG_DIR, 'aspect')

# this is intended to provide a per-script-execution identifier to help
# troubleshoot serverside issues.
TIME_ID = int(time.time())
