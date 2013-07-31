#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .lap_tracking import LAPTracker
from .tester import *

import os
import logging

from .utils.color_system import color

def in_ipython():
    try:
        __IPYTHON__
    except NameError:
        return False
    else:
        return True

if in_ipython():
    logformat = '%(asctime)s' + ':'
    logformat += '%(levelname)s' + ':'
    logformat += '%(name)s' + ':'
    # logformat += '%(funcName)s' + ': '
    logformat += ' %(message)s'
else:
    logformat = color('%(asctime)s', 'BLUE') + ':'
    logformat += color('%(levelname)s', 'RED') + ':'
    logformat += color('%(name)s', 'YELLOW') + ':'
    # logformat += color('%(funcName)s', 'GREEN') + ': '
    logformat += color(' %(message)s', 'ENDC')

thisdir = os.path.abspath(os.path.dirname(__file__))
pkgdir = os.path.dirname(thisdir)
samplesdir = os.path.join(pkgdir, 'samples')

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(logformat, "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import warnings
warnings.filterwarnings("ignore")
try:
    import matplotlib
    matplotlib.rcParams['backend'] = "Agg"
except ImportError:
    logger.warning(
        'Matplotlib has not been detected. Some functions may not work.')

