"""
General utils
"""

import contextlib
import glob

import torch
import numpy as np
import pandas as pd
import pkg_resources as pkg
import torch
import torchvision
import yaml
import logging
import os


def set_logging(name=None, verbose=True):
    # set level and returns logger
    rank = int(os.get_env('RANK', -1))
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO if (verbose and rank in (-1, 0)) else logging.warning())
    return logging.getLogger(name)


LOGGER = set_logging(__name__)

