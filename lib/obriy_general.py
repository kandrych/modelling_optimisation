import distroi
import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict, Optional, Union, Any, List

from distroi.auxiliary import constants
from distroi.data import image
from distroi.data import sed
from distroi.model.geom_comp import geom_comp
from distroi.auxiliary import select_data_oifits
from distroi.data.oi_container import OIContainer

import os
import matplotlib.pyplot as plt

from astropy.io import fits
import astropy.units as u
from IPython.display import display
import glob
import subprocess
from skimage.transform import rescale, resize, downscale_local_mean
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft, AiryDisk2DKernel
import fnmatch
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import random
import pymcfost as mcfost

import astropy
from astropy import units as u
import astropy.units.quantity
from astropy.io import fits
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
from textwrap import wrap
import scipy.ndimage as ndimage
from matplotlib.gridspec import GridSpec
from matplotlib import colors


from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from functools import wraps
import cv2

import shutil

from pathlib import Path





constants.set_matplotlib_params()  # set project matplotlib parameters


###################################
# GENERAL
###################################


def pick_output(fn, idx=0, cast=float):
    """
    Calls fn(*args, **kwargs), returns the element at idx.
    Optionally casts to a scalar type (default: float).
    """
    @wraps(fn)
    def _wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        try:
            val = out[idx]
        except TypeError:
            raise TypeError(f"{fn.__name__} did not return an indexable object, got {type(out)}")
        except IndexError:
            raise IndexError(f"Index {idx} out of range for output {out!r}")
        return cast(val) if cast is not None else val
    return _wrapped


def mas2au(angle_mas: float, distance_pc: float) -> float:
    """
    Convert angular size in milliarcseconds (mas) to linear size in astronomical units (au).

    Parameters
    ----------
    angle_mas : float
        Angular size in milliarcseconds (mas).
    distance_pc : float
        Distance to the object in parsecs (pc).

    Returns
    -------
    size_au : float
        Linear size in astronomical units (au).
    """
    # 1 mas = 1e-3 arcsec, and 1 arcsec at 1 pc = 1 au
    size_au = angle_mas * 1e-3 * distance_pc
    
    #analogous calculation using astropy units
    #size_au = (angle_mas*u.mas).to(u.rad)*(distance_pc)*u.parsec.to(u.au) 

    return size_au



def write_header(path: str) -> None:
    """
    Write the header line to the output file.
    Parameters
    ----------
    path : str
        Path to the output file.
    Returns
    -------
    None
    """
    with open(path, "w") as f:
        f.write(
            f"{'instrument':<10} {'band':<5} {'mode':<13} {'background':<10} {'vistype':<8} "
            f"{'wave_min':<10} {'wave_max':<10} {'chi2':<12} {'chi2_red':<12} {'num_points':<12}\n"
        )

def record(
    path: str,
    instrument: str,
    band: str,
    mode: str,
    background: float,
    vistype: str,
    chi2: float,
    chi2_red: float,
    num_points: int = 0,
    wave_lims: tuple[float, float] | None = None
) -> None:
    """
    Record the fit results (given as parameters) to a file.

    Parameters
    ----------
    path : str
        Path to the output file.
    instrument : str
        Instrument name.
    band : str
        Observing band.
    mode : str
        Observing mode.
    background : float
        Background flux fraction.
    vistype : {'vis2', 'vis', 'fcorr'}
        Type of visibility.
    chi2 : float
        Chi2 value.
    chi2_red : float
        Reduced chi2 value.
    num_points : int, optional
        Number of data points used in the fit. Default is 0.
    wave_lims : tuple of float, optional
        Wavelength limits (min, max) in micrometers. Default is None.
    Returns
    -------
    None
    """
    wmin, wmax = ("", "") if wave_lims is None else (f"{wave_lims[0]:.5g}", f"{wave_lims[1]:.5g}")
    with open(path, "a") as f:
        f.write(
            f"{instrument:<10} {band:<5} {mode:<13} {background:<10} {vistype:<8} "
            f"{wmin:<10} {wmax:<10} {chi2:<12.6g} {chi2_red:<12.6g} {num_points:<10}\n"
        )

