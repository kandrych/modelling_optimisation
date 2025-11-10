
from __future__ import annotations

import argparse
import json
import math
import os
os.environ["MPLBACKEND"] = "Agg"   # must be set before importing matplotlib
import matplotlib
matplotlib.use("Agg", force=True)

import shutil
import subprocess
import sys
import textwrap
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple


sys.path.append(os.path.abspath(".."))  # parent of current working dir
#import lib.Katya_func as kf

import lib.obriy_general as obg
import lib.obriy_sed as obs
import lib.obriy_interferometry as obi
import lib.obriy_polarimetry as obp
import lib.obriy_mcfost as obm


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "DejaVu Serif",         # shipped with matplotlib
    "Liberation Serif",
    "Nimbus Roman",
    "TeX Gyre Termes",
    "Times",                # generic fallback
]



work_root = '/fred/oz061/kandrych/smac/test_new_features/polarimetric_comparison/'
work_root = Path(work_root)
work_root.mkdir(parents=True, exist_ok=True)
work_root = str(work_root)


#polarimetric observations
pdi_folder_v = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/V_band/'
pdi_file_v = 'IRAS08544-4431_dc_notnorm_V_PI_corr_tel+unres.fits'
pdi_v= obp.Loadimage(pdi_folder_v,pdi_file_v)
obp.plot_polarimetric_image(pdi_v, 3.6, title='IRAS08544-4431 V-band PI', save=work_root+'/pi_v_band.png', image_scale='asinh', roi_half_size=50)

pdi_folder_i = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/I_band/'
pdi_file_i = 'IRAS08544-4431_dc_notnorm_I_PI_corr_tel+unres.fits'
pdi_i= obp.Loadimage(pdi_folder_i,pdi_file_i)
obp.plot_polarimetric_image(pdi_i, 3.6, title='IRAS08544-4431 I-band PI', save=work_root+'/pi_i_band.png', image_scale='asinh', roi_half_size=50)

pdi_folder_h = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/H_band/'
pdi_file_h = 'iras08544-4431_calib_H_PI_corr_tel+unres.fits'
pdi_h= obp.Loadimage(pdi_folder_h,pdi_file_h)
obp.plot_polarimetric_image(pdi_h, 12.27, title='IRAS08544-4431 H-band PI', save=work_root+'/pi_h_band.png', image_scale='asinh', roi_half_size=30)

file_psf='iras08544-4431_calib_H_I_meancombined.fits'
psf_h=obp.Loadimage(pdi_folder_h,file_psf)
obp.plot_polarimetric_image(psf_h, 12.27, title='IRAS08544-4431 H-band PSF', save=work_root+'/psf_h_band.png', image_scale='asinh', roi_half_size=30)



metrics = obp.full_image_metrics_noshift(
    pdi_v, pdi_i,
    normalize="zscore",          # good default for morphology
    ssim_win=11,                 # 7–15 is typical
    return_pixel_chi2=True
)
print(metrics)

obp.plot_polarimetric_image(metrics["ssim_image"], 3.6, title='ssim', save=work_root+'/ssim_image.png', image_scale='linear', roi_half_size=50)

