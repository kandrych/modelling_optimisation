"""
SMAC3 + Dask(SLURM) + Multi‑Fidelity skeleton 

- Cluster launcher (Dask + dask_jobqueue.SLURMCluster)
- ConfigSpace with conditional ("flexible") hyperparameters
- Multi-fidelity mapping (budget -> photon packets / resolution / stage F0→F3)
- Objective wrapper that shells out to MCFOST and returns a scalar loss
- SMAC3 MultiFidelityFacade orchestration (parallel, async-friendly)

"""
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
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt



# --- Cluster imports ---
# pip install dask distributed dask-jobqueue
from distributed import Client
from dask_jobqueue import SLURMCluster

# --- SMAC3 / ConfigSpace ---
# pip install smac configspace
from smac import Scenario
from smac.facade.multi_fidelity_facade import MultiFidelityFacade
from smac.runhistory import TrialValue
from smac import HyperparameterOptimizationFacade as HPO  # for callbacks, utils

from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical, Configuration
from ConfigSpace.conditions import InCondition, EqualsCondition

from smac.runhistory.dataclasses import TrialInfo, TrialValue




import distroi

sys.path.append(os.path.abspath(".."))  # parent of current working dir
#import lib.Katya_func as kf

import lib.obriy_general as obg
import lib.obriy_sed as obs
import lib.obriy_interferometry as obi
import lib.obriy_polarimetry as obp
import lib.obriy_mcfost as obm
import lib.obriy_after_optimisation as oao
import lib.obriy_alma as oba


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "DejaVu Serif",         # shipped with matplotlib
    "Liberation Serif",
    "Nimbus Roman",
    "TeX Gyre Termes",
    "Times",                # generic fallback
]


# -----------------------------------------------------------------------------
# Cluster setup
# -----------------------------------------------------------------------------

def start_cluster(n_workers: int, processes_per_worker: int, use_slurm: bool) -> Client:
    """
    NOT REALLY WORKING
    
    Start a Dask cluster. For quick local testing, set use_slurm=False and
    rely on default LocalCluster via `Client()`.

    Start a Dask cluster. For SLURM, we:
      - load the same modules on the worker nodes
      - activate the same venv
      - export all runtime env vars (MCFOST, PATH, OMP, PYTHONPATH)
      - write logs per-job so you can debug startup

    
    """
    if not use_slurm:
        client = Client(n_workers=n_workers, threads_per_worker=processes_per_worker)  # LocalCluster default
        print("[cluster] Local Dask cluster started:", client)
        return client

    work_root = Path(os.environ.get("SMAC_WORK_ROOT", ".")).resolve()
    log_dir   = work_root / "dask-logs"
    tmp_dir   = work_root / "dask-tmp"
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Bind scheduler to a host/interface visible to workers on the compute node
    # (when driver runs under SLURM, hostname is the compute node)
    scheduler_host = socket.gethostname()

    cluster = SLURMCluster(
        queue="normal",               # partition
        processes=1,
        cores=processes_per_worker,   # cpu per worker process
        memory="10GB",
        walltime="02:00:00",
        local_directory=str(tmp_dir),
        job_extra_directives=[
            "--account=oz061",
            f"--output={log_dir}/dask.%x.%j.out",
            f"--error={log_dir}/dask.%x.%j.err",
            "--ntasks=1",
            f"--cpus-per-task={processes_per_worker}",
            "--mem=10G",
            "--time=02:00:00",
        ],
        job_script_prologue=[
            "set -euo pipefail",
            "module --force purge",
            "module load python-scientific/3.11.5-foss-2023b",
            "source /home/kandrych/venvs/obriy311/bin/activate",
            "export OMP_NUM_THREADS=1",
            'echo "PYTHON = $(which python)"',
        ],
        python=sys.executable,
        scheduler_options={
            "host": scheduler_host,        # avoid binding to an external/unroutable IP
            "dashboard_address": ":0",     # disable dashboard port hard-coding
        },
    )

    # Persist the exact worker script we’re submitting
    (log_dir / "dask_worker_template.sh").write_text(cluster.job_script())

    cluster.scale(n_workers)
    client = Client(cluster)
    client.wait_for_workers(n_workers, timeout="180s")
    print("[cluster] up:", client)
    
    return client

# -----------------------------------------------------------------------------
# ConfigSpace (flexible / conditional hyperparameters)
# -----------------------------------------------------------------------------

def build_configspace(config_file: str) -> ConfigurationSpace:

    #cs = ConfigurationSpace.from_yaml('/fred/oz061/kandrych/modelling_optimisation/config/space.yaml')#("/Users/katerynaandrych/Work/lin/python scripts/modelling_optimisation/config/space.yaml")
    cs = ConfigurationSpace.from_yaml(config_file)

    # cs = ConfigurationSpace()

    # # --- Example core physical knobs (change here parameters) ---
    # alpha = Float("alpha_viscosity", (0.0000001, 0.1), default=0.01, log=True)
    # Rc = Float("zone_1_Rc", (7.3, 50.0), default=10.0)
    # surface_density_exp = Float("zone_1_surface_density_exp", (-1.5, 1.5), default=1.0)
    # gamma_exp = Float("zone_1_gamma_exp", (-2.0, -0.2), default=-1.0)   

    # # Flexible dust law choice unlocks different parameters
    # # dust_law = Categorical("dust_law", ["MRN", "powerlaw"], default="MRN")
   
    # # q = Float("q", (-4.5, -2.0), default=-3.5)  # only if powerlaw

    # cs.add([alpha, Rc, surface_density_exp, gamma_exp])


    # cs.to_yaml("/Users/katerynaandrych/Work/lin/python scripts/modelling_optimisation/config/space.yaml")
    # # --- Optional conditions (uncomment if you want them active) ---
    # # Only use a_min/a_max when dust_law == "MRN"
    # # Only use q when dust_law == "powerlaw"
    # # cs.add([
    # #     EqualsCondition(a_min, dust_law, "MRN"),
    # #     EqualsCondition(a_max, dust_law, "MRN"),
    # #     EqualsCondition(q,      dust_law, "powerlaw"),
    # # ])

    # # --- Optional forbiddens example ---
    # # forb = ForbiddenAndConjunction(
    # #     ForbiddenEqualsClause(dust_law, "powerlaw"),
    # #     ...
    # # )
    # # cs.add_forbidden_clause(forb)

    return cs

# -----------------------------------------------------------------------------
# Multi-fidelity mapping
# -----------------------------------------------------------------------------

def map_budget_to_fidelity(budget: float) -> Dict[str, Any]:
    """
    Map the continuous budget in [min_budget, max_budget] to discrete fidelity settings.

    Example strategy:
    - Stage ladder F0→F3 via budget thresholds
    - Photon packets and image resolution scale with budget - TO DO

    """
    # Example budget range: min=0.25, max=3.0 (set in Scenario)
    stage = "F0"
    if budget >= 1.0:
        stage = "F1"
    if budget >= 2.0:
        stage = "F2"
    if budget >= 3.0:
        stage = "F3"
    if budget >= 4.0:
        stage = "F4"
    if budget >= 5.0:
        stage = "F5"
    if budget >= 6.0:
        stage = "F6"
    if budget >= 7.0:
        stage = "F7"
    if budget >= 8.0:
        stage = "F8"
  

    if budget >= 9.0:
        stage = "F9"
    if budget >= 10.0:
        stage = "F10"
    if budget >= 11.0:
        stage = "F11"
    
   
    if budget >= 14.0:
        stage = "F14"
    if budget >= 15.0:
        stage = "F15"
    if budget >= 16.0:
        stage = "F16"

    # scale photons with budget
    # nbr_photons_eq_th = int(1.28e5 * (10**budget))
    # nbr_photons_lambda= int(1.28e3 * (10**budget))
    # nbr_photons_image= int(1.28e4 * (10**budget))


    # image resolution mapping (example)
    if stage == "F0":
        img_res = 10 #mas/pixel
        products = ["sed"]
    elif stage == "F1":
        img_res = 2 #mas/pixel
        products = ["sed", "vis2_1perband"]
    elif stage == "F2":
        img_res = 2
        products = ["sed", "vis2_1perband", "pdi_V", "pdi_I", "pdi_H"]
    elif stage == "F3":
        img_res = 2
        products = ["sed", "vis2_1perband", "pdi_V", "pdi_I", "pdi_H","alma"]
    elif stage == "F4":
        img_res = 2
        products = ["sed", "vis2_chromatic", "pdi_V", "pdi_I", "pdi_H", "alma"]

    elif stage == "F5":
        img_res = 10 #mas/pixel
        products = ["sed"]
    elif stage == "F6":
        img_res = 2 #mas/pixel
        products = ["sed", "vis2_1perband"]
    elif stage == "F7":
        img_res = 2
        products = ["sed", "vis2_chromatic"]


    elif stage == "F9":
        img_res = 2
        products = ["sed"]
    elif stage == "F10":
        img_res = 2
        products = ["sed", "alma"]
    elif stage == "F11":
        img_res = 2
        products = ["sed", "pdi_V", "pdi_I", "pdi_H", "alma"]


    elif stage == "F14":
        img_res = 2
        products = ["sed"]
    elif stage == "F15":
        img_res = 2
        products = ["sed","pdi_V", "pdi_I", "pdi_H"]
    
    elif stage == "F16": #created for AR Pup test
        img_res = 2
        products = ["pdi_V", "pdi_I", "pdi_H"]
    else:
        raise ValueError(f"Unknown stage for budget {budget}: {stage}")

    return {
        "stage": stage,
        # "nbr_photons_eq_th": nbr_photons_eq_th,
        # "nbr_photons_lambda": nbr_photons_lambda,
        # "nbr_photons_image": nbr_photons_image,
        "image_res": img_res,
        "products": products,
    }


# -----------------------------------------------------------------------------
# Objective callable for SMAC (supports multi-fidelity via `budget` argument)
# -----------------------------------------------------------------------------


def load_data(data_root: str, work_root: str, fidelity_products: list) -> Dict[str, Any]:
    
    figdir = Path(work_root)/"data_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figdir=str(figdir)
    alma_spec = None
    alma_header = None
    alma_aperture_radius_mas = None

    #filename of SED catalogue data file
    if data_root =='demo_mac':
        if  "sed" in fidelity_products: 
            data_filename = '/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/SED/IRAS08544-4431.phot'
            data_wave, data_flux, data_err = obs.load_sed_data(data_filename)

        if  "vis2_1perband" in fidelity_products:
            # PIONIER data
            data_dir_pionier, data_file_pionier = "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/PIONIER/", "*.fits"
            container_data_pionier = distroi.read_oi_container_from_oifits(data_dir_pionier, data_file_pionier, wave_lims=(1.63, 1.64))

            # GRAVITY data
            data_dir_gravity, data_file_gravity = "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/GRAVITY/", "*1.fits"
            container_data_gravity = distroi.read_oi_container_from_oifits(data_dir_gravity, data_file_gravity, wave_lims=(2.199, 2.201))

            # VLTI/MATISSE L-band data
            data_dir_matisse_l, data_file_matisse_l = "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/MATISSE_L/", "*.fits"
            container_data_matisse_l = distroi.read_oi_container_from_oifits(data_dir_matisse_l, data_file_matisse_l, wave_lims=(3.48, 3.52))

            # VLTI/MATISSE N-band data
            data_dir_matisse_n, data_file_matisse_n = "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/MATISSE_N/", "*.fits"
            container_data_matisse_n = distroi.read_oi_container_from_oifits(data_dir_matisse_n, data_file_matisse_n, wave_lims=(9.9, 10.10), fcorr=True)
        
        if  "vis2_chromatic" in fidelity_products:
            # PIONIER data
            data_dir_pionier, data_file_pionier = "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/PIONIER/", "*.fits"
            container_data_pionier = distroi.read_oi_container_from_oifits(data_dir_pionier, data_file_pionier)

            # GRAVITY data
            data_dir_gravity, data_file_gravity = "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/GRAVITY/", "*1.fits"
            container_data_gravity = distroi.read_oi_container_from_oifits(data_dir_gravity, data_file_gravity)

            # VLTI/MATISSE L-band data
            data_dir_matisse_l, data_file_matisse_l = "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/MATISSE_L/", "*.fits"
            container_data_matisse_l = distroi.read_oi_container_from_oifits(data_dir_matisse_l, data_file_matisse_l, wave_lims=(2.95, 3.95), v2lim=1e-8)

            # VLTI/MATISSE N-band data
            data_dir_matisse_n, data_file_matisse_n = "/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/MATISSE_N/", "*.fits"
            container_data_matisse_n = distroi.read_oi_container_from_oifits(data_dir_matisse_n, data_file_matisse_n, fcorr=True, wave_lims=(8.0, 13.0), v2lim=1e-8)

        if  "pdi_V" in fidelity_products or "pdi_I" in fidelity_products or "pdi_H" in fidelity_products:
            #real PSF from observations
            star_psf='HD83878'
            figfolder_psf='/Users/katerynaandrych/Work/lin/PhD/SPHERE_reduction_data/paper2/mean_combined/'+star_psf+'/'

        if  "pdi_V" in fidelity_products:    
            file_psf=star_psf+'_'+'V'+'_'+'I'+'_meancombined.fits'
            psf_v=obp.Loadimage(figfolder_psf,file_psf)
            obp.plot_polarimetric_image(psf_v, 3.6, title='IRAS08544-4431 V-band PSF', save=figdir+'/psf_v_band.png', image_scale='asinh', roi_half_size=30)
            #polarimetric observations
            pdi_folder_v = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/V_band/'
            pdi_file_v = 'IRAS08544-4431_dc_notnorm_V_PI_corr_tel+unres.fits'
            pdi_v= obp.Loadimage(pdi_folder_v,pdi_file_v)
            obp.plot_polarimetric_image(pdi_v, 3.6, title='IRAS08544-4431 V-band PI', save=figdir+'/pi_v_band.png', image_scale='asinh', roi_half_size=50)
        
        if  "pdi_I" in fidelity_products:    
            file_psf=star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits'
            psf_i=obp.Loadimage(figfolder_psf,file_psf)
            obp.plot_polarimetric_image(psf_i, 3.6, title='IRAS08544-4431 I-band PSF', save=figdir+'/psf_i_band.png', image_scale='asinh', roi_half_size=30)
            #polarimetric observations
            pdi_folder_i = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/I_band/'
            pdi_file_i = 'IRAS08544-4431_dc_notnorm_I_PI_corr_tel+unres.fits'
            pdi_i= obp.Loadimage(pdi_folder_i,pdi_file_i)
            obp.plot_polarimetric_image(pdi_i, 3.6, title='IRAS08544-4431 I-band PI', save=figdir+'/pi_i_band.png', image_scale='asinh', roi_half_size=50)
            
        if  "pdi_H" in fidelity_products: 
            file_psf='iras08544-4431_calib_H_I_meancombined.fits'
            psf_h=obp.Loadimage(pdi_folder_h,file_psf)
            obp.plot_polarimetric_image(psf_h, 12.27, title='IRAS08544-4431 H-band PSF', save=figdir+'/psf_h_band.png', image_scale='asinh', roi_half_size=30)
            pdi_folder_h = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/H_band/'
            pdi_file_h = 'iras08544-4431_calib_H_PI_corr_tel+unres.fits'
            pdi_h= obp.Loadimage(pdi_folder_h,pdi_file_h)
            obp.plot_polarimetric_image(pdi_h, 12.27, title='IRAS08544-4431 H-band PI', save=figdir+'/pi_h_band.png', image_scale='asinh', roi_half_size=30)
            
        if "alma" in fidelity_products:
            try:
                data_dir_alma, data_file_alma = "/Users/katerynaandrych/Work/lin/Postdoc/Data/ALMA/IRAS08544-4431/", "IRAS08_cont_multiscale_robust0_2mas.image.pbcor.fits"
                alma_cont, alma_header,pix_scale_alma, data_size_alma=oba.Loadimage_alma(data_dir_alma, data_file_alma)
                ps_alma=pix_scale_alma*1.0
                alma_spec = oba.alma_image_spec(alma_header)
                alma_wavelength = alma_spec["wavelength_um"]
                
                print('ALMA data loaded')
                
            except:
                print("[main] ALMA data files not found. Please check the path if your budget expects ALMA data.")
                alma_cont=None
                ps_alma=None
                alma_wavelength=None
                data_size_alma=None
                alma_spec=None



    elif data_root =='demo_ozstar':
        if  "sed" in fidelity_products: 
            try:
                data_filename = '/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/SED/IRAS08544-4431.phot'
                data_wave, data_flux, data_err = obs.load_sed_data(data_filename)
                print('SED data loaded')
            except:
                print("[main] SED data file not found. Please check the path if your budget expects SED.")
                data_wave, data_flux, data_err=[],[],[]
        else:
            data_wave, data_flux, data_err=[],[],[]

        if  "vis2_1perband" in fidelity_products:
            # PIONIER data
            try:        
                data_dir_pionier, data_file_pionier = "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/PIONIER/", "*.fits"
                container_data_pionier = distroi.read_oi_container_from_oifits(data_dir_pionier, data_file_pionier, wave_lims=(1.63, 1.64))
                print('PIONIER data loaded')
            except:
                print("[main] PIONIER data files not found. Please check the path if your budget expects PIONIER data.")
                container_data_pionier=None

            # GRAVITY data
            try:    
                data_dir_gravity, data_file_gravity = "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/GRAVITY/", "*1.fits"
                container_data_gravity = distroi.read_oi_container_from_oifits(data_dir_gravity, data_file_gravity, wave_lims=(2.199, 2.201))
                print('GRAVITY data loaded')
            except:
                print("[main] GRAVITY data files not found. Please check the path if your budget expects GRAVITY data.")
                container_data_gravity=None

            # VLTI/MATISSE L-band data
            try:
                data_dir_matisse_l, data_file_matisse_l = "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/MATISSE_L/", "*.fits"
                container_data_matisse_l = distroi.read_oi_container_from_oifits(data_dir_matisse_l, data_file_matisse_l, wave_lims=(3.48, 3.52))
                print('MATISSE L-band data loaded')
            except:
                print("[main] MATISSE L-band data files not found. Please check the path if your budget expects MATISSE L-band data.")
                container_data_matisse_l=None

            # VLTI/MATISSE N-band data
            try:    
                data_dir_matisse_n, data_file_matisse_n = "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/MATISSE_N/", "*.fits"
                container_data_matisse_n = distroi.read_oi_container_from_oifits(data_dir_matisse_n, data_file_matisse_n, wave_lims=(9.9, 10.10), fcorr=True)
                print('MATISSE N-band data loaded')
            except:
                print("[main] MATISSE N-band data files not found. Please check the path if your budget expects MATISSE N-band data.")
                container_data_matisse_n=None
        elif  ("vis2_chromatic" in fidelity_products):
            # PIONIER data
            try:        
                data_dir_pionier, data_file_pionier = "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/PIONIER/", "*.fits"
                container_data_pionier = distroi.read_oi_container_from_oifits(data_dir_pionier, data_file_pionier)
                print('PIONIER data loaded')
            except:
                print("[main] PIONIER data files not found. Please check the path if your budget expects PIONIER data.")
                container_data_pionier=None

            # GRAVITY data
            try:    
                data_dir_gravity, data_file_gravity = "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/GRAVITY/", "*1.fits"
                container_data_gravity = distroi.read_oi_container_from_oifits(data_dir_gravity, data_file_gravity)
                print('GRAVITY data loaded')
            except:
                print("[main] GRAVITY data files not found. Please check the path if your budget expects GRAVITY data.")
                container_data_gravity=None

            # VLTI/MATISSE L-band data
            try:
                data_dir_matisse_l, data_file_matisse_l = "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/MATISSE_L/", "*.fits"
                container_data_matisse_l = distroi.read_oi_container_from_oifits(data_dir_matisse_l, data_file_matisse_l, wave_lims=(2.95, 3.95), v2lim=1e-8)
                print('MATISSE L-band data loaded')
            except:
                print("[main] MATISSE L-band data files not found. Please check the path if your budget expects MATISSE L-band data.")
                container_data_matisse_l=None

            # VLTI/MATISSE N-band data
            try:    
                data_dir_matisse_n, data_file_matisse_n = "/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/MATISSE_N/", "*.fits"
                container_data_matisse_n = distroi.read_oi_container_from_oifits(data_dir_matisse_n, data_file_matisse_n, wave_lims=(8.0, 13.0), v2lim=1e-8, fcorr=True)
                print('MATISSE N-band data loaded')
            except:
                print("[main] MATISSE N-band data files not found. Please check the path if your budget expects MATISSE N-band data.")
                container_data_matisse_n=None
        else:
            container_data_pionier=None
            container_data_gravity=None
            container_data_matisse_l=None
            container_data_matisse_n=None

        if  "pdi_V" in fidelity_products or "pdi_I" in fidelity_products or "pdi_H" in fidelity_products:
            
            #real PSF from observations
            figfolder_psf='/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/psf/'
            star_psf='HD83878'
            pdi_h={}
            pdi_i={}
            pdi_v={}
            
            if  "pdi_V" in fidelity_products:
                file_psf=star_psf+'_'+'V'+'_'+'I'+'_meancombined.fits'
                psf_v=obp.Loadimage(figfolder_psf,file_psf)
                obp.plot_polarimetric_image(psf_v, 3.6, title='IRAS08544-4431 V-band PSF', save=figdir+'/psf_v_band.png', image_scale='asinh', roi_half_size=30)
                #polarimetric observations
                pdi_folder_v = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/V_band/'
                file_type=['PI','Q','U', "Q_phi", "U_phi"]
                for ft in file_type:
                    pdi_file_v = f'IRAS08544-4431_dc_notnorm_V_{ft}_corr_tel+unres.fits'
                    pdi_v[ft]= obp.Loadimage(pdi_folder_v,pdi_file_v)
                    obp.plot_polarimetric_image(pdi_v[ft], 3.6, title=f'IRAS08544-4431 V-band {ft}', save=figdir+f'/v_band_{ft}.png', image_scale='asinh', roi_half_size=50)
                
                pdi_file_v = 'IRAS08544-4431_dc_notnorm_V_I_meancombined.fits'
                pdi_v['I']= obp.Loadimage(pdi_folder_v,pdi_file_v)
                obp.plot_polarimetric_image(pdi_v['I'], 3.6, title='IRAS08544-4431 V-band I', save=figdir+'/v_band_I.png', image_scale='asinh', roi_half_size=50)

            
            if  "pdi_I" in fidelity_products:
                file_psf=star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits'
                psf_i=obp.Loadimage(figfolder_psf,file_psf)
                obp.plot_polarimetric_image(psf_i, 3.6, title='IRAS08544-4431 I-band PSF', save=figdir+'/psf_i_band.png', image_scale='asinh', roi_half_size=30)
                #polarimetric observations
                pdi_folder_i = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/I_band/'
                file_type=['PI','Q','U', "Q_phi", "U_phi"]
                for ft in file_type:
                    pdi_file_i = f'IRAS08544-4431_dc_notnorm_I_{ft}_corr_tel+unres.fits'
                    pdi_i[ft]= obp.Loadimage(pdi_folder_i,pdi_file_i)
                    obp.plot_polarimetric_image(pdi_i[ft], 3.6, title=f'IRAS08544-4431 I-band {ft}', save=figdir+f'/i_band_{ft}.png', image_scale='asinh', roi_half_size=50)
                
                pdi_file_i = 'IRAS08544-4431_dc_notnorm_I_I_meancombined.fits'
                pdi_i['I']= obp.Loadimage(pdi_folder_i,pdi_file_i)
                obp.plot_polarimetric_image(pdi_i['I'], 3.6, title='IRAS08544-4431 I-band I', save=figdir+'/i_band_I.png', image_scale='asinh', roi_half_size=50)


            if  "pdi_H" in fidelity_products:
                pdi_folder_h = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/H_band/'
                
                file_psf='iras08544-4431_calib_H_I_meancombined.fits'
                psf_h=obp.Loadimage(pdi_folder_h,file_psf)
                obp.plot_polarimetric_image(psf_h, 12.27, title='IRAS08544-4431 H-band PSF', save=figdir+'/psf_h_band.png', image_scale='asinh', roi_half_size=30)
                pdi_h['I']= psf_h
                obp.plot_polarimetric_image(pdi_h['I'], 12.27, title='IRAS08544-4431 H-band I', save=figdir+'/h_band_I.png', image_scale='asinh', roi_half_size=30)
                #polarimetric observations
                file_type=['PI', "Q_phi", "U_phi"]
                for ft in file_type:
                    pdi_file_h = f'iras08544-4431_calib_H_{ft}_corr_tel+unres.fits'
                    pdi_h[ft]= obp.Loadimage(pdi_folder_h,pdi_file_h)
                    obp.plot_polarimetric_image(pdi_h[ft], 12.27, title=f'IRAS08544-4431 H-band {ft}', save=figdir+f'/h_band_{ft}.png', image_scale='asinh', roi_half_size=30)
                file_type=['Q','U']
                for ft in file_type:
                    pdi_file_h = f'iras08544-4431_calib_H_{ft}_meancombined.fits'
                    pdi_h[ft]= obp.Loadimage(pdi_folder_h,pdi_file_h)
                    obp.plot_polarimetric_image(pdi_h[ft], 12.27, title=f'IRAS08544-4431 H-band {ft}', save=figdir+f'/h_band_{ft}.png', image_scale='asinh', roi_half_size=30)
                
                



            print('Polarimetric data loaded')
        else:
            psf_v=None
            psf_i=None
            psf_h=None
            pdi_v=None
            pdi_i=None
            pdi_h=None

        if  "alma" in fidelity_products:
            alma_folder = '/fred/oz061/kandrych/Data/ALMA/IRAS08544-4431/'
            alma_cont_file='IRAS08_cont_multiscale_robust0_2mas.image.pbcor.fits'
            alma_cont, alma_header, ps_alma, data_size_alma=oba.Loadimage_alma(alma_folder, alma_cont_file)
            alma_spec = oba.alma_image_spec(alma_header)
            alma_wavelength = alma_spec["wavelength_um"]
            obp.plot_polarimetric_image(alma_cont, ps_alma, title='IRAS08544-4431 ALMA continuum', save=figdir+'/alma_cont.png', image_scale='linear')
            print('ALMA continuum loaded')
        else:
            alma_cont=None
            ps_alma=None
            alma_wavelength=None
            data_size_alma=None
            alma_spec=None

        

    elif data_root =='ar_pup_ozstar':
        #HERE FILES AND TUPES NOT FIXED YET, TO DO
        #NEW VERSION IS IN IRAS08 OZSTAR

        #real PSF from observations
        if  "pdi_V" in fidelity_products or "pdi_I" in fidelity_products:
            figfolder_psf='/fred/oz061/kandrych/Data/polarimetry/AR_Pup_zimpol_2018/psf/'
            star_psf='HD75885'
        if  "pdi_V" in fidelity_products:
            file_psf=star_psf+'_'+'V'+'_'+'I'+'_meancombined.fits'
            psf_v=obp.Loadimage(figfolder_psf,file_psf)
            obp.plot_polarimetric_image(psf_v, 3.6, title='AR_Pup V-band PSF', save=figdir+'/psf_v_band.png', image_scale='asinh', roi_half_size=60)
            #polarimetric observations
            pdi_folder_v = '/fred/oz061/kandrych/Data/polarimetry/AR_Pup_zimpol_2018/V_band/'
            pdi_file_v = 'AR_Pup_dc_notnorm_V_decon.fits'
            #pdi_file_v = 'AR_Pup_dc_notnorm_V_PI_corr_tel+unres.fits'
            pdi_v= obp.Loadimage(pdi_folder_v,pdi_file_v)
            obp.plot_polarimetric_image(pdi_v, 3.6, title='AR_Pup V-band PI', save=figdir+'/pi_v_band.png', image_scale='asinh', roi_half_size=60)
            # pdi_decon_v= obp.Loadimage(pdi_folder_v,'AR_Pup_dc_notnorm_V_decon.fits')
            pdi_v=obp.center_crop(pdi_v, 150)

        if  "pdi_I" in fidelity_products:
            file_psf=star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits'
            psf_i=obp.Loadimage(figfolder_psf,file_psf)
            obp.plot_polarimetric_image(psf_i, 3.6, title='AR_Pup I-band PSF', save=figdir+'/psf_i_band.png', image_scale='asinh', roi_half_size=60)
            pdi_folder_i = '/fred/oz061/kandrych/Data/polarimetry/AR_Pup_zimpol_2018/I_band/'
            #pdi_file_i = 'AR_Pup_dc_notnorm_I_PI_corr_tel+unres.fits'
            pdi_file_i = 'AR_Pup_dc_notnorm_I_decon.fits'
            pdi_i= obp.Loadimage(pdi_folder_i,pdi_file_i)
            pdi_i=obp.center_crop(pdi_i, 150)
            
            obp.plot_polarimetric_image(pdi_i, 3.6, title='AR Pup I-band PI', save=figdir+'/pi_i_band.png', image_scale='asinh', roi_half_size=60)
            # pdi_decon_i= obp.Loadimage(pdi_folder_i,'AR_Pup_dc_notnorm_I_decon.fits')
            
        pdi_h=None
        psf_h=None
        container_data_pionier=None
        container_data_gravity=None
        container_data_matisse_l=None
        container_data_matisse_n=None
        data_wave, data_flux, data_err=[],[],[]


    else:
        raise ValueError(f"Unknown data_root: {data_root}")
    

    if "pdi_V" in fidelity_products:    
        # Calculate metrics for arcsinh-scaled images to highlight morphology
        radial_profile_v={}
        azimuthal_profile_v={}
        for ft in ['PI',"Q_phi",'I']:
            radial_profile_v[ft], azimuthal_profile_v[ft] = obp.profiles(pdi_v[ft], 3.6, 
                                                    profile_type="both",
                                                    mode="sum",
                                                    radial_limit_mas=500,
                                                    plot=False,
                                                    deprojection_inc_pa_deg=(0.0, 0.0),
                                                    center=None,
                                                    az_nbins=18,
                                                    azimuthal_r_in_mas=0.0,
                                                    azimuthal_r_out_mas=500.0,
                                                    theta0=0.0
                                                    ) 
    else:
        radial_profile_v=None
        azimuthal_profile_v=None

    if "pdi_I" in fidelity_products:    
        radial_profile_i={}
        azimuthal_profile_i={}
        # Calculate metrics for arcsinh-scaled images to highlight morphology
        for ft in ['PI',"Q_phi",'I']:
            radial_profile_i[ft], azimuthal_profile_i[ft] = obp.profiles(pdi_i[ft], 3.6, 
                                                    profile_type="both",
                                                    mode="sum",
                                                    radial_limit_mas=500,
                                                plot=False,
                                                deprojection_inc_pa_deg=(0.0, 0.0),
                                                center=None,
                                                az_nbins=18,
                                                azimuthal_r_in_mas=0.0,
                                                azimuthal_r_out_mas=500.0,
                                                theta0=0.0
                                                ) 
    else:
        radial_profile_i=None
        azimuthal_profile_i=None

    if "pdi_H" in fidelity_products:
        radial_profile_h={}
        azimuthal_profile_h={}
        # Calculate metrics for arcsinh-scaled images to highlight morphology
        for ft in ['PI',"Q_phi",'I']:
            radial_profile_h[ft], azimuthal_profile_h[ft] = obp.profiles(pdi_h[ft], 12.27, 
                                                    profile_type="both",
                                                    mode="sum",
                                                    radial_limit_mas=500,
                                                    plot=False,
                                                    deprojection_inc_pa_deg=(0.0, 0.0),
                                                    center=None,
                                                    az_nbins=18,
                                                    azimuthal_r_in_mas=0.0,
                                                    azimuthal_r_out_mas=500.0,
                                                theta0=0.0
                                                )
    else:
        radial_profile_h=None
        azimuthal_profile_h=None    

    if "alma" in fidelity_products:
        # Calculate metrics for arcsinh-scaled images to highlight morphology
        
        radial_profile_alma, azimuthal_profile_alma = obp.profiles(alma_cont, 2.0, 
                                                profile_type="both",
                                                mode="mean",
                                                radial_limit_mas=100,
                                                plot=False,
                                                deprojection_inc_pa_deg=(0.0, 0.0),
                                                center=None,
                                                az_nbins=18,
                                                azimuthal_r_in_mas=0.0,
                                                azimuthal_r_out_mas=100.0,
                                                theta0=0.0
                                                )
        ny, nx = alma_cont.shape
        xc = (nx - 1) / 2.0
        yc = (ny - 1) / 2.0
        R,_,_,X, Y = obp.compute_grid(alma_cont, xc=xc, yc=yc)
        noise_level_alma=np.nanstd(alma_cont[R>(150/ps_alma)])#here we take the noise level from the outer part of the image, where there is no emission
        # Fix the aperture from observations once, before any trial is launched.
        centre = (float(alma_header["CRPIX1"])-1, float(alma_header["CRPIX2"])-1)
        alma_aperture_radius_mas = ps_alma * oba.alma_snr_aperture_radius(
            alma_cont, noise_level_alma, center_xy=centre,
            snr_threshold=3.0, padding_pixels=3.0)
        available_radius_mas = min(centre[0]+0.5, nx-0.5-centre[0],
                                   centre[1]+0.5, ny-0.5-centre[1]) * ps_alma
        if alma_aperture_radius_mas > available_radius_mas:
            raise ValueError(
                f"Observed ALMA SNR aperture ({alma_aperture_radius_mas:g} mas) exceeds "
                f"image coverage ({available_radius_mas:g} mas). Inspect edge detections "
                "and choose a valid observational aperture before optimisation.")
        # The 3-sigma criterion selects only the radius, not individual scoring pixels.
        # Retained for reference: former individual-pixel scoring mask and plot.
        # mask_alma = (alma_cont >= 3*noise_level_alma)
        # #local plotting
        # plt.imshow(mask_alma, origin='lower')
        # plt.savefig(figdir+'/alma_cont_mask_3snr.png', dpi=300)
        # plt.close()
        mask_alma = None  # Retained dictionary key for compatibility; not used for scoring.
        
    else:
        radial_profile_alma=None
        azimuthal_profile_alma=None
        mask_alma = None
        noise_level_alma = None 
        noise_level_alma = None

    pdi_data_v={'psf': psf_v, 'pol_images': pdi_v, 'radial_profiles': radial_profile_v, 'azimuthal_profiles': azimuthal_profile_v}
    pdi_data_i={'psf': psf_i, 'pol_images': pdi_i, 'radial_profiles': radial_profile_i, 'azimuthal_profiles': azimuthal_profile_i}
    pdi_data_h={'psf': psf_h, 'pol_images': pdi_h, 'radial_profiles': radial_profile_h, 'azimuthal_profiles': azimuthal_profile_h}

    data_alma={'alma_cont': alma_cont, 'ps_alma': ps_alma,'image_size': data_size_alma, 'alma_wavelength': alma_wavelength, 'image_spec': alma_spec, 'header': alma_header,
               'radial_profile': radial_profile_alma, 'azimuthal_profile': azimuthal_profile_alma, 'mask_alma': mask_alma, 'noise_level_alma': noise_level_alma, 'aperture_radius_mas': alma_aperture_radius_mas}
    if "vis2_1perband" in fidelity_products or "vis2_chromatic" in fidelity_products:
        for container in (container_data_pionier, container_data_gravity,
                          container_data_matisse_l, container_data_matisse_n):
            obi.validate_interferometric_data(
                container, 'vis' if container.vis_in_fcorr else 'vis2')
    data_sed = [data_wave, data_flux, data_err]
    data_arrays = [data_sed, container_data_pionier, container_data_gravity, container_data_matisse_l, container_data_matisse_n,pdi_data_v, pdi_data_i, pdi_data_h, data_alma]
    print('data loaded')

    return data_arrays






# def make_unique_trial_dir(scratch_root: Path, seed: int, budget: float) -> Path:
#     """Previous seed/timestamp naming, retained for reference."""
#     ts = time.strftime("%Y%m%d-%H%M%S")
#     base = Path(scratch_root) / f"trial_seed{seed}_budget{budget:.2f}_{ts}"
#     trial_dir = base
#     idx = 0
#     while True:
#         try:
#             trial_dir.mkdir(parents=True, exist_ok=False)
#             return trial_dir
#         except FileExistsError:
#             idx += 1
#             trial_dir = base.with_name(f"{base.name}_{idx:03d}")


def make_unique_config_trial_dir(scratch_root: Path, config_id: int, budget: float) -> Path:
    """Create ``config_<id>_budget_<budget>`` folders with an atomic suffix.

    A SMAC configuration may be evaluated repeatedly, including at several
    budgets.  The suffix preserves every evaluation while the shared config ID
    remains visible in the folder and its diagnostic plot names.
    """
    base = Path(scratch_root) / f"config_{config_id}_budget_{budget:.2f}"
    trial_dir = base
    suffix = 0
    while True:
        try:
            trial_dir.mkdir(parents=True, exist_ok=False)
            return trial_dir
        except FileExistsError:
            suffix += 1
            trial_dir = base.with_name(f"{base.name}_{suffix:03d}")


def objective(cfg: Dict[str, Any], seed: int, budget: float, data_arg: Dict[str, Any],
              scratch_root: str, args) -> Tuple[float, Dict[str, Any]]:
    fidelity = map_budget_to_fidelity(budget)

    # Each trial gets a private scratch dir
    # trial_dir = Path(scratch_root) / f"trial_seed{seed}_budget{budget:.2f}" /time.strftime("%Y%m%d-%H%M%S")
    # trial_dir.mkdir(parents=True, exist_ok=True)
    # SMAC registers the running trial before dispatching it to Dask and stores
    # the resulting runhistory ID on this serializable Configuration object.
    config_id = getattr(cfg, "config_id", None)
    if config_id is None:
        raise RuntimeError("SMAC Configuration has no config_id before objective evaluation.")
    trial_dir = make_unique_config_trial_dir(Path(scratch_root), config_id, budget)

    # Convert cfg (ConfigSpace.Configuration) to dict
    if hasattr(cfg, "get_dictionary"):
        cfg = dict(cfg)

    if args.puffed_up_rim:
        cfg["puffed_r_rim"] = (float(cfg["zone_1_Rin"]) + float(cfg["puffed_r_offset"]))  # Add puffed-up rim radius to config for MCFOST

    # Write param file and run MCFOST
    par_path = obm.write_mcfost_paramfile(
        cfg, fidelity, trial_dir,
        two_zone_cont_lhc=getattr(args, "two_zone_cont_lhc", False),
        tapered_edge_p1_eq_p2=args.tapered_edge_p1_eq_p2)
    try:
        obm.run_mcfost(fidelity,par_path, trial_dir, args.puffed_up_rim, cfg, alma_spec=(data_arg[8]["image_spec"] if "alma" in fidelity["products"] else None))
    except Exception as error:
        # Trial failed; return a high loss
        failure = {
            "code": "mcfost_execution_failed",
            "stage": "run_mcfost",
            "reason": str(error),
            "exception_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "trial_dir": str(trial_dir)}
        failure.update(getattr(error, "mcfost_details", {}))
        print(f"[objective] Trial failed for cfg={cfg}, dir={trial_dir}: "
              f"{type(error).__name__}: {error}")
        return 1e99, {"failure": failure}

    # Score outputs

    loss, additional_info = obm.load_and_score_outputs(fidelity, trial_dir, data_arg, args,cfg)
    return float(loss), additional_info  # Return loss and additional info for SMAC


def warmstart_from_runhistory_json(
    smac,
    cs,
    warmstart_path: str,
    *,
    only_finished: bool = True,
    max_trials: int | None = None,
) -> int:
    """
    Warmstart SMAC by reading a SMAC runhistory.json and calling smac.tell(...).
    This is robust across SMAC versions.

    warmstart_path may be:
      - directory containing runhistory.json
      - direct path to runhistory.json
    """
    p = Path(warmstart_path).expanduser().resolve()
    rh_path = (p / "runhistory.json") if p.is_dir() else p
    if not rh_path.exists():
        raise FileNotFoundError(f"[warmstart] runhistory.json not found at {rh_path}")

    payload = json.loads(rh_path.read_text())

    # SMAC runhistory.json typically contains:
    #  - payload["configs"]: mapping config_id -> dict of hyperparams
    #  - payload["data"]: list of trials, each referencing config_id
    cfg_table = payload.get("configs", {})
    data = payload.get("data", [])

    if not cfg_table or not data:
        print(f"[warmstart] Nothing to load from {rh_path} (missing 'configs' or 'data').")
        return 0

    told = 0

    # Optionally: feed only the best few (by cost) to reduce noisy history
    trials = []
    for row in data:
        # Status: in many SMAC JSONs, SUCCESS is 1. (You showed status=1.)
        status = row.get("status", None)
        if only_finished and status != 1:
            continue

        cost = row.get("cost", None)
        if cost is None:
            continue

        config_id = row.get("config_id", None)
        if config_id is None:
            continue

        # Keys might be str or int in configs dict
        cfg_dict = cfg_table.get(str(config_id), cfg_table.get(config_id, None))
        if cfg_dict is None:
            continue

        trials.append((cost, row, cfg_dict))

    if not trials:
        print(f"[warmstart] No eligible trials found in {rh_path}.")
        return 0

    # Sort by best cost first
    trials.sort(key=lambda t: t[0])

    if max_trials is not None:
        trials = trials[:max_trials]

    for _, row, cfg_dict in trials:
        # Build Configuration from dict (must be compatible with current cs)
        try:
            cfg = Configuration(cs, values=cfg_dict)
        except Exception as e:
            # Typically means configspace changed; skip incompatible configs
            print(f"[warmstart] Skipping incompatible config_id={row.get('config_id')} ({e})")
            continue

        seed = int(row.get("seed", 0) or 0)
        budget = row.get("budget", None)

        # TrialInfo: include budget for multi-fidelity
        # (instance is usually None in your setup)
        info_kwargs = dict(config=cfg, seed=seed)
        if budget is not None:
            info_kwargs["budget"] = float(budget)

        info = TrialInfo(**info_kwargs)

        # TrialValue: cost is required. You can also pass time/cpu_time, but not necessary.
        value = TrialValue(cost=float(row["cost"]))

        smac.tell(info, value)
        told += 1

    print(f"[warmstart] Told SMAC about {told} previous trials from {rh_path}")
    return told


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="SMAC3 + Dask(SLURM) + Multi-fidelity for MCFOST")
    p.add_argument("--data-root", type=str, required=True, help="Path to observed-data root (for scoring) or 'demo'.")
    p.add_argument("--working-root", type=str, required=True, help="Working directory for trial runs")
    p.add_argument("--use-slurm", action="store_true", help="Use SLURMCluster instead of LocalCluster")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--procs-per-worker", type=int, default=1)
    p.add_argument("--min-budget", type=float, default=0.25)
    p.add_argument("--max-budget", type=float, default=3.0)
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--seed", type=int, default=-1)# Random seed for SMAC
    p.add_argument("--correct-unresolved-polarimetry", action="store_true", help="Use unresolved-corrected model images and profiles for polarimetry comparisons; requires --unresolved-correction-radius-px")
    p.add_argument("--warmstart", type=str, default=None, help="Path to previous SMAC run directory to warmstart from")
    p.add_argument("--plot-intermediate", action="store_true", help="Plot intermediate results during scoring")
    p.add_argument("--puffed-up-rim", action="store_true", help="Enable puffed up rim feature")
    p.add_argument("--2ZONE_CONT_LHC", dest="two_zone_cont_lhc", action="store_true",
                   help="Two contiguous zones (zone 1 power law; zone 2 power law or tapered): derive boundaries and dust masses from disk_Rmid [au] and disk_total_dust_mass [solar masses]")
    p.add_argument("--tapered-edge-p1-eq-p2", action="store_true", help="Tie p2 to p1 in the last zone of the MCFOST template")
    p.add_argument("--overresolved_flux_fit_for_interferometry", type=float, default=None, help="Optional: fit overresolved flux to interferometry at the supplied wavelength [micron]. Disabled by default.")
    p.add_argument("--unresolved-correction-radius-px", type=float, default=None, help="Unresolved-polarisation aperture radius in instrument pixels; required when correction is enabled")
    p.add_argument("--pdi-constraint-tolerances", type=float, nargs=3, default=(0.05, 0.05, 0.05), metavar=("FRACTION", "RADIAL", "QUADRANT"), help="Deprecated: ignored; PDI uses raw squared differences")
    p.add_argument("--pdi-absolute-error-floors", type=float, nargs=3, default=(0., 0., 0.),
                   metavar=("FRACTION", "RADIAL", "QUADRANT"), help="Deprecated: ignored; PDI uses raw squared differences")
    p.add_argument("--pdi-radial-bin-mas", type=float, default=25.0, help="Non-overlapping PDI radial bins [mas]")
    args = p.parse_args()
    
    WORK_ROOT = Path(args.working_root).resolve()
    os.environ["SMAC_WORK_ROOT"] = str(WORK_ROOT)
    # (WORK_ROOT / "dask-logs").mkdir(parents=True, exist_ok=True)
    # (WORK_ROOT / "dask-tmp").mkdir(parents=True, exist_ok=True)

    print(f"[main] Working root: {WORK_ROOT}")


    config_candidates = list(WORK_ROOT.rglob("*.yaml")) + list(WORK_ROOT.rglob("*.yml"))
    if not config_candidates:
        raise FileNotFoundError(f"No *.yaml found under {WORK_ROOT}")
    config_file = sorted(config_candidates)[0]

    print(f"[main] Using config space file: {config_file}")

    #client = start_cluster(args.n_workers, args.procs_per_worker, args.use_slurm)

    cs = build_configspace(config_file)

    scenario = Scenario(
        configspace=cs,
        name="optimization",
        n_trials=args.n_trials,
        output_directory=args.working_root,
        n_workers=args.n_workers,           # enables parallel evaluations
        deterministic=False,                # mark as noisy if your objective is stochastic
        seed=args.seed,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        # You can also set output directories, logging, etc.
    )
    work_root = Path(args.working_root)
    max_fidelity = map_budget_to_fidelity(args.max_budget)
    data_arg = load_data(args.data_root, str(work_root),max_fidelity["products"])

    trial_folder= work_root/"trials/"
    trial_folder.mkdir(exist_ok=True)

    # SMAC Objective wrapper with extra kwargs via lambda/closure
    def smac_objective(cfg, seed: int, budget: float) -> float:
        return objective(cfg, seed, budget, data_arg=data_arg,
                         scratch_root=trial_folder, args=args)
    

    
    
    smac_kwargs = dict(
        scenario=scenario,
        target_function=smac_objective,
        overwrite=False,
        initial_design=None
        #dask_client=client,   
    )

    smac = MultiFidelityFacade(**smac_kwargs)

    if args.warmstart is not None:
        warmstart_from_runhistory_json(
            smac=smac,
            cs=cs,
            warmstart_path=args.warmstart,
            only_finished=True,
            max_trials=None,  # or set e.g. 50
        )

    # Optional: callbacks (e.g., log incumbent every K trials)
    # smac.register_callback(HPO.LoggingCallback())

    incumbent = smac.optimize()

    print("\n=== Optimization finished ===")
    print("Incumbent config:")
    print(incumbent)

    # Save results
    results_dir = Path(args.working_root +"smac_results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "incumbent.json", "w") as f:
        json.dump(dict(incumbent) if hasattr(incumbent, "get_dictionary") else dict(incumbent), f, indent=2)

    # Convert cfg (ConfigSpace.Configuration) to dict
    if hasattr(incumbent, "get_dictionary"):
        incumbent = dict(incumbent)
    fidelity_result={}
    
    fidelity_result=map_budget_to_fidelity(args.max_budget)
    # Verify template parameter file for mcfost exists
    template_para= Path(results_dir.parent/"simulation.para")
    assert template_para.exists(), f"Missing template .para at {template_para}"

    # Write param file and run MCFOST
    par_path = obm.write_mcfost_paramfile(
        incumbent, fidelity_result, results_dir,
        two_zone_cont_lhc=args.two_zone_cont_lhc,
        tapered_edge_p1_eq_p2=args.tapered_edge_p1_eq_p2)
    obm.run_mcfost(fidelity_result,par_path, results_dir, args.puffed_up_rim, incumbent, alma_spec=(data_arg[8]["image_spec"] if "alma" in fidelity_result["products"] else None))
    # Score outputs
    args.plot_intermediate=True #to plot final results
    loss = obm.load_and_score_outputs(fidelity_result, results_dir, data_arg, args, incumbent)




    # Trajectory / RunHistory (optional deep dive)
    runhistory = smac.runhistory
    traj = smac.intensifier.trajectory
    print(f"Trials run: {len(runhistory)}")
    
    df = oao.runhistory_to_df(smac, cs)
    df.to_csv(results_dir / "trials.csv", index=False)
 

    # Persist runhistory in JSON for later analysis
    try:
        from smac.utils.logging import save_configspace
        save_configspace(cs, results_dir / "configspace.json")
    except Exception:
        pass


if __name__ == "__main__":
    main()
