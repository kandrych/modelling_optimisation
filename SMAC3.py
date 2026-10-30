"""Optimise MCFOST models against the configured observable stages.

SMAC manages local workers inside the current allocation. Fidelity stages
select observables; they do not change image resolution or photon counts.
The old experimental SLURM cluster launcher is archived in legacy/.
"""
from __future__ import annotations

import argparse
from functools import partial
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



# --- SMAC3 / ConfigSpace ---
# pip install smac configspace
from smac import Scenario
from smac.facade.multi_fidelity_facade import MultiFidelityFacade
from smac.runhistory import TrialValue
from smac import HyperparameterOptimizationFacade as HPO  # for callbacks, utils

from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical, Configuration
from ConfigSpace.conditions import InCondition, EqualsCondition

from smac.runhistory.dataclasses import TrialInfo, TrialValue
from smac.initial_design.sobol_design import SobolInitialDesign




import distroi

sys.path.append(os.path.abspath(".."))  # parent of current working dir
#import lib.Katya_func as kf

from lib.obriy_fidelity import load_fidelity_config, map_budget_to_fidelity, fidelity_products
from lib.obriy_interferometry_data import load_interferometry_data

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


def validate_configuration_names(configspace, template_path, *, puffed_up_rim=False,
                                 two_zone_cont_lhc=False):
    """Reject unsupported names before loading observations or starting workers.

    Check the whole ConfigSpace, including currently inactive parameters.
    Physical parameter names come from the actual simulation template.
    """
    template = obm.ParaFile(str(template_path)).params
    names = set(configspace.keys())
    allowed = set(template)

    # This alias sets both imin and imax in the parameter file.
    allowed.add('inclination')

    # These inputs are consumed by enabled features rather than written directly.
    if puffed_up_rim:
        allowed.update(('puffed_r_offset', 'puffed_h_rim_over_h0', 'puffed_delta_r'))
    if two_zone_cont_lhc:
        allowed.update(('disk_Rmid', 'disk_total_dust_mass'))

    # Conditional dust fractions have a user-chosen label after the suffix.
    # Accept them only for a real mixture and a configured material selector.
    for name in names:
        marker = '_component_2_volume_fraction_'
        if marker not in name:
            continue
        species, label = name.split(marker, 1)
        material = species + '_component_2_optical_indices_file'
        first_fraction = species + '_component_1_volume_fraction'
        second_fraction = species + '_component_2_volume_fraction'
        if (label and material in names and material in template
                and first_fraction in template and second_fraction in template):
            allowed.add(name)

    unknown = names - allowed
    if unknown:
        message = (f"Unsupported configuration parameter(s): {', '.join(sorted(unknown))}. "
                   f"Names must match parameters in {template_path} or supported derived inputs.")
        if any(name.startswith('puffed_') for name in unknown):
            message += (" Puffed-rim inputs require --puffed-up-rim and must be named "
                        "puffed_r_offset, puffed_h_rim_over_h0 or puffed_delta_r. "
                        "puffed_r_rim is calculated from zone_1_Rin + puffed_r_offset; "
                        "do not put it in ConfigSpace.")
        if unknown & {'disk_Rmid', 'disk_total_dust_mass'}:
            message += " disk_Rmid and disk_total_dust_mass require --2ZONE_CONT_LHC."
        raise ValueError(message)

# -----------------------------------------------------------------------------
# Multi-fidelity mapping
# -----------------------------------------------------------------------------

# Legacy budget mapping retained for reference.
# def map_budget_to_fidelity(budget: float) -> Dict[str, Any]:
#     """
#     Map the continuous budget in [min_budget, max_budget] to discrete fidelity settings.
#
#     Example strategy:
#     - Stage ladder F0→F3 via budget thresholds
#     - Photon packets and image resolution scale with budget - TO DO
#
#     """
#     # Example budget range: min=0.25, max=3.0 (set in Scenario)
#     stage = "F0"
#     if budget >= 1.0:
#         stage = "F1"
#     if budget >= 2.0:
#         stage = "F2"
#     if budget >= 3.0:
#         stage = "F3"
#     if budget >= 4.0:
#         stage = "F4"
#     if budget >= 5.0:
#         stage = "F5"
#     if budget >= 6.0:
#         stage = "F6"
#     if budget >= 7.0:
#         stage = "F7"
#     if budget >= 8.0:
#         stage = "F8"
#
#
#     if budget >= 9.0:
#         stage = "F9"
#     if budget >= 10.0:
#         stage = "F10"
#     if budget >= 11.0:
#         stage = "F11"
#
#
#     if budget >= 14.0:
#         stage = "F14"
#     if budget >= 15.0:
#         stage = "F15"
#     if budget >= 16.0:
#         stage = "F16"
#
#     # scale photons with budget
#     # nbr_photons_eq_th = int(1.28e5 * (10**budget))
#     # nbr_photons_lambda= int(1.28e3 * (10**budget))
#     # nbr_photons_image= int(1.28e4 * (10**budget))
#
#
#     # image resolution mapping (example)
#     if stage == "F0":
#         img_res = 10 #mas/pixel
#         products = ["sed"]
#     elif stage == "F1":
#         img_res = 2 #mas/pixel
#         products = ["sed", "vis2_1perband"]
#     elif stage == "F2":
#         img_res = 2
#         products = ["sed", "vis2_1perband", "pdi_V", "pdi_I", "pdi_H"]
#     elif stage == "F3":
#         img_res = 2
#         products = ["sed", "vis2_1perband", "pdi_V", "pdi_I", "pdi_H","alma"]
#     elif stage == "F4":
#         img_res = 2
#         products = ["sed", "vis2_chromatic", "pdi_V", "pdi_I", "pdi_H", "alma"]
#
#     elif stage == "F5":
#         img_res = 10 #mas/pixel
#         products = ["sed"]
#     elif stage == "F6":
#         img_res = 2 #mas/pixel
#         products = ["sed", "vis2_1perband"]
#     elif stage == "F7":
#         img_res = 2
#         products = ["sed", "vis2_chromatic"]
#
#
#     elif stage == "F9":
#         img_res = 2
#         products = ["sed"]
#     elif stage == "F10":
#         img_res = 2
#         products = ["sed", "alma"]
#     elif stage == "F11":
#         img_res = 2
#         products = ["sed", "pdi_V", "pdi_I", "pdi_H", "alma"]
#
#
#     elif stage == "F14":
#         img_res = 2
#         products = ["sed"]
#     elif stage == "F15":
#         img_res = 2
#         products = ["sed","pdi_V", "pdi_I", "pdi_H"]
#
#     elif stage == "F16": #created for AR Pup test
#         img_res = 2
#         products = ["pdi_V", "pdi_I", "pdi_H"]
#     else:
#         raise ValueError(f"Unknown stage for budget {budget}: {stage}")
#
#     return {
#         "stage": stage,
#         # "nbr_photons_eq_th": nbr_photons_eq_th,
#         # "nbr_photons_lambda": nbr_photons_lambda,
#         # "nbr_photons_image": nbr_photons_image,
#         "image_res": img_res,
#         "products": products,
#     }


# -----------------------------------------------------------------------------
# Objective callable for SMAC (supports multi-fidelity via `budget` argument)
# -----------------------------------------------------------------------------


def fixed_pdi_grid(para_path, configspace):
    """Read the image grid from .para and apply fixed configuration overrides."""
    parameters = obm.ParaFile(str(para_path)).params
    grid = {}
    for name in ('grid_nx', 'grid_ny', 'grid_size', 'distance_pc'):
        grid[name] = parameters[name]
        if name in configspace:
            parameter = configspace[name]
            if parameter.size != 1 or configspace.parents_of[name]:
                raise ValueError(f'{name} must be fixed to prepare PDI observation profiles once.')
            grid[name] = parameter.default_value
    return grid


def load_data(data_root: str, work_root: str, fidelity_products: list,
              *, pdi_grid=None, pdi_radial_bin_mas=25.0) -> Dict[str, Any]:
    """Load and validate requested products before profiles or worker startup.

    Optimisation passes the union of all configured stages. Unrequested
    observables remain optional; requested missing or invalid data raise with
    the observable, source path and original exception as the cause.
    """
    from lib.obriy_observations import (
        load_sed_observation, load_pdi_image, validate_pdi_observation,
        load_alma_observation,
    )

    pdi_sources = {band: {} for band in ("V", "I", "H")}

    def read_pdi(folder, filename, band, component):
        pdi_sources[band][component] = str(Path(folder) / filename)
        return load_pdi_image(folder, filename, obp.Loadimage, f"PDI {band} {component}")

    def read_alma(folder, filename):
        return load_alma_observation(folder, filename, oba.Loadimage_alma,
                                     oba.alma_image_spec)

    if data_root == "ar_pup_ozstar":
        unsupported = set(fidelity_products) & {"sed", "alma", "pdi_H"}
        if unsupported:
            raise ValueError(f"No observation files configured for {sorted(unsupported)} in {data_root}.")

    figdir = Path(work_root)/"data_figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figdir=str(figdir)
    # Absent observables still have a well-defined payload (including SED-only runs).
    data_wave, data_flux, data_err = [], [], []
    psf_v = psf_i = psf_h = None
    pdi_v = pdi_i = pdi_h = None
    alma_cont = ps_alma = data_size_alma = alma_wavelength = None
    interferometry_data = load_interferometry_data(
        data_root, fidelity_products, distroi.read_oi_container_from_oifits,
        validator=obi.validate_interferometric_data)
    alma_spec = None
    alma_header = None
    alma_aperture_radius_mas = None
    noise_level_alma = None

    #filename of SED catalogue data file
    if data_root =='demo_mac':
        if  "sed" in fidelity_products: 
            data_filename = '/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/SED/IRAS08544-4431.phot'
            data_wave, data_flux, data_err = load_sed_observation(data_filename, obs.load_sed_data)

        if  "pdi_V" in fidelity_products or "pdi_I" in fidelity_products or "pdi_H" in fidelity_products:
            #real PSF from observations
            star_psf='HD83878'
            figfolder_psf='/Users/katerynaandrych/Work/lin/PhD/SPHERE_reduction_data/paper2/mean_combined/'+star_psf+'/'

        if  "pdi_V" in fidelity_products:    
            file_psf=star_psf+'_'+'V'+'_'+'I'+'_meancombined.fits'
            psf_v=read_pdi(figfolder_psf,file_psf, "V", "PSF")
            obp.plot_polarimetric_image(psf_v, 3.6, title='IRAS08544-4431 V-band PSF', save=figdir+'/psf_v_band.png', image_scale='asinh', roi_half_size=30)
            #polarimetric observations
            pdi_folder_v = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/V_band/'
            pdi_file_v = 'IRAS08544-4431_dc_notnorm_V_PI_corr_tel+unres.fits'
            pdi_v= read_pdi(pdi_folder_v,pdi_file_v, "V", "PI")
            obp.plot_polarimetric_image(pdi_v, 3.6, title='IRAS08544-4431 V-band PI', save=figdir+'/pi_v_band.png', image_scale='asinh', roi_half_size=50)
        
        if  "pdi_I" in fidelity_products:    
            file_psf=star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits'
            psf_i=read_pdi(figfolder_psf,file_psf, "I", "PSF")
            obp.plot_polarimetric_image(psf_i, 3.6, title='IRAS08544-4431 I-band PSF', save=figdir+'/psf_i_band.png', image_scale='asinh', roi_half_size=30)
            #polarimetric observations
            pdi_folder_i = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/I_band/'
            pdi_file_i = 'IRAS08544-4431_dc_notnorm_I_PI_corr_tel+unres.fits'
            pdi_i= read_pdi(pdi_folder_i,pdi_file_i, "I", "PI")
            obp.plot_polarimetric_image(pdi_i, 3.6, title='IRAS08544-4431 I-band PI', save=figdir+'/pi_i_band.png', image_scale='asinh', roi_half_size=50)
            
        if  "pdi_H" in fidelity_products: 
            pdi_folder_h = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/H_band/'
            file_psf='iras08544-4431_calib_H_I_meancombined.fits'
            psf_h=read_pdi(pdi_folder_h,file_psf, "H", "PSF")
            obp.plot_polarimetric_image(psf_h, 12.27, title='IRAS08544-4431 H-band PSF', save=figdir+'/psf_h_band.png', image_scale='asinh', roi_half_size=30)
            pdi_file_h = 'iras08544-4431_calib_H_PI_corr_tel+unres.fits'
            pdi_h= read_pdi(pdi_folder_h,pdi_file_h, "H", "PI")
            obp.plot_polarimetric_image(pdi_h, 12.27, title='IRAS08544-4431 H-band PI', save=figdir+'/pi_h_band.png', image_scale='asinh', roi_half_size=30)
            
        if "alma" in fidelity_products:
            data_dir_alma, data_file_alma = "/Users/katerynaandrych/Work/lin/Postdoc/Data/ALMA/IRAS08544-4431/", "IRAS08_cont_multiscale_robust0_2mas.image.pbcor.fits"
            alma_cont, alma_header, ps_alma, data_size_alma, alma_spec = read_alma(data_dir_alma, data_file_alma)
            alma_wavelength = alma_spec["wavelength_um"]
            print('ALMA data loaded')

    elif data_root =='demo_ozstar':
        if  "sed" in fidelity_products: 
            data_filename = '/fred/oz061/kandrych/Data/interferometry/IRAS08544-4431/SED/IRAS08544-4431.phot'
            data_wave, data_flux, data_err = load_sed_observation(data_filename, obs.load_sed_data)
            print('SED data loaded')
        else:
            data_wave, data_flux, data_err=[],[],[]

        if  "pdi_V" in fidelity_products or "pdi_I" in fidelity_products or "pdi_H" in fidelity_products:
            
            #real PSF from observations
            figfolder_psf='/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/psf/'
            star_psf='HD83878'
            pdi_h={}
            pdi_i={}
            pdi_v={}
            
            if  "pdi_V" in fidelity_products:
                file_psf=star_psf+'_'+'V'+'_'+'I'+'_meancombined.fits'
                psf_v=read_pdi(figfolder_psf,file_psf, "V", "PSF")
                obp.plot_polarimetric_image(psf_v, 3.6, title='IRAS08544-4431 V-band PSF', save=figdir+'/psf_v_band.png', image_scale='asinh', roi_half_size=30)
                #polarimetric observations
                pdi_folder_v = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/V_band/'
                file_type=['PI','Q','U', "Q_phi", "U_phi"]
                for ft in file_type:
                    pdi_file_v = f'IRAS08544-4431_dc_notnorm_V_{ft}_corr_tel+unres.fits'
                    pdi_v[ft]= read_pdi(pdi_folder_v,pdi_file_v, "V", ft)
                    obp.plot_polarimetric_image(pdi_v[ft], 3.6, title=f'IRAS08544-4431 V-band {ft}', save=figdir+f'/v_band_{ft}.png', image_scale='asinh', roi_half_size=50)
                
                pdi_file_v = 'IRAS08544-4431_dc_notnorm_V_I_meancombined.fits'
                pdi_v["I"]= read_pdi(pdi_folder_v,pdi_file_v, "V", "I")
                obp.plot_polarimetric_image(pdi_v['I'], 3.6, title='IRAS08544-4431 V-band I', save=figdir+'/v_band_I.png', image_scale='asinh', roi_half_size=50)

            
            if  "pdi_I" in fidelity_products:
                file_psf=star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits'
                psf_i=read_pdi(figfolder_psf,file_psf, "I", "PSF")
                obp.plot_polarimetric_image(psf_i, 3.6, title='IRAS08544-4431 I-band PSF', save=figdir+'/psf_i_band.png', image_scale='asinh', roi_half_size=30)
                #polarimetric observations
                pdi_folder_i = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/I_band/'
                file_type=['PI','Q','U', "Q_phi", "U_phi"]
                for ft in file_type:
                    pdi_file_i = f'IRAS08544-4431_dc_notnorm_I_{ft}_corr_tel+unres.fits'
                    pdi_i[ft]= read_pdi(pdi_folder_i,pdi_file_i, "I", ft)
                    obp.plot_polarimetric_image(pdi_i[ft], 3.6, title=f'IRAS08544-4431 I-band {ft}', save=figdir+f'/i_band_{ft}.png', image_scale='asinh', roi_half_size=50)
                
                pdi_file_i = 'IRAS08544-4431_dc_notnorm_I_I_meancombined.fits'
                pdi_i["I"]= read_pdi(pdi_folder_i,pdi_file_i, "I", "I")
                obp.plot_polarimetric_image(pdi_i['I'], 3.6, title='IRAS08544-4431 I-band I', save=figdir+'/i_band_I.png', image_scale='asinh', roi_half_size=50)


            if  "pdi_H" in fidelity_products:
                pdi_folder_h = '/fred/oz061/kandrych/Data/polarimetry/IRAS08544-4431_for_modelling/H_band/'
                
                file_psf='iras08544-4431_calib_H_I_meancombined.fits'
                psf_h=read_pdi(pdi_folder_h,file_psf, "H", "PSF")
                obp.plot_polarimetric_image(psf_h, 12.27, title='IRAS08544-4431 H-band PSF', save=figdir+'/psf_h_band.png', image_scale='asinh', roi_half_size=30)
                pdi_h['I']= psf_h
                pdi_sources["H"]["I"] = pdi_sources["H"]["PSF"]
                obp.plot_polarimetric_image(pdi_h['I'], 12.27, title='IRAS08544-4431 H-band I', save=figdir+'/h_band_I.png', image_scale='asinh', roi_half_size=30)
                #polarimetric observations
                file_type=['PI', "Q_phi", "U_phi"]
                for ft in file_type:
                    pdi_file_h = f'iras08544-4431_calib_H_{ft}_corr_tel+unres.fits'
                    pdi_h[ft]= read_pdi(pdi_folder_h,pdi_file_h, "H", ft)
                    obp.plot_polarimetric_image(pdi_h[ft], 12.27, title=f'IRAS08544-4431 H-band {ft}', save=figdir+f'/h_band_{ft}.png', image_scale='asinh', roi_half_size=30)
                file_type=['Q','U']
                for ft in file_type:
                    pdi_file_h = f'iras08544-4431_calib_H_{ft}_meancombined.fits'
                    pdi_h[ft]= read_pdi(pdi_folder_h,pdi_file_h, "H", ft)
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
            alma_cont, alma_header, ps_alma, data_size_alma, alma_spec = read_alma(alma_folder, alma_cont_file)
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
            psf_v=read_pdi(figfolder_psf,file_psf, "V", "PSF")
            obp.plot_polarimetric_image(psf_v, 3.6, title='AR_Pup V-band PSF', save=figdir+'/psf_v_band.png', image_scale='asinh', roi_half_size=60)
            #polarimetric observations
            pdi_folder_v = '/fred/oz061/kandrych/Data/polarimetry/AR_Pup_zimpol_2018/V_band/'
            pdi_file_v = 'AR_Pup_dc_notnorm_V_decon.fits'
            #pdi_file_v = 'AR_Pup_dc_notnorm_V_PI_corr_tel+unres.fits'
            pdi_v= read_pdi(pdi_folder_v,pdi_file_v, "V", "PI")
            obp.plot_polarimetric_image(pdi_v, 3.6, title='AR_Pup V-band PI', save=figdir+'/pi_v_band.png', image_scale='asinh', roi_half_size=60)
            # pdi_decon_v= obp.Loadimage(pdi_folder_v,'AR_Pup_dc_notnorm_V_decon.fits')
            pdi_v=obp.center_crop(pdi_v, 150)

        if  "pdi_I" in fidelity_products:
            file_psf=star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits'
            psf_i=read_pdi(figfolder_psf,file_psf, "I", "PSF")
            obp.plot_polarimetric_image(psf_i, 3.6, title='AR_Pup I-band PSF', save=figdir+'/psf_i_band.png', image_scale='asinh', roi_half_size=60)
            pdi_folder_i = '/fred/oz061/kandrych/Data/polarimetry/AR_Pup_zimpol_2018/I_band/'
            #pdi_file_i = 'AR_Pup_dc_notnorm_I_PI_corr_tel+unres.fits'
            pdi_file_i = 'AR_Pup_dc_notnorm_I_decon.fits'
            pdi_i= read_pdi(pdi_folder_i,pdi_file_i, "I", "PI")
            pdi_i=obp.center_crop(pdi_i, 150)
            
            obp.plot_polarimetric_image(pdi_i, 3.6, title='AR Pup I-band PI', save=figdir+'/pi_i_band.png', image_scale='asinh', roi_half_size=60)
            # pdi_decon_i= obp.Loadimage(pdi_folder_i,'AR_Pup_dc_notnorm_I_decon.fits')
            
        pdi_h=None
        psf_h=None
        data_wave, data_flux, data_err=[],[],[]


    else:
        raise ValueError(f"Unknown data_root: {data_root}")
    

    for band, images, psf, pixel_scale in (
        ("V", pdi_v, psf_v, 3.6), ("I", pdi_i, psf_i, 3.6), ("H", pdi_h, psf_h, 12.27),
    ):
        if f"pdi_{band}" in fidelity_products:
            validate_pdi_observation(images, psf, pixel_scale, band, pdi_sources[band])

    if "alma" in fidelity_products:
        # Calculate observation-only scoring quantities once, before trials.
        R, *_ = obp.compute_grid(alma_cont)
        noise_level_alma = np.nanstd(alma_cont[R > 150 / ps_alma])
        centre = (float(alma_header["CRPIX1"]) - 1, float(alma_header["CRPIX2"]) - 1)
        alma_aperture_radius_mas = ps_alma * oba.alma_snr_aperture_radius(
            alma_cont, noise_level_alma, center_xy=centre,
            snr_threshold=3.0, padding_pixels=3.0)
        # The 3-sigma criterion selects only the radius, not individual scoring pixels.
        # Retained for reference: former individual-pixel scoring mask and plot.
        # mask_alma = (alma_cont >= 3*noise_level_alma)
        # #local plotting
        # plt.imshow(mask_alma, origin='lower')
        # plt.savefig(figdir+'/alma_cont_mask_3snr.png', dpi=300)
        # plt.close()
        mask_alma = None  # Retained dictionary key for compatibility; not used for scoring.
        
    else:
        mask_alma = None
        noise_level_alma = None 

    # Legacy profile keys remain for compatibility; active comparisons measure their own profiles.
    pdi_data_v={'psf': psf_v, 'pol_images': pdi_v, 'radial_profiles': None, 'azimuthal_profiles': None}
    pdi_data_i={'psf': psf_i, 'pol_images': pdi_i, 'radial_profiles': None, 'azimuthal_profiles': None}
    pdi_data_h={'psf': psf_h, 'pol_images': pdi_h, 'radial_profiles': None, 'azimuthal_profiles': None}

    # Prepare the scored profiles once, before distributing observations to workers.
    if pdi_grid is not None:
        for band, data, pixel_scale in (
            ('V', pdi_data_v, 3.6), ('I', pdi_data_i, 3.6), ('H', pdi_data_h, 12.27),
        ):
            if f'pdi_{band}' in fidelity_products:
                data['prepared_observation'] = obp.prepare_pdi_observation(
                    data['pol_images'], pixel_scale, pdi_grid, pdi_radial_bin_mas)
                radius = data['prepared_observation']['profiles']['aperture_radius_mas']
                print(f'[load_data] PDI {band}: observation profiles prepared within {radius:g} mas')

    data_alma={'alma_cont': alma_cont, 'ps_alma': ps_alma,'image_size': data_size_alma, 'alma_wavelength': alma_wavelength, 'image_spec': alma_spec, 'header': alma_header,
               'radial_profile': None, 'azimuthal_profile': None, 'mask_alma': mask_alma, 'noise_level_alma': noise_level_alma, 'aperture_radius_mas': alma_aperture_radius_mas}
    data_sed = [data_wave, data_flux, data_err]
    data_arrays = {
        "sed": data_sed,
        "interferometry": interferometry_data,
        "pdi_V": pdi_data_v,
        "pdi_I": pdi_data_i,
        "pdi_H": pdi_data_h,
        "alma": data_alma,
    }
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
    """Create trial folders atomically, adding ``_retrial1``, ``_retrial2``, etc.

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
            trial_dir = base.with_name(f"{base.name}_retrial{suffix}")


def prepare_model_config(cfg, puffed_up_rim: bool) -> Dict[str, Any]:
    """Derive execution inputs without changing SMAC's sampled configuration."""
    model_cfg = dict(cfg)
    if puffed_up_rim:
        model_cfg["puffed_r_rim"] = (
            float(model_cfg["zone_1_Rin"]) + float(model_cfg["puffed_r_offset"]))
    return model_cfg


def objective(cfg: Dict[str, Any], seed: int, budget: float, data_arg: Dict[str, Any],
              scratch_root: str, args) -> Tuple[float, Dict[str, Any]]:
    fidelity = map_budget_to_fidelity(budget, args.fidelity_settings)
    print(f"[objective] Stage {fidelity['stage']}, budget {budget}: {fidelity['products']}")

    # Each trial gets a private scratch dir
    # trial_dir = Path(scratch_root) / f"trial_seed{seed}_budget{budget:.2f}" /time.strftime("%Y%m%d-%H%M%S")
    # trial_dir.mkdir(parents=True, exist_ok=True)
    # SMAC registers the running trial before dispatching it to Dask and stores
    # the resulting runhistory ID on this serializable Configuration object.
    config_id = getattr(cfg, "config_id", None)
    if config_id is None:
        raise RuntimeError("SMAC Configuration has no config_id before objective evaluation.")
    reserved_trial_dir = getattr(cfg, "_obriy_trial_dir", None)
    trial_dir = (Path(reserved_trial_dir) if reserved_trial_dir is not None else
                 make_unique_config_trial_dir(Path(scratch_root), config_id, budget))

    cfg = prepare_model_config(cfg, args.puffed_up_rim)

    # Write param file and run MCFOST
    par_path = obm.write_mcfost_paramfile(
        cfg, fidelity, trial_dir,
        two_zone_cont_lhc=getattr(args, "two_zone_cont_lhc", False),
        tapered_edge_p1_eq_p2=args.tapered_edge_p1_eq_p2,
        share_zone_composition=getattr(args, "share_zone_composition", False))
    try:
        obm.run_mcfost(fidelity,par_path, trial_dir, args.puffed_up_rim, cfg, alma_spec=(data_arg["alma"]["image_spec"] if "alma" in fidelity["products"] else None))
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
        return 1e99, {"failure": failure, "fidelity": fidelity}

    # Score outputs

    loss, additional_info = obm.load_and_score_outputs(fidelity, trial_dir, data_arg, args,cfg)
    additional_info["fidelity"] = fidelity
    return float(loss), additional_info  # Return loss and additional info for SMAC


def warmstart_from_runhistory_json(
    smac,
    cs,
    warmstart_path: str,
    *,
    only_finished: bool = True,
    max_trials: int | None = None,
    fidelity_settings=None,
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

        if fidelity_settings is not None:
            recorded = (row.get("additional_info") or {}).get("fidelity")
            try:
                expected = map_budget_to_fidelity(row.get("budget"), fidelity_settings)
            except (ValueError, TypeError):
                continue
            if (not isinstance(recorded, dict)
                    or recorded.get("products") != expected["products"]
                    or recorded.get("image_res") != expected["image_res"]):
                print(f"[warmstart] Skipping config {row.get('config_id')}: missing/incompatible fidelity metadata")
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
        value = TrialValue(cost=float(row["cost"]), additional_info=row.get("additional_info", {}))

        smac.tell(info, value)
        told += 1

    print(f"[warmstart] Told SMAC about {told} previous trials from {rh_path}")
    return told


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------

def prepare_final_results_dir(work_root: Path) -> Path:
    """Preserve the previous final run and return an empty output directory.

    Image-directory existence is used to skip repeated wavelengths within a
    simulation. No outputs from a previous incumbent may enter that cache.
    """
    work_root = Path(work_root)
    results_dir = work_root / "smac_results"
    if results_dir.is_symlink() or (results_dir.exists() and not results_dir.is_dir()):
        raise ValueError(f"Expected a regular results directory: {results_dir}")
    if results_dir.exists():
        history_dir = work_root / "smac_results_history"
        history_dir.mkdir(exist_ok=True)
        archive_dir = Path(tempfile.mkdtemp(prefix="run_", dir=history_dir))
        archived_results = archive_dir / "smac_results"
        results_dir.rename(archived_results)
        print(f"[main] Previous final results preserved in {archived_results}", flush=True)
    results_dir.mkdir(exist_ok=False)
    return results_dir


def final_model(cfg, seed, budget, *, fidelity_result, par_path, results_dir,
                data_arg, args, incumbent):
    """Module-level target so pynisher can serialize it with spawn."""
    model_cfg = prepare_model_config(incumbent, args.puffed_up_rim)
    obm.run_mcfost(
        fidelity_result, par_path, results_dir, args.puffed_up_rim, model_cfg,
        alma_spec=(data_arg["alma"]["image_spec"]
                   if "alma" in fidelity_result["products"] else None))
    args.plot_intermediate = True
    return obm.load_and_score_outputs(fidelity_result, results_dir, data_arg, args, model_cfg)


def main():
    from lib.obriy_trial_timeout import (
        positive_seconds, timeout_cost, TrialTimeoutRunner,
        TimeoutAwareEncoder, TimeoutAwareHyperband,
    )

    p = argparse.ArgumentParser(description="SMAC3 observable-stage optimisation for MCFOST, with local workers")
    p.add_argument("--data-root", type=str, required=True, help="Path to observed-data root (for scoring) or 'demo'.")
    p.add_argument("--working-root", type=str, required=True, help="Working directory for trial runs")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--fidelity-config", type=str, default=None, help="Stage file; defaults to WORK_ROOT/fidelity.txt")
    p.add_argument("--config-space", type=str, default=None, help="Explicit ConfigSpace YAML path")
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--trial-walltime-limit", type=positive_seconds, default=None, metavar="SECONDS",
                   help="Optional whole-trial walltime limit in seconds, including MCFOST and scoring. Default: unlimited.")
    p.add_argument("--timeout-cost", type=timeout_cost, default=None, metavar="NUMBER|none",
                   help="Timeout policy: a finite positive penalty, or 'none' to exclude timeouts from surrogate training (default).")
    p.add_argument("--seed", type=int, default=-1)# Random seed for SMAC
    p.add_argument("--correct-unresolved-polarimetry", action="store_true", help="Use unresolved-corrected model images and profiles for polarimetry comparisons; requires --unresolved-correction-radius-px")
    p.add_argument("--warmstart", type=str, default=None, help="Path to previous SMAC run directory to warmstart from")
    p.add_argument("--plot-intermediate", action="store_true", help="Plot intermediate results during scoring")
    p.add_argument("--puffed-up-rim", action="store_true", help="Enable puffed up rim feature")
    p.add_argument("--2ZONE_CONT_LHC", dest="two_zone_cont_lhc", action="store_true",
                   help="Two contiguous zones (zone 1 power law; zone 2 power law or tapered): derive boundaries and dust masses from disk_Rmid [au] and disk_total_dust_mass [solar masses]")
    p.add_argument("--tapered-edge-p1-eq-p2", action="store_true", help="Tie p2 to p1 in the last zone of the MCFOST template")
    p.add_argument("--share-zone-composition", action="store_true",
                   help="Copy zone 1's entire dust population to zone 2, including species, materials, sizes and heating methods")
    p.add_argument("--overresolved_flux_fit_for_interferometry", type=float, default=None, help="Optional: fit overresolved flux to interferometry at the supplied wavelength [micron]. Disabled by default.")
    p.add_argument("--unresolved-correction-radius-px", type=float, default=None, help="Unresolved-polarisation aperture radius in instrument pixels; required when correction is enabled")
    #p.add_argument("--pdi-constraint-tolerances", type=float, nargs=3, default=(0.05, 0.05, 0.05), metavar=("FRACTION", "RADIAL", "QUADRANT"), help="Deprecated: ignored; PDI uses raw squared differences")
    #p.add_argument("--pdi-absolute-error-floors", type=float, nargs=3, default=(0., 0., 0.),
    #               metavar=("FRACTION", "RADIAL", "QUADRANT"), help="Deprecated: ignored; PDI uses raw squared differences")
    p.add_argument("--pdi-radial-bin-mas", type=float, default=25.0, help="Non-overlapping PDI radial bins [mas]")
    p.add_argument("--initial-design-seed", type=int, default=None, help="Seed for the initial design")
    args = p.parse_args()
    if args.timeout_cost is not None and args.trial_walltime_limit is None:
        p.error("--timeout-cost requires --trial-walltime-limit.")
    
    WORK_ROOT = Path(args.working_root).resolve()
    os.environ["SMAC_WORK_ROOT"] = str(WORK_ROOT)
    # (WORK_ROOT / "dask-tmp").mkdir(parents=True, exist_ok=True)

    print(f"[main] Working root: {WORK_ROOT}")


    fidelity_path = Path(args.fidelity_config) if args.fidelity_config else WORK_ROOT / "fidelity.txt"
    args.fidelity_settings = load_fidelity_config(fidelity_path)
    args.min_budget = args.fidelity_settings['stages'][0]['budget']
    args.max_budget = args.fidelity_settings['stages'][-1]['budget']
    manifest = WORK_ROOT / "fidelity_resolved.json"
    if manifest.exists() and json.loads(manifest.read_text()) != args.fidelity_settings:
        raise ValueError("Fidelity stages differ from this run's saved stages. Use a new working root.")
    manifest.write_text(json.dumps(args.fidelity_settings, indent=2))
    for stage in args.fidelity_settings['stages']:
        print(f"[main] Stage {stage['stage']}: budget={stage['budget']}, products={stage['products']}")

    if args.config_space:
        config_file = Path(args.config_space)
    else:
        config_candidates = sorted(path for path in list(WORK_ROOT.rglob("*.yaml")) + list(WORK_ROOT.rglob("*.yml"))
                                   if path.resolve() != fidelity_path.resolve())
        if len(config_candidates) != 1:
            raise ValueError("Specify --config-space: expected exactly one ConfigSpace YAML in working root.")
        config_file = config_candidates[0]
    if config_file.resolve() == fidelity_path.resolve():
        raise ValueError("ConfigSpace and fidelity configuration must be separate files.")

    print(f"[main] Using config space file: {config_file}")


    cs = build_configspace(config_file)
    validate_configuration_names(
        cs, WORK_ROOT / 'simulation.para',
        puffed_up_rim=args.puffed_up_rim,
        two_zone_cont_lhc=args.two_zone_cont_lhc)

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
        trial_walltime_limit=args.trial_walltime_limit,
        # You can also set output directories, logging, etc.
    )
    work_root = Path(args.working_root)
    products = fidelity_products(args.fidelity_settings)
    pdi_grid = None
    if any(product.startswith('pdi_') for product in products):
        pdi_grid = fixed_pdi_grid(work_root / 'simulation.para', cs)
    data_arg = load_data(args.data_root, str(work_root), products,
                         pdi_grid=pdi_grid, pdi_radial_bin_mas=args.pdi_radial_bin_mas)

    trial_folder= work_root/"trials/"
    trial_folder.mkdir(exist_ok=True)

    # Bind fixed inputs; SMAC supplies cfg, seed and budget on each call.
    # Unlike a nested function, this can be pickled by pynisher under spawn.
    smac_objective = partial(objective, data_arg=data_arg,
                             scratch_root=trial_folder, args=args)
    

    initial_design = SobolInitialDesign(
        scenario,
        n_configs=256,
        max_ratio=0.30,   # Allows 256 within a 1,000-trial run.
        seed=args.initial_design_seed,         # Initial-design seed, separate from optimiser seed.
    )
    
    smac_kwargs = dict(
        scenario=scenario,
        target_function=smac_objective,
        overwrite=False,
        initial_design=initial_design
    )

    smac_kwargs["intensifier"] = MultiFidelityFacade.get_intensifier(
        scenario, eta=args.fidelity_settings["eta"])
    if args.trial_walltime_limit is not None:
        smac_kwargs["target_function"] = TrialTimeoutRunner(
            scenario, smac_objective, trial_root=trial_folder,
            directory_factory=make_unique_config_trial_dir,
            timeout_penalty=args.timeout_cost, fidelity_settings=args.fidelity_settings)
        smac_kwargs["runhistory_encoder"] = TimeoutAwareEncoder(scenario)
        smac_kwargs["intensifier"] = TimeoutAwareHyperband(
            scenario, eta=args.fidelity_settings["eta"],
            incumbent_selection="highest_observed_budget")
        print(f"[main] Whole-trial walltime limit: {args.trial_walltime_limit} s; "
              f"timeout cost: {args.timeout_cost} (None means no fit score).")
    smac = MultiFidelityFacade(**smac_kwargs)

    if args.warmstart is not None:
        warmstart_from_runhistory_json(
            smac=smac,
            cs=cs,
            warmstart_path=args.warmstart,
            only_finished=True,
            max_trials=None,  # or set e.g. 50
            fidelity_settings=args.fidelity_settings,
        )

    # Optional: callbacks (e.g., log incumbent every K trials)
    # smac.register_callback(HPO.LoggingCallback())

    incumbent = smac.optimize()

    print("\n=== Optimization finished ===")
    print("Incumbent config:")
    print(incumbent)
    if incumbent is None:
        print("No eligible completed model. Trial statuses are saved under trials/; "
              "increase the walltime limit to evaluate timed-out configurations.")
        return

    # Start clean before writing the new incumbent or any simulation inputs.
    results_dir = prepare_final_results_dir(WORK_ROOT)
    with open(results_dir / "incumbent.json", "w") as f:
        json.dump(dict(incumbent) if hasattr(incumbent, "get_dictionary") else dict(incumbent), f, indent=2)

    # Convert cfg (ConfigSpace.Configuration) to dict
    if hasattr(incumbent, "get_dictionary"):
        incumbent = dict(incumbent)
    fidelity_result={}
    
    fidelity_result=map_budget_to_fidelity(args.max_budget, args.fidelity_settings)
    # Verify template parameter file for mcfost exists
    template_para= Path(results_dir.parent/"simulation.para")
    assert template_para.exists(), f"Missing template .para at {template_para}"

    # Write param file and run MCFOST
    par_path = obm.write_mcfost_paramfile(
        prepare_model_config(incumbent, args.puffed_up_rim), fidelity_result, results_dir,
        two_zone_cont_lhc=args.two_zone_cont_lhc,
        tapered_edge_p1_eq_p2=args.tapered_edge_p1_eq_p2,
        share_zone_composition=getattr(args, "share_zone_composition", False))
    final_target = partial(
        final_model, fidelity_result=fidelity_result, par_path=par_path,
        results_dir=results_dir, data_arg=data_arg, args=args, incumbent=incumbent)

    if args.trial_walltime_limit is not None:
        # Final regeneration can be slower (e.g. at the highest fidelity). Bound
        # it too, without replacing the incumbent's already measured trial cost.
        final_config = Configuration(cs, values=dict(incumbent))
        final_config.config_id = smac.runhistory.get_config_id(final_config)
        final_runner = TrialTimeoutRunner(
            scenario, final_target, trial_root=results_dir,
            directory_factory=lambda root, config_id, budget: root,
            timeout_penalty=args.timeout_cost, fidelity_settings=args.fidelity_settings)
        _, final_value = final_runner.run_wrapper(TrialInfo(final_config, seed=args.seed, budget=args.max_budget))
        if final_value.status.name != "SUCCESS":
            print(f"[main] Final regeneration ended with {final_value.status.name}; "
                  "preserving the optimisation history and original incumbent score.")
    else:
        final_target(incumbent, args.seed, args.max_budget)




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
    # Dask serializes __main__ functions by value. Pynisher subsequently uses
    # standard pickle under spawn, which cannot resolve that copied function.
    # Execute through the canonical import so all targets are module references.
    import multiprocessing as mp

    mp.set_start_method("spawn")

    from SMAC3 import main as run_main
    run_main()
