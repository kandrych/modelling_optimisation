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

import obriy_general as obg
import obriy_interferometry as obi
import obriy_sed as obs
import obriy_polarimetry as obp






constants.set_matplotlib_params()  # set project matplotlib parameters

# Ensure MCFOST is found in PATH on Katya's Mac
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]
os.environ["MCFOST_UTILS"] = os.path.expanduser("/Users/katerynaandrych/software/mcfost/utils")






#########################################
# MCFOST
##########################################

class ParaFile:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = []
        self.params = {}
        self.read()
        self._param_map = {
            "nbr_photons_eq_th": (3,0),
            "nbr_photons_lambda": (4,0),
            "nbr_photons_image": (5,0),
            "n_lambda": (8, 0),
            "lambda_min": (8, 1),
            "lambda_max": (8, 2),
            "compute_temp": (9, 0),
            "compute_sed": (9, 1),
            "use_default_lambda_grid": (9, 2),
            "wavelength_file": (10, 0),
            "separation_of_contributions":(11,0),
            "stokes parameters": (11, 1),
            "grid_geometry": (14, 0),
            "n_rad": (15, 0),
            "nz": (15, 1),
            "n_az": (15, 2),
            "n_rad_in": (15, 3),
            "grid_nx": (18, 0),
            "grid_ny": (18, 1),
            "grid_size": (18, 2),
            "imin": (19, 0),
            "imax": (19, 1),
            "n_incl": (19, 2),
            "centered": (19, 3),
            "az_min": (20, 0),
            "az_max": (20, 1),
            "n_az": (20, 2),
            "distance_pc": (21,0),
            "disk_pa": (22, 0),
            "scattering_mode": (25, 0),
            "image_symmetry": (28, 0),
            "central_symmetry": (29, 0),
            "axial_symmetry": (30, 0),
            "dust_settling": (33, 0),
            "exp_strat": (33, 1),
            "a_strat": (33, 2),
            "dust_radial_migration": (34, 0),
            "sublimate_dust": (35, 0),
            "hydrostatic_equilibrium": (36, 0),
            "viscous_heating": (37, 0),
            "alpha_viscosity": (37, 1),
            "number_of_zones": (40, 0)
        }
        num_zones = int(self.lines[self.find_line_starting_with("#Number of zones") + 1].split()[0])
        
        # Add density structure parameters for each zone
        start_line = self.find_line_starting_with("#Density structure")  # First zone starts 
        if start_line != -1:
            lines_per_zone = 7
            for i in range(num_zones):
                base = start_line +1+ i * lines_per_zone
                self._param_map[f"zone_{i+1}_type"] = (base + 0, 0)
                self._param_map[f"zone_{i+1}_dust_mass"] = (base + 1, 0)
                self._param_map[f"zone_{i+1}_gas_to_dust"] = (base + 1, 1)
                self._param_map[f"zone_{i+1}_scale_height"] = (base + 2, 0)
                self._param_map[f"zone_{i+1}_Rref"] = (base + 2, 1)
                self._param_map[f"zone_{i+1}_vertical_profile_exponent"] = (base + 2, 2)
                self._param_map[f"zone_{i+1}_Rin"] = (base + 3, 0)
                self._param_map[f"zone_{i+1}_edge"] = (base + 3, 1)
                self._param_map[f"zone_{i+1}_Rout"] = (base + 3, 2)
                self._param_map[f"zone_{i+1}_Rc"] = (base + 3, 3)
                self._param_map[f"zone_{i+1}_flaring_exp"] = (base +4,0)
                self._param_map[f"zone_{i+1}_surface_density_exp"] = (base +5,0)
                self._param_map[f"zone_{i+1}_-gamma_exp"] = (base +5,1)

        # Add grain properties for each zone
        start_line = self.find_line_starting_with("#Grain properties") 
        if start_line != -1:
            
            base = start_line + 1
            lines_per_species = 4
            for i in range(num_zones):
                self._param_map[f"zone_{i+1}_number_of_species"] = (base + 0, 0)
                num_species = int(self.lines[base + 0].split()[0])
                for j in range(num_species):
                    basej = base 
                    self._param_map[f"zone_{i+1}_species_{j+1}_grain_type"] = (basej + 1, 0)
                    self._param_map[f"zone_{i+1}_species_{j+1}_N_components"] = (basej + 1, 1)
                    self._param_map[f"zone_{i+1}_species_{j+1}_mixing_rule"] = (basej + 1, 2)
                    self._param_map[f"zone_{i+1}_species_{j+1}_porosity"] = (basej + 1, 3)
                    self._param_map[f"zone_{i+1}_species_{j+1}_mass_fraction"] = (basej + 1, 4)
                    self._param_map[f"zone_{i+1}_species_{j+1}_Vmax"] = (basej + 1, 5)
                    self._param_map[f"zone_{i+1}_species_{j+1}_optical_indices_file"] = (basej + 2, 0)
                    self._param_map[f"zone_{i+1}_species_{j+1}_volume_fraction"] = (basej + 2, 1)
                    self._param_map[f"zone_{i+1}_species_{j+1}_heating_method"] = (basej + 3, 0)
                    self._param_map[f"zone_{i+1}_species_{j+1}_amin"] = (basej +4,0)
                    self._param_map[f"zone_{i+1}_species_{j+1}_amax"] = (basej +4,1)
                    self._param_map[f"zone_{i+1}_species_{j+1}_aexp"] = (basej +4,2)
                    self._param_map[f"zone_{i+1}_species_{j+1}_n_grains"] = (basej +4,3)
                base = basej+4 +2
                
        
        #Add star properties
        start_line= self.find_line_starting_with("#Star properties")
        num_stars = int(self.lines[self.find_line_starting_with("#Star properties") + 1].split()[0])
        self._param_map["number_of_stars"] = (start_line + 1, 0)
        # Add  parameters for each star
        if start_line != -1:
            lines_per_star = 4
            for i in range(num_stars):
                base = start_line +2+ i * lines_per_star
                self._param_map[f"star_{i+1}_Temp"] = (base + 0, 0)
                self._param_map[f"star_{i+1}_R"] = (base + 0, 1)
                self._param_map[f"star_{i+1}_M"] = (base + 0, 2)
                self._param_map[f"star_{i+1}_x"] = (base + 0, 3)
                self._param_map[f"star_{i+1}_y"] = (base + 0, 4)
                self._param_map[f"star_{i+1}_z"] = (base + 0, 5)
                self._param_map[f"star_{i+1}_autometic_spectrum"] = (base +0, 6)
                self._param_map[f"star_{i+1}_spectrum_file"] = (base + 1, 0)
                self._param_map[f"star_{i+1}_fUV"] = (base + 2, 0)
                self._param_map[f"star_{i+1}_slope_fUV"] = (base + 2, 1)
                
        
        self._extract_params()

        


    def read(self):
        with open(self.filepath, "r") as f:
            self.lines = f.readlines()
        

    def _extract_params(self):
        for name, (line_no, col_no) in self._param_map.items():
            line = self.lines[line_no].strip()
            # print(line.split())
            value = line.split()[col_no]
            self.params[name] = value
   
    def find_line_starting_with(self, prefix):
        for i, line in enumerate(self.lines):
            if line.strip().startswith(prefix):
                return i
        return -1  # not found
        


    def set_param(self, param_name, new_value):
        if param_name not in self._param_map:
            raise ValueError(f"Unknown parameter name: {param_name}")
        line_no = self._param_map[param_name][0]
        col_no = self._param_map[param_name][1]
        old_line = self.lines[line_no]
        parts = old_line.split()
        parts[col_no] = str(new_value)
        self.lines[line_no] = "  "+"  ".join(parts) + "\n"
        self.params[param_name] = new_value

    def save(self, out_path):
        with open(out_path, "w") as f:
            f.writelines(self.lines)

    def print_params(self):
        for k, v in self.params.items():
            print(f"{k}: {v}")



def run_mcfost_image(wavelength, folder):
    """
    Python wrapper to run MCFOST for a given wavelength and simulation folder

    Parameters:
    wavelength: wavelength in micrometer
    folder_sim: folder name where the simulation.para file and data_th folder is located

    Returns: 
    None
    """
    if not os.path.exists(folder):
        raise ValueError(f"Folder {folder} does not exist. Please check the folder name and try again.")
    if not os.path.exists(folder+"data_th/sed_rt.fits.gz"):
        raise ValueError(f"SED file does not exist in {folder+'data_th/'}. Please run MCFOST simulation first.")

    
    os.chdir(folder+'/') #this is to change directory to where the simulation.para file is and then run mcfost there
    mcfost.run(folder+'/simulation.para',options = "-img "+str(wavelength), delete_previous=False, silent=True)




def run_mcfost_chi2(param, keys, data_arg, pybads_dir):
    """
    Run MCFOST for given parameters, calculate chi2 for SED and interferometric data
    Parameters
    ----------
    param : list
        List of parameters to set in the simulation.para file.
    keys:
        which parameters are in param. Names correspond to MCFOST parameter file
    data_arg : list
        List containing data for SED and interferometric observations. First element is SED data (tuple of wavelength, flux, error),
        second element is PIONIER data (OIContainer), third element is GRAVITY data (OIContainer),
        fourth element is MATISSE L-band data (OIContainer), fifth element is MATISSE N-band data (OIContainer).
    pybads_dir
        Location where we should run simulation
    Returns
    -------
    chi_total: float
        Total chi2
    chi2_red_total : float
        Total reduced chi2 value for SED and interferometric data.
    loglike: float
        Joint Log-likelihood for optimisation

    """
    #print(param)
    data_sed = data_arg[0]
    container_data_pionier = data_arg[1]
    container_data_gravity = data_arg[2]
    container_data_matisse_l = data_arg[3]
    container_data_matisse_n = data_arg[4]

    pf = ParaFile(pybads_dir+"simulation.para")
    folder_sim=""
    for i in range(0,len(keys)):
        pf.set_param(keys[i], param[i])
        folder_sim+=keys[i]+"_"+str(param[i])+"_"
    
   
     
    simulation_dir = pybads_dir+folder_sim+"/"
    # if not os.path.exists(simulation_dir):
    os.makedirs(simulation_dir, exist_ok=True)  # no error if it already exists

    # Save the modified file
    pf.save(simulation_dir+"simulation.para")
    try:
        #WORKS
        os.chdir(simulation_dir) #this is to change directory to where the simulation.para file is and then run mcfost there
        mcfost.run(simulation_dir+'/simulation.para', delete_previous=True, silent=True)

        for wave in [1.63, 2.20, 3.50, 10.0]:
            mcfost.run(simulation_dir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
    except:
        print(f"MCFOST run failed for parameters: {keys} = {param}")
        return 10e6
    # if os.path.exists(simulation_dir):
    #     print('path exists')
    #     if not os.path.exists(simulation_dir+'data_th/sed_rt.fits.gz'):
    #         pf.save(simulation_dir+"simulation.para")

    #         os.chdir(simulation_dir) #this is to change directory to where the simulation.para file is and then run mcfost there
    #         mcfost.run(simulation_dir+'/simulation.para', delete_previous=True, silent=True)
    #     else:
    #         print('SED file exists')
    #     for wave in [1.63, 2.20, 3.50, 10.0]:
    #         if not os.path.exists(simulation_dir+'data_'+str(wave)+'/'+'RT.fits.gz'):
    #             mcfost.run(simulation_dir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
    #         else:
    #             print('Image at '+str(wave)+' micron exists')

        
    chi2_sed, chi2_reduced_sed, loglike_sed= obs.chi2_SED_with_reddening(folder_sim, pybads_dir, data_wave=data_sed[0], data_flux=data_sed[1],data_err=data_sed[2],
                                       plot=True, description=f"{keys} = {param}")
    
    chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.monochromatic_chi(simulation_dir, img_dir="data_1.63/", container_data=container_data_pionier, vistype='vis2', plot=True, fig_dir=simulation_dir+'figures/', extra_title="PIONIER 1.63", log_plotv=False)
    chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.monochromatic_chi(simulation_dir, img_dir="data_2.2/", container_data=container_data_gravity, vistype='vis2', plot=True, fig_dir=simulation_dir+'figures/', extra_title="GRAVITY 2.2", log_plotv=False)
    chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.monochromatic_chi(simulation_dir, img_dir="data_3.5/", container_data=container_data_matisse_l,vistype='vis2', plot=True, fig_dir=simulation_dir+'figures/', extra_title="MATISSE L 3.5", log_plotv=True)
    chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.monochromatic_chi(simulation_dir, img_dir="data_10.0/", container_data=container_data_matisse_n, vistype='vis', plot=True, fig_dir=simulation_dir+'figures/', extra_title="MATISSE N 10.0", log_plotv=False)
    
    
    chi_total= chi2_sed + chi2_pionier + chi2_gravity + chi2_matisse_l + chi2_matisse_n
    num_points_total= len(data_sed[0]) + num_points_pionier + num_points_gravity + num_points_matisse_l + num_points_matisse_n
    
    chi2_red_total = chi_total/(num_points_total-5)
    loglike_total=loglike_sed+loglike_pionier+loglike_gravity+loglike_matisse_l+loglike_matisse_n
   
    return  chi_total, chi2_red_total, loglike_total 