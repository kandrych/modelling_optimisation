
import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict, Optional, Union, Any, List


import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astropy.io import fits
import astropy.units as u
from IPython.display import display

import subprocess
from skimage.transform import rescale, resize, downscale_local_mean
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft, AiryDisk2DKernel
import fnmatch
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import curve_fit
import random
import json

import astropy
from astropy import units as u
import astropy.units.quantity
from astropy.io import fits
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable



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

import lib.obriy_general as obg
import lib.obriy_interferometry as obi
import lib.obriy_sed as obs
import lib.obriy_polarimetry as obp
import lib.obriy_alma as oba


import shutil, subprocess
from pathlib import Path






#constants.set_matplotlib_params()  # set project matplotlib parameters
os.environ.setdefault("MCFOST_NO_UPDATE", "1") # prevent MCFOST from checking for updates every time it is run within this script


# # Ensure MCFOST is found in PATH on Katya's Mac
# os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]
# os.environ["MCFOST_UTILS"] = os.path.expanduser("/Users/katerynaandrych/software/mcfost/utils")




plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.edgecolor"] = "grey"
plt.rcParams["legend.framealpha"] = 0.5
plt.rcParams["lines.markersize"] = 6.0
plt.rcParams["lines.linewidth"] = 2.0

plt.rc("font", size=16)  # controls default text sizes
plt.rc("axes", titlesize=14)  # fontsize of the axes title
plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
plt.rc("ytick", labelsize=14)  # fontsize of the tick labels
plt.rc("legend", fontsize=14)  # legend fontsize
plt.rc("figure", titlesize=14)  # fontsize of the figure title



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
            try:
                value = line.split()[col_no]
            except IndexError:
                raise IndexError(f"Could not find column {col_no} in line {line_no} for parameter {name}. Line content: '{line}'")
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



def run_mcfost_safe(param_path: Path, workdir: Path, options: list[str] = None,
                    logfile: str | None = None) -> None:
    
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    exe = shutil.which("mcfost")
    assert exe, "mcfost not found in PATH"

    cmd = [exe, str(param_path)]
    if options:
        cmd += options

    # stream to file if requested
    if logfile:
        with open(workdir / logfile, "w") as f:
            subprocess.run(cmd, cwd=workdir, check=True, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        subprocess.run(cmd, cwd=workdir, check=True)




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
    
    if os.path.exists(folder+"data_"+str(wavelength)+"/"):   
        print(f"Image at {wavelength} micron already exists in {folder+'data_'+str(wavelength)+'/'} folder. Skipping simulation.")
    
    else:
        run_mcfost_safe(Path(folder+'/simulation.para'), Path(folder), options=["-img", f"{wavelength}"])

        


def plot_mcfost_disk_structure(main_dir: str, sub_dir: str,  az_disk=0) -> None:

    """
    Plot the disk structure from an MCFOST parameter file.

    Parameters
    ----------
    main_dir : str
        Main directory where the simulation folder is located.
    sub_dir : str
        Subdirectory within the main directory where the simulation files are located.
    az_disk : int, optional
        Azimuthal zone to consider for the disk structure (default is 0).
    Returns
    -------
    None
    plots two (general and zoomed in inner rim) 2D cuts of the disk structure for temperature, dust mass density, and gas density.
    """
    simulation_dir = main_dir+sub_dir+'/'
    
    #make folder to store images if needed
    if os.path.exists(simulation_dir+"figures/")==False:
        os.system("mkdir "+simulation_dir+"figures/")


    #open the required grid structure file
    hdul=fits.open(simulation_dir+'data_disk/grid.fits.gz')
    grid_struct=hdul[0].data
    #note we don't check for dimensionality because it's always 4D for this file
    #extract cylindrical radius and height above midplane of computation grid cell
    r = grid_struct[0, az_disk, :, :]
    z = grid_struct[1, az_disk, :, :]


    paths = [
    ('data_th/Temperature.fits.gz', r'$T \, \mathrm{[K]}$'),
    ('data_disk/dust_mass_density.fits.gz', r'$\rho_{dust} \, \mathrm{[g\,cm^{-3}]}$'),
    ('data_disk/gas_density.fits.gz', r'$\rho_{gas} \, \mathrm{[g\,cm^{-3}]}$')
    ]

    quantities = []

    # Read each file only once
    for path, label in paths:
        with fits.open(simulation_dir + path) as hdul:
            q = hdul[0].data

        if q.ndim > 2:
            q = q[az_disk]
        quantities.append((q, label))




    fig, ax = plt.subplots(len(quantities), 1, figsize=(9, 5*len(quantities)))

    for i, (quantity, label) in enumerate(quantities):
        #check if the quantity has any positive finite values
        q = np.asarray(quantity)
        valid = np.isfinite(q) & (q > 0)
        if not np.any(valid):
            ax[i].text(
                0.5, 0.5,
                "No positive finite values to plot",
                ha="center", va="center",
                transform=ax[i].transAxes,
            )
            ax[i].set_title(label)
            continue

        vmin = np.nanmin(q[valid])
        vmax = np.nanmax(q[valid])

        if vmin >= vmax:
            ax[i].text(
                0.5, 0.5,
                "Constant values; cannot use LogNorm",
                ha="center", va="center",
                transform=ax[i].transAxes,
            )
            ax[i].set_title(label)
            continue

        cmesh = ax[i].pcolormesh(
            r, z, q,
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="nearest"
        )
        
        cb = plt.colorbar(cmesh, ax=ax[i])
        cb.set_label(label)

        ax[i].set_xlabel(r'$r\,[AU]$')
        ax[i].set_ylabel(r'$z\,[AU]$')
        ax[i].set_xlim(0, np.max(r))
    plt.suptitle('2D cut disk structure')
    plt.tight_layout()
    fig.savefig(simulation_dir+'figures/disk_structure2D'+'.png', dpi= 150, bbox_inches='tight')
    plt.close()
    


    fig, ax = plt.subplots(len(quantities), 1, figsize=(9, 5*len(quantities)))

    for i, (quantity, label) in enumerate(quantities):

        #q_norm = quantity / colmax[None, :]
        q = np.asarray(quantity)
        vmin = np.nanmin(q[(r < 100) & (z < 100) & np.isfinite(q) & (q > 0)])
        vmax = np.nanmax(q[(r < 100) & (z < 100) & np.isfinite(q) & (q > 0)])
        cmesh = ax[i].pcolormesh(r, z, q, cmap='viridis',shading="nearest", vmin=vmin, vmax=vmax)


        cb = plt.colorbar(cmesh, ax=ax[i])
        cb.set_label(label)

        ax[i].set_xlabel(r'$r\,[AU]$')
        ax[i].set_ylabel(r'$z\,[AU]$')
        ax[i].set_xlim(0, 100)
        ax[i].set_ylim(0, 100)
    plt.suptitle('Zoomed inner rim 2D cut disk structure')
    #save the plot
    plt.tight_layout()
    fig.savefig(simulation_dir+'figures/disk_structure2D_zoomed'+'.png', dpi= 150, bbox_inches='tight')
    plt.close()




# def plot_mcfost_disk_structure(main_dir: str, sub_dir: str,  az_disk=0) -> None:

#     """
#     Plot the disk structure from an MCFOST parameter file.

#     Parameters
#     ----------
#     main_dir : str
#         Main directory where the simulation folder is located.
#     sub_dir : str
#         Subdirectory within the main directory where the simulation files are located.
#     az_disk : int, optional
#         Azimuthal zone to consider for the disk structure (default is 0).
#     Returns
#     -------
#     None
#     plots two (general and zoomed in inner rim) 2D cuts of the disk structure for temperature, dust mass density, and gas density.
#     """
#     simulation_dir = main_dir+sub_dir+'/'
    
#     #make folder to store images if needed
#     if os.path.exists(simulation_dir+"figures/")==False:
#         os.system("mkdir "+simulation_dir+"figures/")

#     #open the required grid structure file
#     hdul=fits.open(simulation_dir+'data_disk/grid.fits.gz')
#     grid_struct=hdul[0].data
#     #note we don't check for dimensionality because it's always 4D for this file
#     #extract cylindrical radius and height above midplane of computation grid cell
#     r = grid_struct[0, az_disk, :, :]
#     z = grid_struct[1, az_disk, :, :]

#     #open the required temperature, dust and gas mass density fits files and plot them
#     paths=['data_th/Temperature.fits.gz', 'data_disk/dust_mass_density.fits.gz', 'data_disk/gas_density.fits.gz']
#     fig, ax = plt.subplots(len(paths), 1, figsize=(9, 5*len(paths)))
#     for i, path in enumerate(paths):
#         hdul=fits.open(simulation_dir+path)
#         quantity_struct=hdul[0].data
#         #see if we're dealing with different azimuthal zones by checking dimesnionality
#         #of file & select data out of hdu accordingly
#         if quantity_struct.ndim > 2:
#             quantity = quantity_struct[az_disk, :, :]
#         else:
#             quantity = quantity_struct[:, :]
#         #plotting
#         cmesh = ax[i].pcolormesh(r, z, quantity, cmap='viridis', norm=LogNorm())
#         cb = plt.colorbar(cmesh, ax=ax[i])
#         if path == 'data_th/Temperature.fits.gz':
#             cb.set_label(r'$T \, \mathrm{[K]}$')
#         elif path == 'data_disk/dust_mass_density.fits.gz':
#             cb.set_label(r'$\rho_{dust} \, \mathrm{[g \, cm^{-3}]}$')
#         elif path == 'data_disk/gas_density.fits.gz':
#             cb.set_label(r'$\rho_{gas} \, \mathrm{[g \, cm^{-3}]}$')
#         ax[i].set_xlabel(r'$r \, \mathrm{[AU]}$')
#         #ax[i].set_yscale('log')

#         ax[i].set_ylabel(r'$z \, \mathrm{[AU]}$')
#         ax[i].set_xlim(0, np.max(r))
#     plt.suptitle('2D cut disk structure')
#     plt.tight_layout()
#     fig.savefig(simulation_dir+'figures/disk_structure2D'+'.png', dpi= 150, bbox_inches='tight')

#     paths=['data_th/Temperature.fits.gz', 'data_disk/dust_mass_density.fits.gz', 'data_disk/gas_density.fits.gz']
#     fig, ax = plt.subplots(len(paths), 1, figsize=(9, 5*len(paths)))
#     for i, path in enumerate(paths):
#         hdul=fits.open(simulation_dir+path)
#         quantity_struct=hdul[0].data
#         if quantity_struct.ndim > 2:
#             quantity = quantity_struct[az_disk, :, :]
#         else:
#             quantity = quantity_struct[:, :]
#         colmax = np.nanmax(quantity, axis=0)
#         colmax[colmax == 0] = np.nan
#         q_norm = quantity / colmax[None, :] #normalize each column to its maximum value

#         #q_norm = quantity / np.nanmax(quantity, axis=0)[None, :] #normalize each column to its maximum value
#         #plotting
#         cmesh = ax[i].pcolormesh(r, z, q_norm, cmap='viridis')
#         cb = plt.colorbar(cmesh, ax=ax[i])
#         if path == 'data_th/Temperature.fits.gz':
#             cb.set_label(r'Normalized $T \, \mathrm{[K]}$')
#         elif path == 'data_disk/dust_mass_density.fits.gz':
#             cb.set_label(r'Normalized $\rho_{dust} \, \mathrm{[g \, cm^{-3}]}$')
#         elif path == 'data_disk/gas_density.fits.gz':
#             cb.set_label(r'$\rho_{gas} \, \mathrm{[g \, cm^{-3}]}$')
#         ax[i].set_xlabel(r'$r \, \mathrm{[AU]}$')
#         #ax[i].set_yscale('log')

#         ax[i].set_ylabel(r'$z \, \mathrm{[AU]}$')
#         ax[i].set_xlim(0, 20)
#         ax[i].set_ylim(0, 20)
        
#     plt.suptitle('Zoomed inner rim 2D cut disk structure')
#     #save the plot
#     plt.tight_layout()
#     fig.savefig(simulation_dir+'figures/disk_structure2D_zoomed'+'.png', dpi= 150, bbox_inches='tight')






def write_mcfost_paramfile(cfg: Dict[str, Any], fidelity: Dict[str, Any], outdir: Path) -> Path:
    """
    test
    Materialize an MCFOST parameter file in `outdir` from the sampled configuration.
    Returns the path to the written .para file.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    param_path = outdir / "model.para"

    # Json dump of config + fidelity for record-keeping
    with open(outdir / "config_used.json", "w") as f:
        json.dump({"cfg": cfg, "fidelity": fidelity}, f, indent=2)
    # Load a base .para file template from folder that was passed as working root
    print(outdir.parent.name)
    if outdir.parent.name != "trials":
        try :
            pf = ParaFile(str(outdir.parent/"simulation.para"))
        except:
            raise ValueError("Base MCFOST parameter file not found in the working directory. Please ensure 'simulation.para' exists.")
    else:    
        try:
            pf = ParaFile(str(outdir.parent.parent/"simulation.para"))
        except:
            raise ValueError("Base MCFOST parameter file not found in the working directory. Please ensure 'simulation.para' exists.")
    
    for key in cfg.keys():
        if key in pf.params:
            pf.set_param(key, cfg[key])
            if key=='zone_1_Rin':
                pf.set_param('zone_1_Rref', cfg[key]) #scale-height is setted up at the inner rim.
        elif key=='inclination':
            pf.set_param('imin', cfg[key])
            pf.set_param('imax', cfg[key])
        elif key.startswith('puffed_'):
            #puffed up rim parameters are set during run_mcfost function
            continue
        
        else:
            print(f"Warning: parameter {key} not found in MCFOST parameter file.")
    # Set fidelity-related params
    # pf.set_param("nbr_photons_eq_th", fidelity["nbr_photons_eq_th"])
    # pf.set_param("nbr_photons_lambda", fidelity["nbr_photons_lambda"])
    # pf.set_param("nbr_photons_image", fidelity["nbr_photons_image"])
    
    # Save the modified file
    pf.save(param_path)
    
    return param_path


def run_mcfost(fidelity: dict, param_path: Path, workdir: Path, puffed_up_rim: bool=False, cfg: Dict[str, Any]={}) -> None:

    print(f"run mcfost in {workdir}")
    # base
    print(fidelity)

    
    if puffed_up_rim:
        run_mcfost_safe(param_path, workdir, options=["-disk_struct","-puffed_up_rim", f"{cfg.get('puffed_h_rim_over_h0', 0)}", f"{cfg.get('puffed_r_rim', 0)}", f"{cfg.get('puffed_delta_r', 0)}"], logfile="mcfost_base_struct.log")
        run_mcfost_safe(param_path, workdir, options=["-puffed_up_rim", f"{cfg.get('puffed_h_rim_over_h0', 0)}", f"{cfg.get('puffed_r_rim', 0)}", f"{cfg.get('puffed_delta_r', 0)}"], logfile="mcfost_temp.log")
               
    else:
        
        run_mcfost_safe(param_path, workdir, options=["-disk_struct"], logfile="mcfost_base_struct.log")
        run_mcfost_safe(param_path, workdir, logfile="mcfost_temp.log")
    
    if "vis2_1perband" in fidelity["products"]:
        for w in [1.63, 2.20, 3.50, 10.0]:
            if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
                print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
                continue
    
            print(f"Running MCFOST for vis2_1perband at {w} micron")
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")
    if "pdi_V" in fidelity["products"]:
        for w in [0.55]:
            if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
                print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
                continue
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")
    if "pdi_I" in fidelity["products"]:
        for w in [0.82]:
            if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
                print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
                continue
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")
    if "pdi_H" in fidelity["products"]:
        for w in [1.63]:
            if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
                print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
                continue
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")
    
    if "alma" in fidelity["products"]:
        for w in [870.0]:
            if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
                print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
                continue
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}", "-casa"], logfile=f"mcfost_{w:.2f}.log")

    if "vis2_chromatic" in fidelity["products"]: # all wavelengths for chromatic visibilities (PIONIER, MATISSE, GRAVITY) + full PDI
        for w in [0.55, 0.82, 1.5,1.55,1.6,1.63,1.65,1.7,1.75,1.8,1.85,1.9,
                  1.95,2.0,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,
                  2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,
                  7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0]:
            if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
                print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
                continue
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")




def load_and_score_outputs(fidelity: Dict[str, Any], workdir: Path, data_arg:Dict[str, Any], args) -> Tuple[float, Dict[str, Any]]:
    """
    Read MCFOST outputs and compute a single scalar loss.
    

    Return the likelihood (lower is better) and additional_info dictionary.
    Additional info contains extra information, such as individual chi2 values for each observable.
    """
    print(f'[obriy_mcfost] fidelity["stage"] = {fidelity["stage"]}')
    print(f'[obriy_mcfost] fidelity["products"] = {fidelity["products"]}')  
    additional_info = {}

    if args.plot_intermediate:
        if not os.path.exists(str(workdir)+'/figures/'):
            os.makedirs(str(workdir)+'/figures/')

    if "sed" in fidelity["products"]:
       data_sed = data_arg[0]

    if ("vis2_1perband" in fidelity["products"]) or  ("vis2_chromatic" in fidelity["products"]):
        container_data_pionier = data_arg[1]
        container_data_gravity = data_arg[2]
        container_data_matisse_l = data_arg[3]
        container_data_matisse_n = data_arg[4]

    if "pdi_V" in fidelity["products"]:
        pdi_data_v = data_arg[5] #each disc with data not deconvolved q_phi, u_phi, pi, and psf
    if "pdi_I" in fidelity["products"]:
        pdi_data_i = data_arg[6] 
    if "pdi_H" in fidelity["products"]:
        pdi_data_h = data_arg[7]
    if "alma" in fidelity["products"]:
        data_alma = data_arg[8]
    
    print('[obriy_mcfost] Data for scoring loaded successfully')

    simulation_name = workdir.name


    sed_path = workdir / "data_th" / "sed_rt.fits.gz"
    if not sed_path.exists():
        print(f"Temperature file {sed_path} not found.")
        # trial ran but produced no SED -> invalid config or earlier failure
        return 1e99, additional_info
    plot_mcfost_disk_structure(str(workdir.parent)+'/', simulation_name,  az_disk=0)

    ebminv_sed=0.0
    if "sed" in fidelity["products"]:

        chi2_sed, chi2_reduced_sed, loglike_sed, ebminv_sed= obs.chi2_SED_with_reddening(str(workdir.name), str(workdir.parent)+'/', data_wave=data_sed[0], data_flux=data_sed[1],data_err=data_sed[2],
                                                plot=True, description=simulation_name)
        additional_info["sed"] = {
            "chi2": chi2_sed,
            "chi2_reduced": chi2_reduced_sed,
            "loglike": loglike_sed,
            "ebminv": ebminv_sed
        }
    
    if "vis2_1perband" in fidelity["products"]:
        
        if args.overresolved_flux_fit_for_interferometry:
            full_wavelengths=[1.63, 2.2, 3.5, 10.0]
            wave_for_background=args.overresolved_flux_fit_for_interferometry
            closest = min(full_wavelengths, key=lambda x: abs(x - wave_for_background))
            container_data=container_data_pionier if closest in pionier_wavelengths else container_data_gravity if closest in gravity_wavelengths else container_data_matisse_l if closest in matisse_l_wavelengths else container_data_matisse_n
            img_dir = f"data_{closest}/"
            _, _, _, _, frac_closest_wavelength_optimised= obi.monochromatic_chi_with_background(str(workdir), img_dir=img_dir, container_data=container_data, wave_for_background=wave_for_background, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title=f"Reference for overresolved flux, wavelength {closest}", log_plotv=False)
                

            chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier, _= obi.monochromatic_chi_with_background(str(workdir), img_dir="data_1.63/", container_data=container_data_pionier, wave_for_background=args.overresolved_flux_fit_for_interferometry,frac_for_background=frac_closest_wavelength_optimised, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER 1.63", log_plotv=False)
            chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity, _= obi.monochromatic_chi_with_background(str(workdir), img_dir="data_2.2/", container_data=container_data_gravity, wave_for_background=args.overresolved_flux_fit_for_interferometry, frac_for_background=frac_closest_wavelength_optimised, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY 2.2", log_plotv=False)
            chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l, _= obi.monochromatic_chi_with_background(str(workdir), img_dir="data_3.5/", container_data=container_data_matisse_l, wave_for_background=args.overresolved_flux_fit_for_interferometry,frac_for_background=frac_closest_wavelength_optimised,  vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L 3.5", log_plotv=True)
            chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n, _= obi.monochromatic_chi_with_background(str(workdir), img_dir="data_10.0/", container_data=container_data_matisse_n, wave_for_background=args.overresolved_flux_fit_for_interferometry, frac_for_background=frac_closest_wavelength_optimised, vistype='vis', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N 10.0", log_plotv=False)

        else:
            chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.monochromatic_chi(str(workdir), img_dir="data_1.63/", container_data=container_data_pionier, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER 1.63", log_plotv=False)
            chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.monochromatic_chi(str(workdir), img_dir="data_2.2/", container_data=container_data_gravity, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY 2.2", log_plotv=False)
            chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.monochromatic_chi(str(workdir), img_dir="data_3.5/", container_data=container_data_matisse_l,vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L 3.5", log_plotv=True)
            chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.monochromatic_chi(str(workdir), img_dir="data_10.0/", container_data=container_data_matisse_n, vistype='vis', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N 10.0", log_plotv=False)
            frac_closest_wavelength_optimised=None
            wave_for_background=None
            
        additional_info["vis2_1perband"] = {
            'pionier':{
                "chi2": chi2_pionier,
                "chi2_reduced": chi2_red_pionier,
                "loglike": loglike_pionier,
                "num_points": num_points_pionier
            },
            'gravity':{
                "chi2": chi2_gravity,
                "chi2_reduced": chi2_red_gravity,
                "loglike": loglike_gravity,
                "num_points": num_points_gravity
            },
                
            'matisse_l':{
                "chi2": chi2_matisse_l,
                "chi2_reduced": chi2_red_matisse_l,
                "loglike": loglike_matisse_l,
                "num_points": num_points_matisse_l
                
            },
            'matisse_n':{
                "chi2": chi2_matisse_n,
                "chi2_reduced": chi2_red_matisse_n,
                "loglike": loglike_matisse_n,
                "num_points": num_points_matisse_n
            },
            'fraction_overresolved': {"value": frac_closest_wavelength_optimised,
                                                  "wavelength": wave_for_background}

        }

    if "vis2_chromatic" in fidelity["products"]:
        
        pionier_wavelengths = [1.5,1.55,1.6,1.63,1.65,1.7,1.75,1.8,1.85, 1.9]
        gravity_wavelengths = [1.95,2.0,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5]
        matisse_l_wavelengths = [2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3]
        matisse_n_wavelengths = [7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0]
        full_wavelengths = pionier_wavelengths + gravity_wavelengths + matisse_l_wavelengths + matisse_n_wavelengths
                              
        if args.overresolved_flux_fit_for_interferometry:
            wave_for_background=args.overresolved_flux_fit_for_interferometry
            closest = min(full_wavelengths, key=lambda x: abs(x - wave_for_background))
            container_data=container_data_pionier if closest in pionier_wavelengths else container_data_gravity if closest in gravity_wavelengths else container_data_matisse_l if closest in matisse_l_wavelengths else container_data_matisse_n
            img_dir = f"data_{closest}/"
            _, _, _, _, frac_closest_wavelength_optimised= obi.monochromatic_chi_with_background(str(workdir), img_dir=img_dir, container_data=container_data, wave_for_background=wave_for_background, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title=f"Reference for overresolved flux, wavelength {closest}", log_plotv=False)
                        
        else:
            wave_for_background=None
            frac_closest_wavelength_optimised=None
                

        pionier_img_dirs=[f"data_{w}/" for w in pionier_wavelengths]
        chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.chromatic_chi(str(workdir), img_dir=pionier_img_dirs, container_data=container_data_pionier, vistype='vis2',wave_for_background=wave_for_background,frac_for_background=frac_closest_wavelength_optimised, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER", log_plotv=False, ebminv=ebminv_sed)
        gravity_img_dirs=[f"data_{w}/" for w in gravity_wavelengths]
        chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.chromatic_chi(str(workdir), img_dir=gravity_img_dirs, container_data=container_data_gravity, vistype='vis2',wave_for_background=wave_for_background,frac_for_background=frac_closest_wavelength_optimised, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY", log_plotv=False, ebminv=ebminv_sed)
        matisse_l_img_dirs=[f"data_{w}/" for w in matisse_l_wavelengths]
        chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.chromatic_chi(str(workdir), img_dir=matisse_l_img_dirs, container_data=container_data_matisse_l,vistype='vis2',wave_for_background=wave_for_background,frac_for_background=frac_closest_wavelength_optimised, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L", log_plotv=True, ebminv=ebminv_sed)
        matisse_n_img_dirs=[f"data_{w}/" for w in matisse_n_wavelengths]
        chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.chromatic_chi(str(workdir), img_dir=matisse_n_img_dirs, container_data=container_data_matisse_n, vistype='vis',wave_for_background=wave_for_background,frac_for_background=frac_closest_wavelength_optimised, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N", log_plotv=False, ebminv=ebminv_sed)
        
        additional_info["vis2_chromatic"] = {
            'pionier':{
                "chi2": chi2_pionier,
                "chi2_reduced": chi2_red_pionier,
                "loglike": loglike_pionier,
                "num_points": num_points_pionier,
                "wavelengths": pionier_wavelengths
            },
            'gravity':{
                "chi2": chi2_gravity,
                "chi2_reduced": chi2_red_gravity,
                "loglike": loglike_gravity,
                "num_points": num_points_gravity,
                "wavelengths": gravity_wavelengths
            },
            'matisse_l':{
                "chi2": chi2_matisse_l,
                "chi2_reduced": chi2_red_matisse_l,
                "loglike": loglike_matisse_l,
                "num_points": num_points_matisse_l,
                "wavelengths": matisse_l_wavelengths
            },
            'matisse_n':{
                "chi2": chi2_matisse_n,
                "chi2_reduced": chi2_red_matisse_n,
                "loglike": loglike_matisse_n,
                "num_points": num_points_matisse_n,
                "wavelengths": matisse_n_wavelengths
            },
            'fraction_overresolved': {"value": frac_closest_wavelength_optimised,
                                      "wavelength": wave_for_background}
        }
        
    if ("pdi_I" in fidelity["products"]) or ("pdi_V" in fidelity["products"]) or ("pdi_H" in fidelity["products"]):
        loss_i=np.nan
        loss_v=np.nan
        loss_h=np.nan

        print('[obriy_mcfost] Polarimetric analysis started')
        if "pdi_I" in fidelity["products"]:
    
            results_i=obp.polarimetric_analysis(str(workdir), 0.82, camera='zimpol',convolution_mode='file', psf_array=pdi_data_i['psf'], psf_cut=100, 
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Iband')
            
            if args.correct_unresolved_polarimetry:
                print('[obriy_mcfost] Applying unresolved polarization correction for I band')
                data_cropped_i, model_cropped_i= obp.crop_to_same_size(pdi_data_i['pol_images']['Q_phi'], results_i['mcfost_convolved_unresolved_corrected']['q_phi'])
                model_rad_prof= results_i['mcfost_convolved_unresolved_corrected']['radial_profiles']['q_phi']
                model_azimuthal_prof= results_i['mcfost_convolved_unresolved_corrected']['azimuthal_profiles']['q_phi']
                
            else:
                print('[obriy_mcfost] No unresolved polarization correction applied for I band')
                data_cropped_i, model_cropped_i= obp.crop_to_same_size(pdi_data_i['pol_images']['Q_phi'], results_i['mcfost_convolved']['q_phi'])
                model_rad_prof= results_i['mcfost_convolved']['radial_profiles']['q_phi']
                model_azimuthal_prof= results_i['mcfost_convolved']['azimuthal_profiles']['q_phi']
                   
            # Calculate metrics for arcsinh-scaled images to highlight morphology
            obs_rad_prof_pi, obs_az_prof_pi = pdi_data_i['radial_profiles']['Q_phi'], pdi_data_i['azimuthal_profiles']['Q_phi']
            
            profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof_pi, model_rad_prof, 3.6, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_i_")
            profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof, 3.6, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_i_")
            profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike


            metrics_i = obp.full_image_metrics_noshift(
                np.arcsinh(data_cropped_i), np.arcsinh(model_cropped_i),
                normalize="zscore",          # good default for morphology
                ssim_win=None,                 # 7–15 is typical
                return_pixel_chi2=True
            )
            if args.plot_intermediate:
                obp.plot_polarimetric_image(results_i['mcfost_convolved_unresolved_corrected']['q_phi_deconvolved'], 3.6, title=f'Model Qphi, conv, unres corr, decon', save=str(workdir)+'/figures'+'/model_q_phi_corr_conv_deconv_I.png', image_scale='asinh', roi_half_size=100)
                obp.plot_polarimetric_image(results_i['mcfost_convolved']['q_phi_deconvolved'], 3.6, title=f'Model Qphi, conv, decon', save=str(workdir)+'/figures'+'/model_q_phi_conv_deconv_I.png', image_scale='asinh', roi_half_size=100)

                obp.plot_polarimetric_image(metrics_i["ssim_image"], 3.6, title=f'ssim, score {metrics_i["ssim"]}', save=str(workdir)+'/figures'+'/ssim_image_I.png', image_scale='linear', roi_half_size=50)

            obp.save_band_metrics(
                        workdir,
                        band="I",
                        analysis_metrics=results_i['mcfost_convolved_unresolved_corrected']['metrics'],
                        ssim_score=metrics_i.get("ssim"),
                        ncc_score=metrics_i.get("ncc"),
                        extras={"ps_mas": 3.6, "notes": "zscore"}
                        )
            
            if args.plot_intermediate:
                images_list = [np.arcsinh(data_cropped_i), np.arcsinh(model_cropped_i)]
                    
                titles = ['Data', 'Model']
                
                fig, axs = obp.plot_image_grid(
                                images=images_list,
                                ps_mas=3.6,
                                nrows=1,
                                ncols=2,
                                titles=titles,
                                group_headers=[(0.5, 'I-band')],
                                scale="linear",
                                roi_half_size=60,          
                                per_panel_autoscale=True,
                                normalize_image=True,
                                colorbar="individual",
                                figsize=(8, 4),
                                show=False
                                )
                fig.savefig(str(workdir)+'/figures'+'/i_data_model_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
            print(f'[obriy_mcfost] I band metrics: SSIM={metrics_i["ssim"]}, NCC={metrics_i["ncc"]}, profile_pi_chi2_red={profile_pi_chi2_red}')
            loss_i=1-(metrics_i['ssim']+metrics_i['ncc'])/2 #+ profile_pi_chi2_red # weights can be adjusted
            #Loss based only on profile chi2 to test if it can drive the fit
            #loss_i=profile_pi_chi2_red

            additional_info["pdi"] = {'I': {
                "ssim": metrics_i.get("ssim"),
                "ncc": metrics_i.get("ncc"),
                "profile_pi_chi2_red": profile_pi_chi2_red,
                "profile_pi_loglike": profile_loglike,
                "profile_rad_pi_chi2": profile_rad_pi_chi2,
                "profile_rad_pi_npoints": profile_rad_pi_npoints,
                "profile_az_pi_chi2": profile_az_pi_chi2,
                "profile_az_pi_npoints": profile_az_pi_npoints,
                "loss": loss_i
            }}


        if "pdi_V" in fidelity["products"]:
    
            results_v=obp.polarimetric_analysis(str(workdir), 0.55, camera='zimpol',convolution_mode='file', psf_array=pdi_data_v['psf'],psf_cut=100, 
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Vband')
            if args.plot_intermediate:
                obp.plot_polarimetric_image(results_v['mcfost_convolved_unresolved_corrected']['q_phi_deconvolved'], 3.6, title=f'Model Qphi, conv, unres corr, decon', save=str(workdir)+'/figures'+'/model_q_phi_corr_conv_deconv_V.png', image_scale='asinh', roi_half_size=100)
                obp.plot_polarimetric_image(results_v['mcfost_convolved']['q_phi_deconvolved'], 3.6, title=f'Model Qphi, conv, decon', save=str(workdir)+'/figures'+'/model_q_phi_conv_deconv_V.png', image_scale='asinh', roi_half_size=100)
            
            if args.correct_unresolved_polarimetry:
                print('[obriy_mcfost] Applying unresolved polarization correction for I band')
                data_cropped_v, model_cropped_v= obp.crop_to_same_size(pdi_data_v['pol_images']['Q_phi'], results_v['mcfost_convolved_unresolved_corrected']['q_phi'])
                model_rad_prof= results_v['mcfost_convolved_unresolved_corrected']['radial_profiles']['q_phi']
                model_azimuthal_prof= results_v['mcfost_convolved_unresolved_corrected']['azimuthal_profiles']['q_phi']
                 
            else:
                print('[obriy_mcfost] No unresolved polarization correction applied for I band')
                data_cropped_v, model_cropped_v= obp.crop_to_same_size(pdi_data_v['pol_images']['Q_phi'], results_v['mcfost_convolved']['q_phi'])   
                model_rad_prof= results_v['mcfost_convolved']['radial_profiles']['q_phi']
                model_azimuthal_prof= results_v['mcfost_convolved']['azimuthal_profiles']['q_phi']
            #CHANGE HERE for profiles that are already calculated in loading data initially to avoid recalculating them and speed up the process
            obs_rad_prof, obs_az_prof= pdi_data_v['radial_profiles']['Q_phi'], pdi_data_v['azimuthal_profiles']['Q_phi']
            
            profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof, model_rad_prof, 3.6, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_v_")
            profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof, model_azimuthal_prof, 3.6, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_v_")
            profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike

            metrics_v = obp.full_image_metrics_noshift(
                np.arcsinh(data_cropped_v), np.arcsinh(model_cropped_v),
                normalize="zscore",          # good default for morphology
                ssim_win=None,                 # 7–15 is typical
                return_pixel_chi2=True
            )
            if args.plot_intermediate:
                obp.plot_polarimetric_image(metrics_v["ssim_image"], 3.6, title=f'ssim, score {metrics_v["ssim"]}', save=str(workdir)+'/figures'+'/ssim_image_V.png', image_scale='linear', roi_half_size=50)

            obp.save_band_metrics(
                        workdir,
                        band="V",
                        analysis_metrics=results_v['mcfost_convolved_unresolved_corrected']['metrics'],
                        ssim_score=metrics_v.get("ssim"),
                        ncc_score=metrics_v.get("ncc"),
                        extras={"ps_mas": 3.6, "notes": "zscore"}
                        )
            if args.plot_intermediate:
                images_list = [np.arcsinh(data_cropped_v), np.arcsinh(model_cropped_v)]
                    
                titles = ['Data', 'Model']
                
                fig, axs = obp.plot_image_grid(
                                images=images_list,
                                ps_mas=3.6,
                                nrows=1,
                                ncols=2,
                                titles=titles,
                                group_headers=[(0.5, 'V-band')],
                                scale="linear",
                                roi_half_size=60,          
                                per_panel_autoscale=True,
                                normalize_image=True,
                                colorbar="individual",
                                figsize=(8, 4),
                                show=False
                                )
                fig.savefig(str(workdir)+'/figures'+'/v_data_model_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
            print(f'[obriy_mcfost] V band metrics: SSIM={metrics_v["ssim"]}, NCC={metrics_v["ncc"]}, profile_pi_chi2_red={profile_pi_chi2_red}')
            loss_v=1-(metrics_v['ssim']+metrics_v['ncc'])/2 #+ profile_pi_chi2_red # weights can be adjusted
            #Loss based only on profile chi2 to test if it can drive the fit
            #loss_v= profile_pi_chi2_red
            additional_info["pdi"] = {'V': {
                            "ssim": metrics_v.get("ssim"),
                            "ncc": metrics_v.get("ncc"),
                            "profile_pi_chi2_red": profile_pi_chi2_red,
                            "profile_pi_loglike": profile_loglike,
                            "profile_rad_pi_chi2": profile_rad_pi_chi2,
                            "profile_rad_pi_npoints": profile_rad_pi_npoints,
                            "profile_az_pi_chi2": profile_az_pi_chi2,
                            "profile_az_pi_npoints": profile_az_pi_npoints,
                            "loss": loss_v
                        }}

        if "pdi_H" in fidelity["products"]:
            results_h=obp.polarimetric_analysis(str(workdir), 1.63, camera='irdis',convolution_mode='file', psf_array=pdi_data_h['psf'],psf_cut=100, 
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Hband')
            
           
            if args.correct_unresolved_polarimetry:
                print('[obriy_mcfost] Applying unresolved polarization correction for H band')
                data_cropped_h, model_cropped_h= obp.crop_to_same_size(pdi_data_h['pol_images']['Q_phi'], results_h['mcfost_convolved_unresolved_corrected']['q_phi']) 
                model_rad_prof= results_h['mcfost_convolved_unresolved_corrected']['radial_profiles']['q_phi']
                model_azimuthal_prof= results_h['mcfost_convolved_unresolved_corrected']['azimuthal_profiles']['q_phi']
            else:
                print('[obriy_mcfost] No unresolved polarization correction applied for H band')
                data_cropped_h, model_cropped_h= obp.crop_to_same_size(pdi_data_h['pol_images']['Q_phi'], results_h['mcfost_convolved']['q_phi']) 
                model_rad_prof= results_h['mcfost_convolved']['radial_profiles']['q_phi']
                model_azimuthal_prof= results_h['mcfost_convolved']['azimuthal_profiles']['q_phi']
            
            obs_rad_prof_pi, obs_az_prof_pi = pdi_data_h['radial_profiles']['Q_phi'], pdi_data_h['azimuthal_profiles']['Q_phi']     
            profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof_pi, model_rad_prof, 12.27, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_h_")
            profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof, 12.27, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_h_")
            profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike


            metrics_h = obp.full_image_metrics_noshift(
                np.arcsinh(data_cropped_h), np.arcsinh(model_cropped_h),
                normalize="zscore",          # good default for morphology
                ssim_win=None,                 # 7–15 is typical
                return_pixel_chi2=True
            )
            if args.plot_intermediate:
                obp.plot_polarimetric_image(metrics_h["ssim_image"], 12.27, title=f'ssim, score {metrics_h["ssim"]}', save=str(workdir)+'/figures'+'/ssim_image_H.png', image_scale='linear', roi_half_size=30)
            obp.save_band_metrics(
                        workdir,
                        band="H",
                        analysis_metrics=results_h['mcfost_convolved_unresolved_corrected']['metrics'],
                        ssim_score=metrics_h.get("ssim"),
                        ncc_score=metrics_h.get("ncc"),
                        extras={"ps_mas": 12.27, "notes": "zscore"}
                        )
        # except Exception as e:
        #     print(f"Error in H-band polarimetric analysis: {e}")
        #     data_cropped_h= np.zeros((10,10))
        #     model_cropped_h= np.zeros((10,10))
        #     metrics_h={'ssim':-1.0,'ncc':-1.0} #        
            if args.plot_intermediate:
                images_list = [data_cropped_h, model_cropped_h]
                titles = [
                        'Data', 'Model']
                
                fig, axs = obp.plot_image_grid(
                                images=images_list,
                                ps_mas=12.27,
                                nrows=1,
                                ncols=2,
                                titles=titles,
                                group_headers=[(0.5, 'H-band')],
                                scale="linear",
                                roi_half_size=50,          
                                per_panel_autoscale=True,
                                normalize_image=True,
                                colorbar="individual",
                                figsize=(8, 4),
                                show=False
                                )
                fig.savefig(str(workdir)+'/figures'+'/h_data_model_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()  
            print(f'[obriy_mcfost] H band metrics: SSIM={metrics_h["ssim"]}, NCC={metrics_h["ncc"]}, profile_pi_chi2_red={profile_pi_chi2_red}')
            loss_h=1-(metrics_h['ssim']+metrics_h['ncc'])/2 #+ profile_pi_chi2_red # weights can be adjusted
            #Loss based only on profile chi2 to test if it can drive the fit
            #loss_h=profile_pi_chi2_red # weights can be adjusted
            additional_info["pdi"] = {'H': {
                            "ssim": metrics_h.get("ssim"),
                            "ncc": metrics_h.get("ncc"),
                            "profile_pi_chi2_red": profile_pi_chi2_red,
                            "profile_pi_loglike": profile_loglike,
                            "profile_rad_pi_chi2": profile_rad_pi_chi2,
                            "profile_rad_pi_npoints": profile_rad_pi_npoints,
                            "profile_az_pi_chi2": profile_az_pi_chi2,
                            "profile_az_pi_npoints": profile_az_pi_npoints,
                            "loss": loss_h
                        }}




        
         
        print(f"PDI losses: I-band: {loss_i}, V-band: {loss_v}, H-band: {loss_h}")
    


    if "alma" in fidelity["products"]:
        
        alma_cont = data_alma['alma_cont']
        obs_rad_prof = data_alma['radial_profile']
        obs_az_prof = data_alma['azimuthal_profile']
        ps_alma = data_alma['ps_alma']
        data_size_alma = data_alma['image_size']
        wave=data_alma['alma_wavelength']
        mask_alma = data_alma['mask_alma']

        
        _, _, simulated_itot, pix_scale = oba.load_mcfost_image_alma_casa(str(workdir), '870.0')  
        
        if args.plot_intermediate: obp.plot_polarimetric_image(simulated_itot, ps_alma, title=f'Model Itot, alma_cont', save=str(workdir)+'/figures/'+'/model_itot_alma.png', image_scale='asinh', roi_half_size=100)
        simulated_itot_resc=oba.rescale_alma(simulated_itot, pix_scale, ps_alma)

        if simulated_itot_resc.shape[0] > alma_cont.shape[0]:
            simulated_itot_as_data=oba.cut_down_alma(simulated_itot_resc, alma_cont)
        elif simulated_itot_resc.shape[0] < alma_cont.shape[0]:
            print(f"[obriy_mcfost] WARNING: simulated_itot_resc.shape[0] < alma_cont.shape[0], cutting down alma_cont to match simulated_itot_resc")
            simulated_itot_as_data=simulated_itot_resc
            alma_cont=oba.cut_down_alma(alma_cont,simulated_itot_as_data)
            mask_alma=oba.cut_down_alma(mask_alma,simulated_itot_as_data)
        else:
            simulated_itot_as_data=simulated_itot_resc


        residuals_map=(simulated_itot_as_data-alma_cont)**2
        residuals_map_masked = np.where(mask_alma, residuals_map, np.nan)
        residuals_reduced=np.nansum(residuals_map_masked)/np.sum(mask_alma) #does not have error estimate, so not a proper chi2, but takes into account only 3snr points
        print(f"[obriy_mcfost] ALMA residuals image snr>=2 = {residuals_reduced}, sum of mask = {np.sum(mask_alma)}")
        print(f"[obriy_mcfost] ALMA total flux: data = {np.nansum(alma_cont)}, model = {np.nansum(simulated_itot_as_data)}")

    
        if args.plot_intermediate:
            #do some plotting
            fig, ax = plt.subplots(1, 3, figsize=(16,6))
            fig.subplots_adjust(wspace=0.5)

            color_map = 'viridis' #'afmhot'
            im0=ax[0].imshow(alma_cont, color_map, extent=[+alma_cont.shape[0]/2, -alma_cont.shape[0]/2, -alma_cont.shape[1]/2, alma_cont.shape[1]/2])
            ax[0].set_title("Data I$_{tot}$")
            obg.add_colorbar(fig, ax[0], im0)
            im1=ax[1].imshow(simulated_itot_as_data, color_map, extent=[+simulated_itot_as_data.shape[0]/2, -simulated_itot_as_data.shape[0]/2, -simulated_itot_as_data.shape[1]/2, simulated_itot_as_data.shape[1]/2])
            ax[1].set_title('Simulated I$_{tot}$')
            obg.add_colorbar(fig, ax[1], im1)
            im2=ax[2].imshow(residuals_map_masked, color_map,extent=[+alma_cont.shape[0]/2, -alma_cont.shape[0]/2, -alma_cont.shape[1]/2, alma_cont.shape[1]/2])
            ax[2].set_title("Residual I$_{tot}$")
            obg.add_colorbar(fig, ax[2], im2)
            plt.suptitle("ALMA, "+str(wave)+"$\mu m$, reduces chi2 "+ residuals_reduced.astype(str)) #does not have error estimate, so not a proper chi2
            fig.savefig(str(workdir)+'_alma_sim_vs_data_'+str(wave)+'.png', dpi= 150, bbox_inches='tight')
            plt.close(fig)



        metrics_alma = obp.full_image_metrics_noshift(
                alma_cont, simulated_itot_as_data,
                normalize="zscore",          # good default for morphology
                ssim_win=None,                 # 7–15 is typical
                return_pixel_chi2=True
            )
        
        try:
            chi2_red_alma_profiles, profile_rad_pi_chi2, profile_az_pi_chi2, profile_rad_pi_npoints, profile_az_pi_npoints = oba.chi2_ALMA(str(workdir), data_alma=data_alma, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_ALMA_')
        except Exception as e:
            print(f"Error computing ALMA chi2: {e}")
            chi2_red_alma_profiles = 1e99
        
        #loss_alma= 1-(metrics_alma['ssim']+metrics_alma['ncc'])/2 #
        loss_alma=residuals_reduced 
        #loss_alma=chi2_red_alma_profiles

        additional_info["alma"] = {
            "ssim": metrics_alma.get("ssim"),
            "ncc": metrics_alma.get("ncc"),
            "chi2_red_alma_profiles": chi2_red_alma_profiles,
            "profile_rad_pi_chi2": profile_rad_pi_chi2,
            "profile_az_pi_chi2": profile_az_pi_chi2,
            "profile_rad_pi_npoints": profile_rad_pi_npoints,
            "profile_az_pi_npoints": profile_az_pi_npoints,
            "residuals_reduced": residuals_reduced,
            "loss": loss_alma
        }

        print(f'[obriy_mcfost] ALMA metrics: SSIM={metrics_alma["ssim"]}, NCC={metrics_alma["ncc"]}, chi2_red_alma_profiles={chi2_red_alma_profiles}, residuals image snr>=3 = {residuals_reduced}')
            
        #print(f"ALMA chi2: {chi2_red_alma}")
    
    #initialize totals so eve if there is no sed and interferometry - we can still compute pdi only chi2
    chi_total=0.0
    num_points_total=0
    loglike_total=0.0
    chi2_red_total=0.0
    i_num=0
    
    if "sed" in fidelity["products"]:
        
        chi_total+= chi2_sed 
        num_points_total+= len(data_sed[0])
        loglike_total+=loglike_sed
        i_num=1
        if num_points_total==0 or num_points_total-i_num==0:
            print("No or just 1 SED data points found. Setting chi2 to 0.")
            chi_total=0.0
            chi2_red_total = 0.0
        else:
            chi2_red_total = chi2_reduced_sed

    if ("vis2_1perband" in fidelity["products"]) or ("vis2_chromatic" in fidelity["products"]):
        
        chi_total= chi2_pionier + chi2_gravity + chi2_matisse_l + chi2_matisse_n
        num_points_total=num_points_pionier + num_points_gravity + num_points_matisse_l + num_points_matisse_n
        loglike_total+=loglike_pionier+loglike_gravity+loglike_matisse_l+loglike_matisse_n
        i_num=4

        chi2_red_total += chi2_red_pionier+chi2_red_gravity+chi2_red_matisse_l+chi2_red_matisse_n
        #chi_total/(num_points_total-i_num)  # reduced chi2 - not sure about number of free parameters here
    
    if "pdi_I" in fidelity["products"]:
        chi2_red_total+=(loss_i)#*100 # weighting factor to bring SSIM losses to similar scale as chi2
   
    
    if "pdi_V" in fidelity["products"]:
        chi2_red_total+=(loss_v)#*100 # weighting factor to bring SSIM losses to similar scale as chi2
   
    if "pdi_H" in fidelity["products"]:
        chi2_red_total+=(loss_h)#*100 # weighting factor to bring SSIM losses to similar scale as chi2
   
        #chi2_red_total=metrics_i["chi2_red"]+metrics_v["chi2_red"] #sum of reduced chi2 values for I and V bands for AR Pup fitting
    
    if "alma" in fidelity["products"]:
        chi2_red_total+=loss_alma 
    print(f"Total reduced chi2: {chi2_red_total}, loglike: {loglike_total}")
    
    return chi2_red_total, additional_info
