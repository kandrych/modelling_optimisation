
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
from scipy.integrate import quad
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
import distroi


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
        # These values define the number of following records.  Editing a
        # token cannot safely create or remove those records, so keep the
        # grain-section structure fixed by the input template.
        self._structural_params = set()
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
        self._structural_params.add("number_of_zones")
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

        # Add grain properties for each zone.  A species occupies a variable
        # number of rows: one header, N_components material rows, one heating
        # row and one grain-size row.  Do not use fixed line offsets here.
        start_line = self.find_line_starting_with("#Grain properties")
        if start_line != -1:
            cursor = self._next_parameter_line(start_line + 1)
            for i in range(num_zones):
                zone_key = f"zone_{i + 1}_number_of_species"
                self._param_map[zone_key] = (cursor, 0)
                self._structural_params.add(zone_key)
                num_species = int(self.lines[cursor].split()[0])
                cursor = self._next_parameter_line(cursor + 1)

                for j in range(num_species):
                    species_prefix = f"zone_{i + 1}_species_{j + 1}"
                    header_line = cursor
                    self._param_map[f"{species_prefix}_grain_type"] = (header_line, 0)
                    component_key = f"{species_prefix}_N_components"
                    self._param_map[component_key] = (header_line, 1)
                    self._structural_params.add(component_key)
                    self._param_map[f"{species_prefix}_mixing_rule"] = (header_line, 2)
                    self._param_map[f"{species_prefix}_porosity"] = (header_line, 3)
                    self._param_map[f"{species_prefix}_mass_fraction"] = (header_line, 4)
                    self._param_map[f"{species_prefix}_Vmax"] = (header_line, 5)

                    n_components = int(self.lines[header_line].split()[1])
                    if n_components < 1:
                        raise ValueError(f"{component_key} must be at least one.")
                    cursor = self._next_parameter_line(header_line + 1)
                    for k in range(n_components):
                        component_prefix = f"{species_prefix}_component_{k + 1}"
                        self._param_map[f"{component_prefix}_optical_indices_file"] = (cursor, 0)
                        self._param_map[f"{component_prefix}_volume_fraction"] = (cursor, 1)
                        cursor = self._next_parameter_line(cursor + 1)

                    self._param_map[f"{species_prefix}_heating_method"] = (cursor, 0)
                    cursor = self._next_parameter_line(cursor + 1)
                    self._param_map[f"{species_prefix}_amin"] = (cursor, 0)
                    self._param_map[f"{species_prefix}_amax"] = (cursor, 1)
                    self._param_map[f"{species_prefix}_aexp"] = (cursor, 2)
                    self._param_map[f"{species_prefix}_n_grains"] = (cursor, 3)
                    cursor = self._next_parameter_line(cursor + 1)
                
        
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

    def _next_parameter_line(self, start):
        """Return the next non-empty, non-heading line in a parameter block."""
        for line_no in range(start, len(self.lines)):
            stripped = self.lines[line_no].strip()
            if stripped and not stripped.startswith("#"):
                return line_no
        raise ValueError("Unexpected end of MCFOST parameter file.")
        


    def set_param(self, param_name, new_value):
        if param_name not in self._param_map:
            raise ValueError(f"Unknown parameter name: {param_name}")
        if param_name in self._structural_params:
            current_value = self.params[param_name]
            if str(new_value) != str(current_value):
                raise ValueError(
                    f"{param_name} defines the MCFOST grain-section layout and must "
                    "remain fixed. Use a template with the required species and "
                    "material-component records."
                )
            return
        line_no = self._param_map[param_name][0]
        col_no = self._param_map[param_name][1]
        old_line = self.lines[line_no]
        parts = old_line.split()
        parts[col_no] = str(new_value)
        self.lines[line_no] = "  "+"  ".join(parts) + "\n"
        for name, location in self._param_map.items():
            if location == (line_no, col_no):
                self.params[name] = new_value

    def save(self, out_path, *, share_zone_composition=False):
        lines = self.lines
        if share_zone_composition:
            if int(self.params["number_of_zones"]) != 2:
                raise ValueError("Shared composition requires exactly two zones.")
            # Copy complete records, not just counts: zones may have different
            # numbers of species and material components in the input template.
            start_1 = self._param_map["zone_1_number_of_species"][0]
            start_2 = self._param_map["zone_2_number_of_species"][0]
            last_species = int(self.params["zone_2_number_of_species"])
            end_2 = self._param_map[f"zone_2_species_{last_species}_n_grains"][0] + 1
            lines = lines[:start_2] + lines[start_1:start_2] + lines[end_2:]
        with open(out_path, "w") as f:
            f.writelines(lines)

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

    # Close/flush the log before collecting diagnostics from a failed command.
    try:
        if logfile:
            with open(workdir / logfile, "w") as f:
                subprocess.run(cmd, cwd=workdir, check=True, stdout=f, stderr=subprocess.STDOUT, text=True)
        else:
            subprocess.run(cmd, cwd=workdir, check=True)
    except (subprocess.CalledProcessError, OSError) as error:
        details = {"command": cmd}
        if isinstance(error, subprocess.CalledProcessError):
            details["exit_code"] = error.returncode
        if logfile:
            log_path = workdir / logfile
            details["log_path"] = str(log_path)
            try:
                # Bound run-history size even when MCFOST produces a large log.
                with log_path.open("rb") as log:
                    log.seek(0, os.SEEK_END)
                    log.seek(max(0, log.tell() - 8192))
                    details["log_tail"] = "\n".join(
                        log.read().decode("utf-8", errors="replace").splitlines()[-30:])
            except OSError as log_error:
                details["log_read_error"] = str(log_error)
        error.mcfost_details = details
        raise




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
        run_mcfost_safe(Path(folder+'/model.para'), Path(folder), options=["-img", f"{wavelength}"])

        


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






def _apply_selected_second_component_fractions(pf: ParaFile, cfg: Dict[str, Any]) -> None:
    """Set a two-material mixture from the active conditional configuration.

    For every sampled ``*_component_2_optical_indices_file`` key, exactly one
    active key named ``*_component_2_volume_fraction_<label>`` is required.
    Its range and its association with a material are defined solely in the
    ConfigSpace conditions.  Component 1 receives the complementary fraction.
    """
    material_keys = [
        key for key in cfg if key.endswith("_component_2_optical_indices_file")
    ]
    for material_key in material_keys:
        component_2_prefix = material_key.removesuffix("_optical_indices_file")
        conditional_fraction_prefix = f"{component_2_prefix}_volume_fraction_"
        active_fraction_keys = [
            key for key in cfg if key.startswith(conditional_fraction_prefix)
        ]
        if len(active_fraction_keys) != 1:
            raise ValueError(
                f"{material_key} requires exactly one active conditional fraction "
                f"named {conditional_fraction_prefix}<label>; got {active_fraction_keys}."
            )

        component_1_prefix = component_2_prefix.replace(
            "_component_2", "_component_1"
        )
        component_1_fraction_key = f"{component_1_prefix}_volume_fraction"
        component_2_fraction_key = f"{component_2_prefix}_volume_fraction"
        required_keys = (material_key, component_1_fraction_key, component_2_fraction_key)
        missing_keys = [key for key in required_keys if key not in pf.params]
        if missing_keys:
            raise ValueError(
                "The simulation template must contain both material-component "
                f"records before optimizing this mixture; missing: {missing_keys}."
            )

        second_fraction = float(cfg[active_fraction_keys[0]])
        pf.set_param(material_key, cfg[material_key])
        pf.set_param(component_1_fraction_key, 1.0 - second_fraction)
        pf.set_param(component_2_fraction_key, second_fraction)


def _2zone_cont_calc_dmass(cfg_dict) -> None:
    """Derive contiguous zone boundaries and dust masses in place.

    Radii are in au and dust masses in solar masses. Profiles are normalised
    to the same surface density at disk_Rmid; their slopes may differ.
    Zone 1 is a power law; zone 2 may be a power law or exponentially tapered.
    """
    total_mass = float(cfg_dict["disk_total_dust_mass"])
    r_mid = float(cfg_dict["disk_Rmid"])
    r_in = float(cfg_dict["zone_1_Rin"])
    r_out = float(cfg_dict["zone_2_Rout"])
    exponent_in = float(cfg_dict["zone_1_surface_density_exp"])
    exponent_out = float(cfg_dict["zone_2_surface_density_exp"])
    if not np.all(np.isfinite([
        total_mass, r_in, r_mid, r_out, exponent_in, exponent_out
    ])):
        raise ValueError("2ZONE_CONT_LHC requires finite masses, radii and exponents.")
    if total_mass <= 0 or not 0 < r_in < r_mid < r_out:
        raise ValueError("2ZONE_CONT_LHC requires positive total mass and 0 < Rin < Rmid < Rout.")
    outer_type = float(cfg_dict.get("zone_2_type", 1))
    if float(cfg_dict.get("zone_1_type", 1)) != 1 or outer_type not in (1, 2):
        raise ValueError("2ZONE_CONT_LHC requires zone 1 type=1 and zone 2 type=1 or 2.")

    # M_i = 2*pi*Sigma_mid*Rmid**2 * integral(x**(s_i+1), dx).
    # expm1 avoids cancellation near s_i == -2, whose integral is logarithmic.
    def radial_integral(lower, upper, exponent):
        power = exponent + 2.0
        log_span = np.log(upper / lower)
        if power == 0.0:
            return log_span
        return np.exp(power * np.log(lower)) * np.expm1(power * log_span) / power

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        integral_in = radial_integral(r_in / r_mid, 1.0, exponent_in)
        if outer_type == 1:
            integral_out = radial_integral(1.0, r_out / r_mid, exponent_out)
        else:
            r_c = float(cfg_dict["zone_2_Rc"])
            taper_power = 2.0 + float(cfg_dict["zone_2_-gamma_exp"])
            if not np.all(np.isfinite([r_c, taper_power])) or r_c <= 0 or taper_power <= 0:
                raise ValueError("Tapered zone 2 requires finite Rc > 0 and -gamma_exp > -2.")
            taper_at_boundary = (r_mid / r_c) ** taper_power
            if not np.isfinite(taper_at_boundary):
                raise ValueError("Tapered zone 2 boundary normalisation overflowed.")

            def tapered_integrand(log_x):
                # x=r/Rmid, t=log(x): x**(s+1) dx = exp((s+2)*t) dt.
                # Subtract the boundary taper so Sigma_2(Rmid)=Sigma_mid.
                taper = taper_at_boundary * np.expm1(taper_power * log_x)
                return np.exp((exponent_out + 2.0) * log_x - taper)

            result = quad(tapered_integrand, 0.0, np.log(r_out / r_mid),
                          epsabs=0.0, epsrel=1e-9, limit=200, full_output=1)
            integral_out, error = result[:2]
            if len(result) != 3 or not np.isfinite(error) or error > 1e-7 * integral_out:
                raise ValueError("Tapered zone 2 mass integral did not converge accurately.")
        integral_total = integral_in + integral_out
    if (not np.all(np.isfinite([integral_in, integral_out, integral_total]))
            or integral_in <= 0 or integral_out <= 0):
        raise ValueError("2ZONE_CONT_LHC mass integrals are invalid or overflowed.")
    mass_in = total_mass * (integral_in / integral_total)
    mass_out = total_mass * (integral_out / integral_total)
    if mass_in <= 0 or mass_out <= 0:
        raise ValueError("2ZONE_CONT_LHC derived mass underflowed to zero.")
    cfg_dict.update(zone_1_Rout=r_mid, zone_2_Rin=r_mid,
                    zone_1_dust_mass=float(mass_in), zone_2_dust_mass=float(mass_out))


def write_mcfost_paramfile(cfg: Dict[str, Any], fidelity: Dict[str, Any], outdir: Path,
                          *, two_zone_cont_lhc: bool = False,
                          tapered_edge_p1_eq_p2: bool = False,
                          share_zone_composition: bool = False) -> Path:
    """
    test
    Materialize an MCFOST parameter file in `outdir` from the sampled configuration.
    Returns the path to the written .para file.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    param_path = outdir / "model.para"

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
    
    if share_zone_composition:
        if int(pf.params["number_of_zones"]) != 2:
            raise ValueError("--share-zone-composition requires exactly two template zones.")
        conflicting = [key for key in cfg if key.startswith("zone_2_species_")
                       or key == "zone_2_number_of_species"]
        if conflicting:
            raise ValueError("Shared composition derives zone 2 dust properties; remove from config: "
                             + ", ".join(sorted(conflicting)))

    height_1 = "zone_1_scale_height"
    height_2 = "zone_2_scale_height"
    if cfg.get(height_1) == height_2:
        raise ValueError("Only zone_2_scale_height may reference zone_1_scale_height, not the reverse.")
    for target, source in ((height_2, height_1),):
        if cfg.get(target) != source:
            continue
        if "zone_2_scale_height" not in pf.params:
            raise ValueError("The scale-height tie requires a zone 2 in the template.")
        source_height = cfg.get(source, pf.params[source])
        try:
            shared_height = float(source_height)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source} must be numeric for the scale-height tie.") from error
        if not np.isfinite(shared_height) or shared_height <= 0:
            raise ValueError(f"{source} must be finite and positive for the scale-height tie.")
        cfg = dict(cfg)
        cfg[target] = shared_height

    flaring_1 = "zone_1_flaring_exp"
    flaring_2 = "zone_2_flaring_exp"
    if cfg.get(flaring_1) == flaring_2 and cfg.get(flaring_2) == flaring_1:
        raise ValueError("Circular flaring-exponent tie: one zone must supply a numeric exponent.")
    for target, source in ((flaring_2, flaring_1), (flaring_1, flaring_2)):
        if cfg.get(target) != source:
            continue
        if flaring_2 not in pf.params:
            raise ValueError("The flaring-exponent tie requires a zone 2 in the template.")
        try:
            shared_flaring = float(cfg.get(source, pf.params[source]))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source} must be numeric for the flaring-exponent tie.") from error
        if not np.isfinite(shared_flaring):
            raise ValueError(f"{source} must be finite for the flaring-exponent tie.")
        # With equal reference heights and radii, equal flaring exponents
        # give the same H(r) on either side of the zone boundary.
        cfg = dict(cfg)
        cfg[target] = shared_flaring

    if tapered_edge_p1_eq_p2:
        cfg = dict(cfg)
        last_zone = int(pf.params["number_of_zones"])
        slope_key = f"zone_{last_zone}_surface_density_exp"
        # Resolve before mass splitting so integration and MCFOST use the same p2.
        cfg[f"zone_{last_zone}_-gamma_exp"] = cfg.get(slope_key, pf.params[slope_key])

    if two_zone_cont_lhc:
        cfg = dict(cfg)  # Keep the sampled configuration separate from derived values.
        for key in ("disk_Rmid", "disk_total_dust_mass"):
            if key not in cfg:
                raise ValueError(f"2ZONE_CONT_LHC requires configuration parameter {key}.")
        derived_keys = {"zone_1_Rout", "zone_2_Rin", "zone_1_dust_mass", "zone_2_dust_mass"}
        if derived_keys.intersection(cfg):
            raise ValueError("2ZONE_CONT_LHC derives these parameters; remove them from the config: "
                             + ", ".join(sorted(derived_keys.intersection(cfg))))
        if int(pf.params["number_of_zones"]) != 2:
            raise ValueError("2ZONE_CONT_LHC requires a template containing exactly two zones.")
        for zone in (1, 2):
            for suffix in ("type", "edge", "surface_density_exp"):
                key = f"zone_{zone}_{suffix}"
                cfg.setdefault(key, pf.params[key])
            allowed_types = (1,) if zone == 1 else (1, 2)
            if float(cfg[f"zone_{zone}_type"]) not in allowed_types or float(cfg[f"zone_{zone}_edge"]) != 0:
                raise ValueError("2ZONE_CONT_LHC requires zone 1 type=1, zone 2 type=1 or 2, and edge=0 in both zones.")
        if float(cfg["zone_2_type"]) == 2:
            for key in ("zone_2_Rc", "zone_2_-gamma_exp"):
                cfg.setdefault(key, pf.params[key])
        for key in ("zone_1_Rin", "zone_2_Rout"):
            cfg.setdefault(key, pf.params[key])
        _2zone_cont_calc_dmass(cfg)

    rref_key = "zone_1_Rref"
    rin_key = "zone_1_Rin"
    rref_tracks_rin = cfg.get(rref_key) == rin_key
    for key in cfg.keys():
        if two_zone_cont_lhc and key in ("disk_Rmid", "disk_total_dust_mass"):
            continue
        if key == rref_key and rref_tracks_rin:
            # Resolve this symbolic configuration value after the full trial
            # configuration has been applied, rather than writing text to .para.
            continue
        if key in pf.params:
            pf.set_param(key, cfg[key])
        elif "_component_2_volume_fraction_" in key:
            # This is translated into MCFOST's component-2 fraction below.
            continue
        elif key=='inclination':
            pf.set_param('imin', cfg[key])
            pf.set_param('imax', cfg[key])
        elif key.startswith('puffed_'):
            #puffed up rim parameters are set during run_mcfost function
            continue
        
        else:
            print(f"Warning: parameter {key} not found in MCFOST parameter file.")

    # The scale-height reference radius may be an independent sampled value,
    # or deliberately tied to the inner rim through the symbolic YAML value
    # ``zone_1_Rin``.  Retain the historical Rin fallback when Rref is absent.
    if rref_tracks_rin or (rref_key not in cfg and rin_key in cfg):
        pf.set_param(rref_key, cfg.get(rin_key, pf.params[rin_key]))
        print(f"Warning: Set {rref_key} to be at inner rim with value {cfg.get(rin_key, pf.params[rin_key])}")
    _apply_selected_second_component_fractions(pf, cfg)
    # Set fidelity-related params
    # pf.set_param("nbr_photons_eq_th", fidelity["nbr_photons_eq_th"])
    # pf.set_param("nbr_photons_lambda", fidelity["nbr_photons_lambda"])
    # pf.set_param("nbr_photons_image", fidelity["nbr_photons_image"])
    
    # Save the modified file
    if share_zone_composition:
        pf.save(param_path, share_zone_composition=True)
        # Record effective dust properties after conditional mixtures resolve.
        cfg = dict(cfg)
        for key, value in pf.params.items():
            if key.startswith("zone_1_species_") or key == "zone_1_number_of_species":
                cfg[key] = value
                cfg[key.replace("zone_1_", "zone_2_", 1)] = value
    else:
        pf.save(param_path)
    with open(outdir / "config_used.json", "w") as f:
        json.dump({"cfg": cfg, "fidelity": fidelity}, f, indent=2)
    
    return param_path


def run_mcfost(fidelity: dict, param_path: Path, workdir: Path, puffed_up_rim: bool=False, cfg: Dict[str, Any]={},*,alma_spec=None) -> None:

    print(f"run mcfost in {workdir}")
    # base
    print(fidelity)

    
    if puffed_up_rim:
        run_mcfost_safe(param_path, workdir, options=["-puffed_up_rim", f"{cfg.get('puffed_h_rim_over_h0', 0)}", f"{cfg.get('puffed_r_rim', 0)}", f"{cfg.get('puffed_delta_r', 0)}"], logfile="mcfost_temp.log")
        run_mcfost_safe(param_path, workdir, options=["-disk_struct","-puffed_up_rim", f"{cfg.get('puffed_h_rim_over_h0', 0)}", f"{cfg.get('puffed_r_rim', 0)}", f"{cfg.get('puffed_delta_r', 0)}"], logfile="mcfost_base_struct.log")
                
    else:
        run_mcfost_safe(param_path, workdir, logfile="mcfost_temp.log")
        run_mcfost_safe(param_path, workdir, options=["-disk_struct"], logfile="mcfost_base_struct.log")
        
    
    if "vis2_1perband" in fidelity["products"]:
        for w in [1.65, 2.20, 3.50, 10.0]:
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
        if alma_spec is None:
            raise ValueError("ALMA products require observation image metadata.")

        # Copy the trial's physical parameters; change only image geometry.
        alma_parameters = ParaFile(str(param_path))
        distance_pc = float(alma_parameters.params["distance_pc"])

        if not np.isfinite(distance_pc) or distance_pc <= 0:
            raise ValueError("Distance must be positive and finite.")

        # Angular size [mas] × distance [pc] / 1000 = physical size [au].
        map_size_au = alma_spec["fov_mas"] * distance_pc / 1000.0

        alma_parameters.set_param("grid_nx", alma_spec["npix"])
        alma_parameters.set_param("grid_ny", alma_spec["npix"])
        alma_parameters.set_param("grid_size", map_size_au)

        alma_param_path = Path(workdir) / "model_alma.para"
        alma_parameters.save(alma_param_path)

        wavelength_text = f'{alma_spec["wavelength_um"]:.3f}'

        run_mcfost_safe(
            alma_param_path.resolve(),
            workdir,
            options=["-img", wavelength_text, "-casa"],
            logfile="mcfost_alma.log",
        )




    #initiall version of the code to run MCFOST for 870 micron image for ALMA data.
    # for w in [870.0]:
    #     if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
    #         print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
    #         continue
    #     run_mcfost_safe(param_path, workdir, options=["-img", f"{w}", "-casa"], logfile=f"mcfost_{w:.2f}.log")

    if "vis2_chromatic" in fidelity["products"]: # all wavelengths for chromatic visibilities (PIONIER, MATISSE, GRAVITY) + full PDI
        for w in [0.55, 0.82, 1.5,1.55,1.6,1.63,1.65,1.7,1.75,1.8,1.85,1.9,
                  1.95,2.0,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,
                  2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,
                  7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0]:
            if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
                print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
                continue
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")




def load_and_score_outputs(fidelity: Dict[str, Any], workdir: Path, data_arg:Dict[str, Any], args,cfg) -> Tuple[float, Dict[str, Any]]:
    """
    Read MCFOST outputs and compute a single scalar loss.
    

    Return the likelihood (lower is better) and additional_info dictionary.
    Additional info contains extra information, such as individual chi2 values for each observable.
    """
    print(f'[obriy_mcfost] fidelity["stage"] = {fidelity["stage"]}')
    print(f'[obriy_mcfost] fidelity["products"] = {fidelity["products"]}')  
    additional_info = {}

    if args.plot_intermediate:
        with obg.diagnostic_plot("Figure directory creation"):
            if not os.path.exists(str(workdir)+'/figures/'):
                os.makedirs(str(workdir)+'/figures/')

    if "sed" in fidelity["products"]:
       data_sed = data_arg["sed"]

    if ("vis2_1perband" in fidelity["products"]) or  ("vis2_chromatic" in fidelity["products"]):
        # Select observations for this trial, not the union of all fidelity stages.
        mode = "vis2_chromatic" if "vis2_chromatic" in fidelity["products"] else "vis2_1perband"
        selected_data = data_arg["interferometry"][mode]
        container_data_pionier = selected_data["pionier"]
        container_data_gravity = selected_data["gravity"]
        container_data_matisse_l = selected_data["matisse_l"]
        container_data_matisse_n = selected_data["matisse_n"]

    if "pdi_V" in fidelity["products"]:
        pdi_data_v = data_arg["pdi_V"] #each disc with data not deconvolved q_phi, u_phi, pi, and psf
    if "pdi_I" in fidelity["products"]:
        pdi_data_i = data_arg["pdi_I"]
    if "pdi_H" in fidelity["products"]:
        pdi_data_h = data_arg["pdi_H"]
    if "alma" in fidelity["products"]:
        data_alma = data_arg["alma"]
    
    print('[obriy_mcfost] Data for scoring loaded successfully')

    simulation_name = workdir.name


    sed_path = workdir / "data_th" / "sed_rt.fits.gz"
    if not sed_path.exists():
        reason = f"Required MCFOST SED output not found: {sed_path}"
        print(f"[obriy_mcfost] {reason}")
        additional_info["failure"] = {
            "code": "missing_sed_output",
            "stage": "load_and_score_outputs",
            "reason": reason,
            "path": str(sed_path),
            "trial_dir": str(workdir),
        }
        return 1e99, additional_info
    # These plots are diagnostics, not objective inputs. A missing or incompatible
    # structure output must not discard an otherwise scoreable trial.
    if args.plot_intermediate:
        for plot_function, plot_args, plot_kwargs in (
            (plot_mcfost_disk_structure, (str(workdir.parent) + '/', simulation_name),
             {"az_disk": 0}),
            (plot_mcfost_density_temperature_cuts, (workdir,),
             {"reference_radius_au": cfg.get("zone_1_Rc")}),
        ):
            existing_figures = set(plt.get_fignums())
            try:
                plot_function(*plot_args, **plot_kwargs)
            except Exception as error:
                message = f"{plot_function.__name__}: {type(error).__name__}: {error}"
                print(f"[obriy_mcfost] Diagnostic plot failed; continuing scoring: {message}")
                additional_info.setdefault("diagnostic_plot_errors", []).append(message)
            finally:
                for figure_number in set(plt.get_fignums()) - existing_figures:
                    plt.close(figure_number)

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
            full_wavelengths=[1.65, 2.2, 3.5, 10.0]
            wave_for_background=args.overresolved_flux_fit_for_interferometry
            closest = min(full_wavelengths, key=lambda x: abs(x - wave_for_background))
            model_sed=distroi.read_sed_mcfost(str(sed_path))

            container_data=container_data_pionier if closest==1.65 else container_data_gravity if closest ==2.2 else container_data_matisse_l if closest==3.5 else container_data_matisse_n if closest==10.0 else None
            img_dir = f"data_{closest}/"
            _, _, _, _, frac_closest_wavelength_optimised= obi.monochromatic_chi_with_background(str(workdir), img_dir=img_dir, container_data=container_data, img_sed=model_sed, wave_for_background=wave_for_background, vistype='vis' if container_data.vis_in_fcorr else 'vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title=f"Reference for overresolved flux, wavelength {closest}", log_plotv=False, ebminv=ebminv_sed)
                

            chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier, _= obi.monochromatic_chi_with_background(str(workdir), img_dir="data_1.65/", container_data=container_data_pionier,  img_sed=model_sed, wave_for_background=args.overresolved_flux_fit_for_interferometry,frac_for_background=frac_closest_wavelength_optimised, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER 1.65", log_plotv=False, ebminv=ebminv_sed)
            chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity, _= obi.monochromatic_chi_with_background(str(workdir), img_dir="data_2.2/", container_data=container_data_gravity,  img_sed=model_sed, wave_for_background=args.overresolved_flux_fit_for_interferometry, frac_for_background=frac_closest_wavelength_optimised, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY 2.2", log_plotv=False, ebminv=ebminv_sed)
            chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l, _= obi.monochromatic_chi_with_background(str(workdir), img_dir="data_3.5/", container_data=container_data_matisse_l,  img_sed=model_sed, wave_for_background=args.overresolved_flux_fit_for_interferometry,frac_for_background=frac_closest_wavelength_optimised,  vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L 3.5", log_plotv=True, ebminv=ebminv_sed)
            chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n, _= obi.monochromatic_chi_with_background(str(workdir), img_dir="data_10.0/", container_data=container_data_matisse_n,  img_sed=model_sed, wave_for_background=args.overresolved_flux_fit_for_interferometry, frac_for_background=frac_closest_wavelength_optimised, vistype='vis', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N 10.0", log_plotv=False, ebminv=ebminv_sed)

        else:
            chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.monochromatic_chi(str(workdir), img_dir="data_1.65/", container_data=container_data_pionier, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER 1.65", log_plotv=False, ebminv=ebminv_sed)
            chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.monochromatic_chi(str(workdir), img_dir="data_2.2/", container_data=container_data_gravity, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY 2.2", log_plotv=False, ebminv=ebminv_sed)
            chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.monochromatic_chi(str(workdir), img_dir="data_3.5/", container_data=container_data_matisse_l,vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L 3.5", log_plotv=True, ebminv=ebminv_sed)
            chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.monochromatic_chi(str(workdir), img_dir="data_10.0/", container_data=container_data_matisse_n, vistype='vis', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N 10.0", log_plotv=False, ebminv=ebminv_sed)
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

        model_sed=distroi.read_sed_mcfost(str(sed_path))
                                          
        if args.overresolved_flux_fit_for_interferometry:
            
            wave_for_background=args.overresolved_flux_fit_for_interferometry
            closest = min(full_wavelengths, key=lambda x: abs(x - wave_for_background))
            container_data=container_data_pionier if closest in pionier_wavelengths else container_data_gravity if closest in gravity_wavelengths else container_data_matisse_l if closest in matisse_l_wavelengths else container_data_matisse_n
            img_dir = f"data_{closest}/"
            _, _, _, _, frac_closest_wavelength_optimised= obi.monochromatic_chi_with_background(str(workdir), img_dir=img_dir, container_data=container_data, img_sed=model_sed, wave_for_background=wave_for_background, vistype='vis' if container_data.vis_in_fcorr else 'vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title=f"Reference for overresolved flux, wavelength {closest}", log_plotv=False, ebminv=ebminv_sed)
                        
        else:
            wave_for_background=None
            frac_closest_wavelength_optimised=None
                

        pionier_img_dirs=[f"data_{w}/" for w in pionier_wavelengths]
        chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.chromatic_chi(str(workdir), img_dir=pionier_img_dirs, container_data=container_data_pionier, vistype='vis2', img_sed=model_sed, wave_for_background=wave_for_background,frac_for_background=frac_closest_wavelength_optimised, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER", log_plotv=False, ebminv=ebminv_sed)
        gravity_img_dirs=[f"data_{w}/" for w in gravity_wavelengths]
        chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.chromatic_chi(str(workdir), img_dir=gravity_img_dirs, container_data=container_data_gravity, vistype='vis2', img_sed=model_sed, wave_for_background=wave_for_background,frac_for_background=frac_closest_wavelength_optimised, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY", log_plotv=False, ebminv=ebminv_sed)
        matisse_l_img_dirs=[f"data_{w}/" for w in matisse_l_wavelengths]
        chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.chromatic_chi(str(workdir), img_dir=matisse_l_img_dirs, container_data=container_data_matisse_l,vistype='vis2', img_sed=model_sed, wave_for_background=wave_for_background,frac_for_background=frac_closest_wavelength_optimised, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L", log_plotv=True, ebminv=ebminv_sed)
        matisse_n_img_dirs=[f"data_{w}/" for w in matisse_n_wavelengths]
        chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.chromatic_chi(str(workdir), img_dir=matisse_n_img_dirs, container_data=container_data_matisse_n, vistype='vis', img_sed=model_sed, wave_for_background=wave_for_background,frac_for_background=frac_closest_wavelength_optimised, plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N", log_plotv=False, ebminv=ebminv_sed)
        
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

        correct_unresolved = bool(args.correct_unresolved_polarimetry)
        correction_radius_px = obp.unresolved_correction_radius(
            correct_unresolved, getattr(args, 'unresolved_correction_radius_px', None))
        model_polarimetry_key = (
            'mcfost_convolved_unresolved_corrected' if correct_unresolved else 'mcfost_convolved')

        print('[obriy_mcfost] Polarimetric analysis started')
        if "pdi_I" in fidelity["products"]:
    
            results_i=obp.polarimetric_analysis(str(workdir), 0.82, unresolved_correction_radius_px=correction_radius_px, camera='zimpol',convolution_mode='file', psf_array=pdi_data_i['psf'], psf_cut=100,
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Iband')
            #CHANGES HERE
            disk_pa_deg=cfg.get('disk_pa', 0)
            #to avoid machine precision issues due to the small values of Q and U, we can multiply them by representative factor of qphi, because we are interested in normalised values anyway and it is not affecting final differential quadrant results. 
            if args.plot_intermediate:
                with obg.diagnostic_plot("I-band diagnostic quadrant preparation and comparison"):
                    max_val_for_order = np.nanmax(np.abs(results_i['mcfost_convolved']['q_phi']))
                    order = np.floor(np.log10(max_val_for_order)) if max_val_for_order > 0 else 0
                    size_convolved_i = results_i['mcfost_convolved']['img_q'].shape[0]
                    r_out_quadrants=(size_convolved_i-1)*results_i['mcfost_convolved']['pixel_scale_mas']/2 #in mas, to avoid issues with the quadrant calculation, we can use the full size of the convolved image
                    quadrant_results_sim_i = obp.differential_quadrants(results_i['mcfost_convolved']['img_q']*10**(-order), results_i['mcfost_convolved']['img_u']*10**(-order), pixel_scale_mas=results_i['mcfost_convolved']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_i_simulation.png', roi_mas=None)
                    quadrant_results_sim_i_conv_unres_corr = obp.differential_quadrants(results_i['mcfost_convolved_unresolved_corrected']['img_q']*10**(-order), results_i['mcfost_convolved_unresolved_corrected']['img_u']*10**(-order), pixel_scale_mas=results_i['mcfost_convolved_unresolved_corrected']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_i_simulation_unresolved_corr.png', roi_mas=None)
                    #we can use the same order as for the model to avoid any issues with the differential quadrant calculation
                    quadrant_results_data_i = obp.differential_quadrants(pdi_data_i['pol_images']['Q']*10**(-order), pdi_data_i['pol_images']['U']*10**(-order), pixel_scale_mas=results_i['mcfost_convolved']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_i_data.png', roi_mas=None)
                    fig, axes=obp.plot_quadrant_comparison([quadrant_results_data_i, quadrant_results_sim_i, quadrant_results_sim_i_conv_unres_corr], ['Data', 'Model', 'Model Conv Unres Corr'], save=str(workdir)+'/figures/'+'quadrants_i_comparison.png')
                    plt.close(fig)
            
            print(f'[obriy_mcfost] I band comparison uses {model_polarimetry_key}')
            data_cropped_i, model_cropped_i = obp.crop_to_same_size(
                pdi_data_i['pol_images']['Q_phi'], results_i[model_polarimetry_key]['q_phi'])
            if args.plot_intermediate:
                with obg.diagnostic_plot("I-band profile preparation"):
                    model_rad_prof = results_i[model_polarimetry_key]['radial_profiles']['q_phi']
                    model_azimuthal_prof = results_i[model_polarimetry_key]['azimuthal_profiles']['q_phi']
                   
                    # Calculate metrics for arcsinh-scaled images to highlight morphology
                    obs_rad_prof_pi, obs_az_prof_pi = pdi_data_i['radial_profiles']['Q_phi'], pdi_data_i['azimuthal_profiles']['Q_phi']
            
            # Disabled legacy profile scores: neither scored nor plotted; small profiles can fail.
            # profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof_pi, model_rad_prof, 3.6, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_i_")
            # profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof, 3.6, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_i_")
            # profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            # profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike
            if args.plot_intermediate:
                with obg.diagnostic_plot("I-band radial and azimuthal profile comparison"):
                    obp.profile_chi2(obs_rad_prof_pi, model_rad_prof, 3.6, profile_type="radial", plot=True, calculate_chi2=False, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_i_")
                    obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof, 3.6, profile_type="azimuthal", plot=True, calculate_chi2=False, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_i_")


            # SSIM is needed only for the optional diagnostic image.
            metrics_i = {}
            if args.plot_intermediate:
                with obg.diagnostic_plot("I-band SSIM calculation"):
                    metrics_i = obp.full_image_metrics_noshift(
                        data_cropped_i, model_cropped_i,
                        normalize="zscore",          # good default for morphology
                        ssim_win=None,                 # 7–15 is typical
                        # return_pixel_chi2=True  # Unused by scoring and plots.
                        calculate_ncc=False,  # NCC is neither scored nor plotted.
                        return_pixel_chi2=False
                    )
            if args.plot_intermediate:
                with obg.diagnostic_plot("I-band deconvolved Qphi and SSIM images"):
                    obp.plot_polarimetric_image(results_i['mcfost_convolved_unresolved_corrected']['q_phi_deconvolved'], 3.6, title=f'Model Qphi, conv, unres corr, decon', save=str(workdir)+'/figures'+'/model_q_phi_corr_conv_deconv_I.png', image_scale='asinh', roi_half_size=100)
                    obp.plot_polarimetric_image(results_i['mcfost_convolved']['q_phi_deconvolved'], 3.6, title=f'Model Qphi, conv, decon', save=str(workdir)+'/figures'+'/model_q_phi_conv_deconv_I.png', image_scale='asinh', roi_half_size=100)

                    obp.plot_polarimetric_image(metrics_i["ssim_image"], 3.6, title=f'ssim, score {metrics_i["ssim"]}', save=str(workdir)+'/figures'+'/ssim_image_I.png', image_scale='linear', roi_half_size=50)

            # Disabled legacy metric logging: these diagnostics are neither scored nor plotted.
            # obp.save_band_metrics(
            # workdir,
            # band="I",
            # analysis_metrics=results_i[model_polarimetry_key]['metrics'],
            # ssim_score=metrics_i.get("ssim"),
            # ncc_score=metrics_i.get("ncc"),
            # extras={"ps_mas": 3.6, "notes": "zscore"}
            # )
            
            if args.plot_intermediate:
                with obg.diagnostic_plot("I-band data/model images"):
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
            # print(f'[obriy_mcfost] I band metrics: SSIM={metrics_i["ssim"]}, NCC={metrics_i["ncc"]}, profile_pi_chi2_red={profile_pi_chi2_red}')
            
            

            constraint_comparison_i = obp.compare_pdi_constraints(
                pdi_data_i['pol_images'], results_i[model_polarimetry_key],
                results_i[model_polarimetry_key]['pixel_scale_mas'],
                radial_bin_mas=getattr(args, 'pdi_radial_bin_mas', 25.0),
                disk_pa_deg=disk_pa_deg,
                output_path=workdir / 'figures' / 'pdi_constraints_i.png', band='I')
            loss_i = constraint_comparison_i['loss']
            #Loss based only on profile chi2 to test if it can drive the fit
            #loss_i=profile_pi_chi2_red

            additional_info.setdefault("pdi", {})['I'] = {
                "ssim": metrics_i.get("ssim"),
                # "ncc": metrics_i.get("ncc"),  # Unused diagnostic disabled.
                # "profile_pi_chi2_red": profile_pi_chi2_red,
                # "profile_pi_loglike": profile_loglike,
                # "profile_rad_pi_chi2": profile_rad_pi_chi2,
                # "profile_rad_pi_npoints": profile_rad_pi_npoints,
                # "profile_az_pi_chi2": profile_az_pi_chi2,
                # "profile_az_pi_npoints": profile_az_pi_npoints,
                "loss": loss_i,
                "constraints": constraint_comparison_i,
                "model_unresolved_corrected": correct_unresolved,
                "unresolved_correction_radius_px": correction_radius_px if correct_unresolved else None,
            }


        if "pdi_V" in fidelity["products"]:
    
            results_v=obp.polarimetric_analysis(str(workdir), 0.55, unresolved_correction_radius_px=correction_radius_px, camera='zimpol',convolution_mode='file', psf_array=pdi_data_v['psf'],psf_cut=100,
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
             
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Vband')

            #CHANGES HERE
            disk_pa_deg=cfg.get('disk_pa', 0)
            #to avoid machine precision issues due to the small values of Q and U, we can multiply them by representative factor of qphi, because we are interested in normalised values anyway and it is not affecting final differential quadrant results. 
            if args.plot_intermediate:
                with obg.diagnostic_plot("V-band diagnostic quadrant preparation and comparison"):
                    max_val_for_order = np.nanmax(np.abs(results_v['mcfost_convolved']['q_phi']))
                    order = np.floor(np.log10(max_val_for_order)) if max_val_for_order > 0 else 0
                    size_convolved_v = results_v['mcfost_convolved']['img_q'].shape[0]
                    r_out_quadrants=(size_convolved_v-1)*results_v['mcfost_convolved']['pixel_scale_mas']/2 #in mas, to avoid issues with the quadrant calculation, we can use the full size of the convolved image
                    quadrant_results_sim_v = obp.differential_quadrants(results_v['mcfost_convolved']['img_q']*10**(-order), results_v['mcfost_convolved']['img_u']*10**(-order), pixel_scale_mas=results_v['mcfost_convolved']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_v_simulation.png', roi_mas=None)
                    quadrant_results_sim_v_conv_unres_corr = obp.differential_quadrants(results_v['mcfost_convolved_unresolved_corrected']['img_q']*10**(-order), results_v['mcfost_convolved_unresolved_corrected']['img_u']*10**(-order), pixel_scale_mas=results_v['mcfost_convolved_unresolved_corrected']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_v_simulation_unresolved_corr.png', roi_mas=None)
                    #we can use the same order as for the model to avoid any issues with the differential quadrant calculation
                    quadrant_results_data_v = obp.differential_quadrants(pdi_data_v['pol_images']['Q']*10**(-order), pdi_data_v['pol_images']['U']*10**(-order), pixel_scale_mas=results_v['mcfost_convolved']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_v_data.png', roi_mas=None)
                    fig, axes=obp.plot_quadrant_comparison([quadrant_results_data_v, quadrant_results_sim_v, quadrant_results_sim_v_conv_unres_corr], ['Data', 'Model', 'Model Conv Unres Corr'], save=str(workdir)+'/figures/'+'quadrants_v_comparison.png')
                    plt.close(fig)
            if args.plot_intermediate:
                with obg.diagnostic_plot("V-band deconvolved Qphi images"):
                    obp.plot_polarimetric_image(results_v['mcfost_convolved_unresolved_corrected']['q_phi_deconvolved'], 3.6, title=f'Model Qphi, conv, unres corr, decon', save=str(workdir)+'/figures'+'/model_q_phi_corr_conv_deconv_V.png', image_scale='asinh', roi_half_size=100)
                    obp.plot_polarimetric_image(results_v['mcfost_convolved']['q_phi_deconvolved'], 3.6, title=f'Model Qphi, conv, decon', save=str(workdir)+'/figures'+'/model_q_phi_conv_deconv_V.png', image_scale='asinh', roi_half_size=100)
            
            print(f'[obriy_mcfost] V band comparison uses {model_polarimetry_key}')
            data_cropped_v, model_cropped_v = obp.crop_to_same_size(
                pdi_data_v['pol_images']['Q_phi'], results_v[model_polarimetry_key]['q_phi'])
            if args.plot_intermediate:
                with obg.diagnostic_plot("V-band profile preparation"):
                    model_rad_prof = results_v[model_polarimetry_key]['radial_profiles']['q_phi']
                    model_azimuthal_prof = results_v[model_polarimetry_key]['azimuthal_profiles']['q_phi']
                    #CHANGE HERE for profiles that are already calculated in loading data initially to avoid recalculating them and speed up the process
                    obs_rad_prof, obs_az_prof= pdi_data_v['radial_profiles']['Q_phi'], pdi_data_v['azimuthal_profiles']['Q_phi']
            
            # Disabled legacy profile scores: neither scored nor plotted; small profiles can fail.
            # profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof, model_rad_prof, 3.6, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_v_")
            # profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof, model_azimuthal_prof, 3.6, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_v_")
            # profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            # profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike
            if args.plot_intermediate:
                with obg.diagnostic_plot("V-band radial and azimuthal profile comparison"):
                    obp.profile_chi2(obs_rad_prof, model_rad_prof, 3.6, profile_type="radial", plot=True, calculate_chi2=False, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_v_")
                    obp.profile_chi2(obs_az_prof, model_azimuthal_prof, 3.6, profile_type="azimuthal", plot=True, calculate_chi2=False, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_v_")

            # SSIM is needed only for the optional diagnostic image.
            metrics_v = {}
            if args.plot_intermediate:
                with obg.diagnostic_plot("V-band SSIM calculation"):
                    metrics_v = obp.full_image_metrics_noshift(
                        data_cropped_v, model_cropped_v,
                        normalize="zscore",          # good default for morphology
                        ssim_win=None,                 # 7–15 is typical
                        # return_pixel_chi2=True  # Unused by scoring and plots.
                        calculate_ncc=False,  # NCC is neither scored nor plotted.
                        return_pixel_chi2=False
                    )
            if args.plot_intermediate:
                with obg.diagnostic_plot("V-band SSIM image"):
                    obp.plot_polarimetric_image(metrics_v["ssim_image"], 3.6, title=f'ssim, score {metrics_v["ssim"]}', save=str(workdir)+'/figures'+'/ssim_image_V.png', image_scale='linear', roi_half_size=50)

            # Disabled legacy metric logging: these diagnostics are neither scored nor plotted.
            # obp.save_band_metrics(
            # workdir,
            # band="V",
            # analysis_metrics=results_v[model_polarimetry_key]['metrics'],
            # ssim_score=metrics_v.get("ssim"),
            # ncc_score=metrics_v.get("ncc"),
            # extras={"ps_mas": 3.6, "notes": "zscore"}
            # )
            if args.plot_intermediate:
                with obg.diagnostic_plot("V-band data/model images"):
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
            # print(f'[obriy_mcfost] V band metrics: SSIM={metrics_v["ssim"]}, NCC={metrics_v["ncc"]}, profile_pi_chi2_red={profile_pi_chi2_red}')
            constraint_comparison_v = obp.compare_pdi_constraints(
                pdi_data_v['pol_images'], results_v[model_polarimetry_key],
                results_v[model_polarimetry_key]['pixel_scale_mas'],
                radial_bin_mas=getattr(args, 'pdi_radial_bin_mas', 25.0),
                disk_pa_deg=disk_pa_deg,
                output_path=workdir / 'figures' / 'pdi_constraints_v.png', band='V')
            loss_v = constraint_comparison_v['loss']
            #Loss based only on profile chi2 to test if it can drive the fit
            #loss_v= profile_pi_chi2_red
            additional_info.setdefault("pdi", {})['V'] = {
                            "ssim": metrics_v.get("ssim"),
                            # "ncc": metrics_v.get("ncc"),  # Unused diagnostic disabled.
                            # "profile_pi_chi2_red": profile_pi_chi2_red,
                            # "profile_pi_loglike": profile_loglike,
                            # "profile_rad_pi_chi2": profile_rad_pi_chi2,
                            # "profile_rad_pi_npoints": profile_rad_pi_npoints,
                            # "profile_az_pi_chi2": profile_az_pi_chi2,
                            # "profile_az_pi_npoints": profile_az_pi_npoints,
                            "loss": loss_v,
                "constraints": constraint_comparison_v,
                "model_unresolved_corrected": correct_unresolved,
                "unresolved_correction_radius_px": correction_radius_px if correct_unresolved else None,
                        }

        if "pdi_H" in fidelity["products"]:
            results_h=obp.polarimetric_analysis(str(workdir), 1.63, unresolved_correction_radius_px=correction_radius_px, camera='irdis',convolution_mode='file', psf_array=pdi_data_h['psf'],psf_cut=100,
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Hband')
            #CHANGES HERE
            disk_pa_deg=cfg.get('disk_pa', 0)
            #to avoid machine precision issues due to the small values of Q and U, we can multiply them by representative factor of qphi, because we are interested in normalised values anyway and it is not affecting final differential quadrant results. 
            if args.plot_intermediate:
                with obg.diagnostic_plot("H-band diagnostic quadrant preparation and comparison"):
                    max_val_for_order = np.nanmax(np.abs(results_h['mcfost_convolved']['q_phi']))
                    order = np.floor(np.log10(max_val_for_order)) if max_val_for_order > 0 else 0
                    size_convolved_h = results_h['mcfost_convolved']['img_q'].shape[0]
                    r_out_quadrants=(size_convolved_h-1)*12.27/2 #in mas, to avoid issues with the quadrant calculation, we can use the full size of the convolved image
                    quadrant_results_sim_h = obp.differential_quadrants(results_h['mcfost_convolved']['img_q']*10**(-order), results_h['mcfost_convolved']['img_u']*10**(-order), pixel_scale_mas=results_h['mcfost_convolved']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_h_simulation.png', roi_mas=None)
                    quadrant_results_sim_h_conv_unres_corr = obp.differential_quadrants(results_h['mcfost_convolved_unresolved_corrected']['img_q']*10**(-order), results_h['mcfost_convolved_unresolved_corrected']['img_u']*10**(-order), pixel_scale_mas=results_h['mcfost_convolved_unresolved_corrected']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_h_simulation_unresolved_corr.png', roi_mas=None)
                    #we can use the same order as for the model to avoid any issues with the differential quadrant calculation
                    quadrant_results_data_h = obp.differential_quadrants(pdi_data_h['pol_images']['Q']*10**(-order), pdi_data_h['pol_images']['U']*10**(-order), pixel_scale_mas=results_h['mcfost_convolved']['pixel_scale_mas'], disk_pa_deg=disk_pa_deg, r_in_mas=0, r_out_mas=r_out_quadrants, flip_disk_y=False, plot=args.plot_intermediate, save=str(workdir)+'/figures/'+'quadrants_h_data.png', roi_mas=None)
                    fig, axes=obp.plot_quadrant_comparison([quadrant_results_data_h, quadrant_results_sim_h, quadrant_results_sim_h_conv_unres_corr], ['Data', 'Model', 'Model Conv Unres Corr'],  save=str(workdir)+'/figures/'+'quadrants_h_comparison.png')
                    plt.close(fig)
                        
            print(f'[obriy_mcfost] H band comparison uses {model_polarimetry_key}')
            data_cropped_h, model_cropped_h = obp.crop_to_same_size(
                pdi_data_h['pol_images']['Q_phi'], results_h[model_polarimetry_key]['q_phi'])
            if args.plot_intermediate:
                with obg.diagnostic_plot("H-band profile preparation"):
                    model_rad_prof = results_h[model_polarimetry_key]['radial_profiles']['q_phi']
                    model_azimuthal_prof = results_h[model_polarimetry_key]['azimuthal_profiles']['q_phi']
            
                    obs_rad_prof_pi, obs_az_prof_pi = pdi_data_h['radial_profiles']['Q_phi'], pdi_data_h['azimuthal_profiles']['Q_phi']
            # Disabled legacy profile scores: neither scored nor plotted; small profiles can fail.
            # profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof_pi, model_rad_prof, 12.27, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_h_")
            # profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof, 12.27, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_h_")
            # profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            # profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike
            if args.plot_intermediate:
                with obg.diagnostic_plot("H-band radial and azimuthal profile comparison"):
                    obp.profile_chi2(obs_rad_prof_pi, model_rad_prof, 12.27, profile_type="radial", plot=True, calculate_chi2=False, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_h_")
                    obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof, 12.27, profile_type="azimuthal", plot=True, calculate_chi2=False, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_h_")


            # SSIM is needed only for the optional diagnostic image.
            metrics_h = {}
            if args.plot_intermediate:
                with obg.diagnostic_plot("H-band SSIM calculation"):
                    metrics_h = obp.full_image_metrics_noshift(
                        data_cropped_h, model_cropped_h,
                        normalize="zscore",          # good default for morphology
                        ssim_win=None,                 # 7–15 is typical
                        # return_pixel_chi2=True  # Unused by scoring and plots.
                        calculate_ncc=False,  # NCC is neither scored nor plotted.
                        return_pixel_chi2=False
                    )
            if args.plot_intermediate:
                with obg.diagnostic_plot("H-band SSIM image"):
                    obp.plot_polarimetric_image(metrics_h["ssim_image"], 12.27, title=f'ssim, score {metrics_h["ssim"]}', save=str(workdir)+'/figures'+'/ssim_image_H.png', image_scale='linear', roi_half_size=30)
            # Disabled legacy metric logging: these diagnostics are neither scored nor plotted.
            # obp.save_band_metrics(
            # workdir,
            # band="H",
            # analysis_metrics=results_h[model_polarimetry_key]['metrics'],
            # ssim_score=metrics_h.get("ssim"),
            # ncc_score=metrics_h.get("ncc"),
            # extras={"ps_mas": 12.27, "notes": "zscore"}
            # )
        # except Exception as e:
        #     print(f"Error in H-band polarimetric analysis: {e}")
        #     data_cropped_h= np.zeros((10,10))
        #     model_cropped_h= np.zeros((10,10))
        #     metrics_h={'ssim':-1.0,'ncc':-1.0} #        
            if args.plot_intermediate:
                with obg.diagnostic_plot("H-band data/model images"):
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
            # print(f'[obriy_mcfost] H band metrics: SSIM={metrics_h["ssim"]}, NCC={metrics_h["ncc"]}, profile_pi_chi2_red={profile_pi_chi2_red}')
            constraint_comparison_h = obp.compare_pdi_constraints(
                pdi_data_h['pol_images'], results_h[model_polarimetry_key],
                results_h[model_polarimetry_key]['pixel_scale_mas'],
                radial_bin_mas=getattr(args, 'pdi_radial_bin_mas', 25.0),
                disk_pa_deg=disk_pa_deg,
                output_path=workdir / 'figures' / 'pdi_constraints_h.png', band='H')
            loss_h = constraint_comparison_h['loss']
            #Loss based only on profile chi2 to test if it can drive the fit
            #loss_h=profile_pi_chi2_red # weights can be adjusted
            additional_info.setdefault("pdi", {})['H'] = {
                            "ssim": metrics_h.get("ssim"),
                            # "ncc": metrics_h.get("ncc"),  # Unused diagnostic disabled.
                            # "profile_pi_chi2_red": profile_pi_chi2_red,
                            # "profile_pi_loglike": profile_loglike,
                            # "profile_rad_pi_chi2": profile_rad_pi_chi2,
                            # "profile_rad_pi_npoints": profile_rad_pi_npoints,
                            # "profile_az_pi_chi2": profile_az_pi_chi2,
                            # "profile_az_pi_npoints": profile_az_pi_npoints,
                            "loss": loss_h,
                "constraints": constraint_comparison_h,
                "model_unresolved_corrected": correct_unresolved,
                "unresolved_correction_radius_px": correction_radius_px if correct_unresolved else None,
                        }




        
         
        print(f"PDI losses: I-band: {loss_i}, V-band: {loss_v}, H-band: {loss_h}")
    


    if "alma" in fidelity["products"]:
        
        alma_cont = data_alma['alma_cont']
        # Unused by the active ALMA comparison; avoid requiring legacy diagnostic data.
        # obs_rad_prof = data_alma['radial_profile']
        # Unused by the active ALMA comparison; avoid requiring legacy diagnostic data.
        # obs_az_prof = data_alma['azimuthal_profile']
        ps_alma = data_alma['ps_alma']
        # Unused by the active ALMA comparison; avoid requiring legacy diagnostic data.
        # data_size_alma = data_alma['image_size']
        wave=data_alma['alma_wavelength']
        # mask_alma = data_alma['mask_alma']  # Former per-pixel 3-sigma scoring mask.
        alma_spec = data_alma['image_spec']
        wavelength_text = f'{alma_spec["wavelength_um"]:.3f}'

        _, model_header, simulated_itot, _ = oba.load_mcfost_image_alma_casa(str(workdir), wavelength_text)

        model_jybeam_native, model_pixel_mas = oba.convolve_alma_to_jybeam(
            simulated_itot,
            model_header,
            data_alma["header"],
        )

        simulated_itot_as_data, alma_alignment = oba.align_alma_to_observation(
            model_jybeam_native, model_header, data_alma["header"], alma_cont.shape,
        )

        if args.plot_intermediate:
            with obg.diagnostic_plot("ALMA matched model image"):
                obp.plot_polarimetric_image(
                    simulated_itot_as_data,
                    ps_alma,
                    title="ALMA model [Jy/beam]",
                    save=str(workdir / "figures" / "model_itot_alma.png"),
                    image_scale="asinh",
                    roi_half_size=100)



        observed_header = data_alma["header"]

        # Explicit initial assumption: source centre is the FITS reference pixel.
        # FITS coordinates are one-based; NumPy coordinates are zero-based.
        center_xy = tuple(alma_alignment["observed_center_xy"])

        alma_comparison = oba.compare_alma_images(
            alma_cont,
            simulated_itot_as_data,
            observed_header,
            ps_alma,
            center_xy=center_xy,
            aperture_radius_mas=data_alma["aperture_radius_mas"],
            snr_threshold=3.0,
            aperture_padding_pixels=3.0,
            noise_rms_jybeam=data_alma["noise_level_alma"],
            save_path=workdir / "figures" / "alma_comparison.png",
        )

        print(
            "[obriy_mcfost] ALMA aperture flux: "
            f"data={alma_comparison['observed_flux_jy']:.6g} Jy, "
            f"model={alma_comparison['model_flux_jy']:.6g} Jy, "
            f"model−data={alma_comparison['flux_difference_jy']:.6g} Jy"
        )


        # Retained for reference: old per-pixel SNR mask and squared-residual plot.
        # #previous verison
        # residuals_map=(simulated_itot_as_data-alma_cont)**2
        # residuals_map_masked = np.where(mask_alma, residuals_map, np.nan)
        # #residuals_reduced=np.nansum(residuals_map_masked)/np.sum(mask_alma) #does not have error estimate, so not a proper chi2, but takes into account only 3snr points
        #      
        # 
        # #updated version, is normalized by the total flux of the data, so it is more comparable between different models and different datasets
        # residuals_reduced = (
        #     np.nansum(residuals_map_masked)/np.nansum(alma_cont[mask_alma]**2)
        # )*100.0  # percentage of the total flux squared, now will match by order with SED and interferometry chi2, so can be used in the total loss function
        # 
        # print(f"[obriy_mcfost] ALMA residuals image snr>=2 = {residuals_reduced}, sum of mask = {np.sum(mask_alma)}")
        # 
        #     
        # if args.plot_intermediate:
        #     #do some plotting
        #     fig, ax = plt.subplots(1, 3, figsize=(16,6))
        #     fig.subplots_adjust(wspace=0.5)
        # 
        #     color_map = 'viridis' #'afmhot'
        #     im0=ax[0].imshow(alma_cont, color_map, extent=[+alma_cont.shape[0]/2, -alma_cont.shape[0]/2, -alma_cont.shape[1]/2, alma_cont.shape[1]/2])
        #     ax[0].set_title("Data I$_{tot}$")
        #     obg.add_colorbar(fig, ax[0], im0)
        #     im1=ax[1].imshow(simulated_itot_as_data, color_map, extent=[+simulated_itot_as_data.shape[0]/2, -simulated_itot_as_data.shape[0]/2, -simulated_itot_as_data.shape[1]/2, simulated_itot_as_data.shape[1]/2])
        #     ax[1].set_title('Simulated I$_{tot}$')
        #     obg.add_colorbar(fig, ax[1], im1)
        #     im2=ax[2].imshow(residuals_map_masked, color_map,extent=[+alma_cont.shape[0]/2, -alma_cont.shape[0]/2, -alma_cont.shape[1]/2, alma_cont.shape[1]/2])
        #     ax[2].set_title("Residual I$_{tot}$")
        #     obg.add_colorbar(fig, ax[2], im2)
        #     plt.suptitle("ALMA, "+str(wave)+"$\mu m$, reduces chi2 "+ residuals_reduced.astype(str)) #does not have error estimate, so not a proper chi2
        #     fig.savefig(str(workdir)+'_alma_sim_vs_data_'+str(wave)+'.png', dpi= 150, bbox_inches='tight')
        #     plt.close(fig)
        # 
        # 
        # 

        # All finite observation pixels in the circular aperture contribute,
        # including pixels below 3 sigma. The factor of 100 is applied once.
        
        print(f"[obriy_mcfost] ALMA aperture residual-energy loss = {alma_comparison['residual_energy_loss']}, pixels = {alma_comparison['aperture_pixel_count']}")

        # Disabled: ALMA SSIM/NCC and pixel chi2 are neither scored nor plotted.
        # metrics_alma = obp.full_image_metrics_noshift(
        # alma_cont, simulated_itot_as_data,
        # normalize="zscore",          # good default for morphology
        # ssim_win=None,                 # 7–15 is typical
        # return_pixel_chi2=True
        # )
        
        #loss_alma= 1-(metrics_alma['ssim']+metrics_alma['ncc'])/2 #
        loss_alma=alma_comparison["residual_energy_loss"] 
        #loss_alma=chi2_red_alma_profiles

        additional_info["alma"] = {
            "alignment": alma_alignment,
            # "ssim": metrics_alma.get("ssim"),
            # "ncc": metrics_alma.get("ncc"),
            "loss": loss_alma
        }

        additional_info["alma"]["comparison"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                                                for key, value in alma_comparison.items()}

        # print(f'[obriy_mcfost] ALMA metrics: SSIM={metrics_alma["ssim"]}, NCC={metrics_alma["ncc"]}, aperture residual-energy loss = {alma_comparison["residual_energy_loss"]}')
            
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




def plot_mcfost_density_temperature_cuts(
    model_dir,
    *,
    az_disk=0,
    height_ratios=(0.0, 0.15),
    reference_radius_au=None,
):
    """Plot radial dust-density and temperature cuts.

    Assumes the existing reader's cylindrical grid layout:
        grid: (2, n_az, n_z, n_rad), coordinates in au
        quantities: (n_az, n_z, n_rad), or (n_z, n_rad) when n_az=1

    Midplane uses the cell nearest z=0.
    Other cuts interpolate vertically without extrapolation.
    """
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits

    model_dir = Path(model_dir)

    def read_array(relative_path):
        path = model_dir / relative_path
        if not path.is_file():
            path = Path(str(path) + ".gz")
        with fits.open(path) as hdul:
            return np.asarray(hdul[0].data, dtype=float).copy()

    grid = read_array("data_disk/grid.fits")
    if grid.ndim != 4 or grid.shape[0] != 2:
        raise ValueError(f"Unexpected grid shape: {grid.shape}")

    radius = grid[0, az_disk]
    height = grid[1, az_disk]
    spatial_shape = grid.shape[1:]

    quantities = {}
    for name, path in (
        ("density", "data_disk/dust_mass_density.fits"),
        ("temperature", "data_th/Temperature.fits"),
    ):
        values = read_array(path)
        # Some axisymmetric outputs omit the singleton azimuth axis. Restore
        # only that axis, and only when the grid confirms an exact spatial match.
        if spatial_shape[0] == 1 and values.shape == spatial_shape[1:]:
            values = values[np.newaxis, ...]
        if values.shape != spatial_shape:
            raise ValueError(
                f"{name}: expected {spatial_shape}, got {values.shape}. "
                "Select additional component axes explicitly."
            )
        quantities[name] = values[az_disk]

    radial_positions = np.median(radius, axis=0)
    if not np.allclose(
        radius, radial_positions[None, :], rtol=1e-6, atol=1e-10
    ):
        raise ValueError(
            "Expected cylindrical grid columns at constant radius."
        )

    def extract_cut(values, height_ratio):
        result = np.full(radial_positions.shape, np.nan)

        for j, r_au in enumerate(radial_positions):
            z, column = height[:, j], values[:, j]
            valid = np.isfinite(z) & np.isfinite(column)
            if not valid.any():
                continue

            z, column = z[valid], column[valid]
            order = np.argsort(z)
            z, column = z[order], column[order]

            if height_ratio == 0:
                result[j] = column[np.argmin(np.abs(z))]
            else:
                target_z = height_ratio * r_au
                if z[0] <= target_z <= z[-1]:
                    result[j] = np.interp(target_z, z, column)

        return result

    fig, axes = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True,
        gridspec_kw={"hspace": 0.05},
    )
    order = np.argsort(radial_positions)
    profiles = {}

    for height_ratio in height_ratios:
        label = (
            "Midplane (nearest cell)"
            if height_ratio == 0 else f"z/r = {height_ratio:g}"
        )
        profiles[height_ratio] = {
            "radius_au": radial_positions[order]
        }

        for ax, name in zip(axes, ("density", "temperature")):
            values = extract_cut(quantities[name], height_ratio)
            profiles[height_ratio][name] = values[order]
            plotted = np.where(values > 0, values, np.nan)
            ax.plot(
                radial_positions[order], plotted[order], label=label
            )

    axes[0].set_ylabel(r"$\rho_{\rm dust}$ [g cm$^{-3}$]")
    axes[1].set_ylabel("T [K]")
    axes[1].set_xlabel("Cylindrical radius [au]")
    axes[0].set_title(model_dir.name)
    axes[0].legend()

    for ax in axes:
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        if reference_radius_au is not None:
            ax.axvline(
                reference_radius_au,
                color="grey", linestyle="--", linewidth=1,
            )

    output = model_dir / "figures" / "density_temperature_cuts.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return profiles
