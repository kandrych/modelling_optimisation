
import numpy as np
import pandas as pd
from typing import Literal, Tuple, Dict, Optional, Union, Any, List


import os
import matplotlib.pyplot as plt

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
        run_mcfost_safe(param_path, workdir, options=["-puffed_up_rim", f"{cfg.get('puffed_h_rim_over_h0', 0)}", f"{cfg.get('puffed_r_rim', 0)}", f"{cfg.get('puffed_delta_r', 0)}"], logfile="mcfost_base.log")
    else:
        run_mcfost_safe(param_path, workdir, options=[], logfile="mcfost_base.log")
    
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
    
    if "vis2_chromatic" in fidelity["products"]: # all wavelengths for chromatic visibilities (PIONIER, MATISSE, GRAVITY) + full PDI
        for w in [0.55, 0.82, 1.5,1.55,1.6,1.63,1.65,1.7,1.75,1.8,1.85,1.9,
                  1.95,2.0,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,
                  2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,
                  7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0]:
            if os.path.exists(str(workdir)+"/data_"+str(w)+"/"):   
                print(f"Image at {w} micron already exists in {str(workdir)+'data_'+str(w)+'/'} folder. Skipping simulation.")
                continue
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")




def load_and_score_outputs(fidelity: Dict[str, Any], workdir: Path, data_arg:Dict[str, Any], args) -> float:
    """
    Read MCFOST outputs and compute a single scalar loss.
    Recommended: Gaussian-error negative log-likelihood combining SED/vis2/PDI.
    

    Return the likelihood (lower is better).
    """
    print(f'[obriy_mcfost] fidelity["stage"] = {fidelity["stage"]}')
    print(f'[obriy_mcfost] fidelity["products"] = {fidelity["products"]}')  
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
    
    print('[obriy_mcfost] Data for scoring loaded successfully')

    simulation_name = workdir.name

    sed_path = workdir / "data_th" / "sed_rt.fits.gz"
    if not sed_path.exists():
        print(f"Temperature file {sed_path} not found.")
        # trial ran but produced no SED -> invalid config or earlier failure
        return 1e99
    
    if "sed" in fidelity["products"]:

        chi2_sed, chi2_reduced_sed, loglike_sed= obs.chi2_SED_with_reddening(str(workdir.name), str(workdir.parent)+'/', data_wave=data_sed[0], data_flux=data_sed[1],data_err=data_sed[2],
                                                plot=True, description=simulation_name)
    
    if "vis2_1perband" in fidelity["products"]:
        chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.monochromatic_chi(str(workdir), img_dir="data_1.63/", container_data=container_data_pionier, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER 1.63", log_plotv=False)
        chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.monochromatic_chi(str(workdir), img_dir="data_2.2/", container_data=container_data_gravity, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY 2.2", log_plotv=False)
        chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.monochromatic_chi(str(workdir), img_dir="data_3.5/", container_data=container_data_matisse_l,vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L 3.5", log_plotv=True)
        chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.monochromatic_chi(str(workdir), img_dir="data_10.0/", container_data=container_data_matisse_n, vistype='vis', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N 10.0", log_plotv=False)
    
    if "vis2_chromatic" in fidelity["products"]:
        chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.monochromatic_chi(str(workdir), img_dir="", container_data=container_data_pionier, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER", log_plotv=False)
        chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.monochromatic_chi(str(workdir), img_dir="", container_data=container_data_gravity, vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY", log_plotv=False)
        chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.monochromatic_chi(str(workdir), img_dir="", container_data=container_data_matisse_l,vistype='vis2', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L", log_plotv=True)
        chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.monochromatic_chi(str(workdir), img_dir="", container_data=container_data_matisse_n, vistype='vis', plot=args.plot_intermediate, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N", log_plotv=False)
    
    if ("pdi_I" in fidelity["products"]) or ("pdi_V" in fidelity["products"]) or ("pdi_H" in fidelity["products"]):
        loss_i=np.nan
        loss_v=np.nan
        loss_h=np.nan

        print('[obriy_mcfost] Polarimetric analysis started')
        if "pdi_I" in fidelity["products"]:
    
            results_i=obp.polarimetric_analysis(str(workdir), 0.55, camera='zimpol',convolution_mode='file', psf_array=pdi_data_i['psf'], psf_cut=100, 
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Iband')
            
            if args.correct_unresolved_polarimetry:
                print('[obriy_mcfost] Applying unresolved polarization correction for I band')
                data_cropped_i, model_cropped_i= obp.crop_to_same_size(pdi_data_i['pi'], results_i['mcfost_convolved_unresolved_corrected']['pi'])
                model_rad_prof_pi= results_i['mcfost_convolved_unresolved_corrected']['radial_profile_pi']
                model_azimuthal_prof_pi= results_i['mcfost_convolved_unresolved_corrected']['azimuthal_profile_pi']
                
            else:
                print('[obriy_mcfost] No unresolved polarization correction applied for I band')
                data_cropped_i, model_cropped_i= obp.crop_to_same_size(pdi_data_i['pi'], results_i['mcfost_convolved']['pi'])
                model_rad_prof_pi= results_i['mcfost_convolved']['radial_profile_pi']
                model_azimuthal_prof_pi= results_i['mcfost_convolved']['azimuthal_profile_pi']
                   
            # Calculate metrics for arcsinh-scaled images to highlight morphology
            obs_rad_prof_pi, obs_az_prof_pi = obp.profiles(data_cropped_i, 3.6, 
                                            profile_type="both",
                                            mode="sum",
                                            radial_limit_mas=500,
                                            plot=args.plot_intermediate,
                                            save_prefix=str(workdir)+'/figures/'+ "data_i_",
                                            deprojection_inc_pa_deg=0.0,
                                            center=None,
                                            az_nbins=20,
                                            azimuthal_r_in_mas=0.0,
                                            azimuthal_r_out_mas=500.0,
                                            theta0=0.0
                                            ) 
            
            profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof_pi, model_rad_prof_pi, 3.6, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_i_")
            profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof_pi, 3.6, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_i_")
            profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike


            metrics_i = obp.full_image_metrics_noshift(
                np.arcsinh(data_cropped_i), np.arcsinh(model_cropped_i),
                normalize="zscore",          # good default for morphology
                ssim_win=11,                 # 7–15 is typical
                return_pixel_chi2=True
            )
            if args.plot_intermediate:
                obp.plot_polarimetric_image(results_i['mcfost_convolved_unresolved_corrected']['pi_deconvolved'], 3.6, title=f'Model PI, conv, unres corr, decon', save=str(workdir)+'/figures'+'/model_pi_corr_conv_deconv_I.png', image_scale='asinh', roi_half_size=100)
                obp.plot_polarimetric_image(results_i['mcfost_convolved']['pi_deconvolved'], 3.6, title=f'Model PI, conv, decon', save=str(workdir)+'/figures'+'/model_pi_conv_deconv_I.png', image_scale='asinh', roi_half_size=100)

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
            loss_i=1-(metrics_i['ssim']+metrics_i['ncc'])/2 + profile_pi_chi2_red # weights can be adjusted
       



        if "pdi_V" in fidelity["products"]:
    
            results_v=obp.polarimetric_analysis(str(workdir), 0.82, camera='zimpol',convolution_mode='file', psf_array=pdi_data_v['psf'],psf_cut=100, 
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Vband')
            if args.plot_intermediate:
                obp.plot_polarimetric_image(results_v['mcfost_convolved_unresolved_corrected']['pi_deconvolved'], 3.6, title=f'Model PI, conv, unres corr, decon', save=str(workdir)+'/figures'+'/model_pi_corr_conv_deconv_V.png', image_scale='asinh', roi_half_size=100)
                obp.plot_polarimetric_image(results_v['mcfost_convolved']['pi_deconvolved'], 3.6, title=f'Model PI, conv, decon', save=str(workdir)+'/figures'+'/model_pi_conv_deconv_V.png', image_scale='asinh', roi_half_size=100)
            
            if args.correct_unresolved_polarimetry:
                print('[obriy_mcfost] Applying unresolved polarization correction for I band')
                data_cropped_v, model_cropped_v= obp.crop_to_same_size(pdi_data_v['pi'], results_v['mcfost_convolved_unresolved_corrected']['pi'])
                model_rad_prof_pi= results_v['mcfost_convolved_unresolved_corrected']['radial_profile_pi']
                model_azimuthal_prof_pi= results_v['mcfost_convolved_unresolved_corrected']['azimuthal_profile_pi']
                 
            else:
                print('[obriy_mcfost] No unresolved polarization correction applied for I band')
                data_cropped_v, model_cropped_v= obp.crop_to_same_size(pdi_data_v['pi'], results_v['mcfost_convolved']['pi'])   
                model_rad_prof_pi= results_v['mcfost_convolved']['radial_profile_pi']
                model_azimuthal_prof_pi= results_v['mcfost_convolved']['azimuthal_profile_pi']

            obs_rad_prof_pi, obs_az_prof_pi = obp.profiles(data_cropped_v, 3.6, 
                                            profile_type="both",
                                            mode="sum",
                                            radial_limit_mas=500,
                                            plot=args.plot_intermediate,
                                            save_prefix=str(workdir)+'/figures/'+ "data_v_",
                                            deprojection_inc_pa_deg=0.0,
                                            center=None,
                                            az_nbins=20,
                                            azimuthal_r_in_mas=0.0,
                                            azimuthal_r_out_mas=500.0,
                                            theta0=0.0
                                            ) 
            
            profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof_pi, model_rad_prof_pi, 3.6, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_v_")
            profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof_pi, 3.6, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_v_")
            profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike

            metrics_v = obp.full_image_metrics_noshift(
                np.arcsinh(data_cropped_v), np.arcsinh(model_cropped_v),
                normalize="zscore",          # good default for morphology
                ssim_win=11,                 # 7–15 is typical
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
            loss_v=1-(metrics_v['ssim']+metrics_v['ncc'])/2 + profile_pi_chi2_red # weights can be adjusted
        


        if "pdi_H" in fidelity["products"]:
            results_h=obp.polarimetric_analysis(str(workdir), 1.63, camera='irdis',convolution_mode='file', psf_array=pdi_data_h['psf'],psf_cut=100, 
                                                                                                        image_scale='asinh', radial_limit_mas=500.0,
                                                                                                        deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                        theta0=0.0, plot=args.plot_intermediate, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Hband')
            
           
            if args.correct_unresolved_polarimetry:
                print('[obriy_mcfost] Applying unresolved polarization correction for H band')
                data_cropped_h, model_cropped_h= obp.crop_to_same_size(pdi_data_h['pi'], results_h['mcfost_convolved_unresolved_corrected']['pi']) 
                model_rad_prof_pi= results_h['mcfost_convolved_unresolved_corrected']['radial_profile_pi']
                model_azimuthal_prof_pi= results_h['mcfost_convolved_unresolved_corrected']['azimuthal_profile_pi']
            else:
                print('[obriy_mcfost] No unresolved polarization correction applied for H band')
                data_cropped_h, model_cropped_h= obp.crop_to_same_size(pdi_data_h['pi'], results_h['mcfost_convolved']['pi']) 
                model_rad_prof_pi= results_h['mcfost_convolved']['radial_profile_pi']
                model_azimuthal_prof_pi= results_h['mcfost_convolved']['azimuthal_profile_pi']
            
            obs_rad_prof_pi, obs_az_prof_pi = obp.profiles(data_cropped_h, 12.27, 
                                            profile_type="both",
                                            mode="sum",
                                            radial_limit_mas=500,
                                            plot=args.plot_intermediate,
                                            save_prefix=str(workdir)+'/figures/'+ "data_h_",
                                            deprojection_inc_pa_deg=0.0,
                                            center=None,
                                            az_nbins=20,
                                            azimuthal_r_in_mas=0.0,
                                            azimuthal_r_out_mas=500.0,
                                            theta0=0.0
                                            )       
            profile_rad_pi_chi2, _,profile_rad_pi_loglike, profile_rad_pi_npoints = obp.profile_chi2(obs_rad_prof_pi, model_rad_prof_pi, 12.27, profile_type="radial", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"radial_profile_pi_h_")
            profile_az_pi_chi2, _,profile_az_pi_loglike, profile_az_pi_npoints = obp.profile_chi2(obs_az_prof_pi, model_azimuthal_prof_pi, 12.27, profile_type="azimuthal", plot=args.plot_intermediate, save_prefix=str(workdir)+'/figures/'+"azimuthal_profile_pi_h_")
            profile_pi_chi2_red= (profile_rad_pi_chi2 + profile_az_pi_chi2) / (profile_rad_pi_npoints + profile_az_pi_npoints -2)
            profile_loglike= profile_rad_pi_loglike + profile_az_pi_loglike


            metrics_h = obp.full_image_metrics_noshift(
                np.arcsinh(data_cropped_h), np.arcsinh(model_cropped_h),
                normalize="zscore",          # good default for morphology
                ssim_win=11,                 # 7–15 is typical
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
            loss_h=1-(metrics_h['ssim']+metrics_h['ncc'])/2 + profile_pi_chi2_red # weights can be adjusted





        
         
        print(f"PDI losses: I-band: {loss_i}, V-band: {loss_v}, H-band: {loss_h}")
    #initialize totals so eve if there is no sed and interferometry - we can still compute pdi only chi2
    chi_total=0.0
    num_points_total=1
    i_num=0

    if "sed" in fidelity["products"]:
        
        chi_total= chi2_sed 
        num_points_total= len(data_sed[0]) 
        loglike_total=loglike_sed
        i_num=1

    if ("vis2_1perband" in fidelity["products"]) or ("vis2_chromatic" in fidelity["products"]):
        
        chi_total+= chi2_pionier + chi2_gravity + chi2_matisse_l + chi2_matisse_n
        num_points_total+=num_points_pionier + num_points_gravity + num_points_matisse_l + num_points_matisse_n
        loglike_total+=loglike_pionier+loglike_gravity+loglike_matisse_l+loglike_matisse_n
        i_num+=4

    chi2_red_total = chi_total/(num_points_total-i_num)  # reduced chi2 - not sure about number of free parameters here
    
    if "pdi_I" in fidelity["products"]:
        chi2_red_total+=(loss_i)*100 # weighting factor to bring SSIM losses to similar scale as chi2
   
    
    if "pdi_V" in fidelity["products"]:
        chi2_red_total+=(loss_v)*100 # weighting factor to bring SSIM losses to similar scale as chi2
   
    if "pdi_H" in fidelity["products"]:
        chi2_red_total+=(loss_h)*100 # weighting factor to bring SSIM losses to similar scale as chi2
   
        #chi2_red_total=metrics_i["chi2_red"]+metrics_v["chi2_red"] #sum of reduced chi2 values for I and V bands for AR Pup fitting

    print(f"Total reduced chi2: {chi2_red_total}, loglike: {loglike_total}")
    
    return chi2_red_total
