
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

    run_mcfost_safe(Path(folder+'/simulation.para'), Path(folder), options=["-img", f"{wavelength}"])



def write_mcfost_paramfile(cfg: Dict[str, Any], fidelity: Dict[str, Any], outdir: Path) -> Path:
    """
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
    # Set fidelity-related params
    # pf.set_param("nbr_photons_eq_th", fidelity["nbr_photons_eq_th"])
    # pf.set_param("nbr_photons_lambda", fidelity["nbr_photons_lambda"])
    # pf.set_param("nbr_photons_image", fidelity["nbr_photons_image"])

    if fidelity["stage"] == "F0":
        #set up pixel scale for F0
        pf.set_param("grid_nx", 512)
        pf.set_param("grid_ny", 512)
        pf.set_param("grid_size", 400)  # VALUES to be adjusted 

    # Save the modified file
    pf.save(param_path)
    
    return param_path


def run_mcfost(fidelity: dict, param_path: Path, workdir: Path) -> None:

    print(f"run mcfost in {workdir}")
    # base

    run_mcfost_safe(param_path, workdir, options=[], logfile="mcfost_base.log")

    if fidelity["stage"] in ("F1", "F2", "F3"):
        for w in [1.63, 2.20, 3.50, 10.0]:
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")
    if fidelity["stage"] == "F2":
        for w in [0.55, 0.82]:
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")
    if fidelity["stage"] == "F3": # all wavelengths for chromatic visibilities (PIONIER, MATISSE, GRAVITY) + full PDI
        for w in [1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,
                  1.95,2.0,2.05,2.1,2.15,2.2,2.25,2.3,2.35,2.4,2.45,2.5,
                  2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0,4.1,4.2,4.3,
                  7,8,9,10,11,12,13,14]:
            run_mcfost_safe(param_path, workdir, options=["-img", f"{w}"], logfile=f"mcfost_{w:.2f}.log")






def load_and_score_outputs(fidelity: Dict[str, Any], workdir: Path, data_arg:Dict[str, Any]) -> float:
    """
    Read MCFOST outputs and compute a single scalar loss.
    Recommended: Gaussian-error negative log-likelihood combining SED/vis2/PDI.
    

    Return the likelihood (lower is better).
    """

    data_sed = data_arg[0]
    if fidelity["stage"] in ["F1", "F2", "F3"]:
        container_data_pionier = data_arg[1]
        container_data_gravity = data_arg[2]
        container_data_matisse_l = data_arg[3]
        container_data_matisse_n = data_arg[4]
    if fidelity["stage"] in ['F2','F3']:
        pdi_data_v = data_arg[5] #each disc with data not deconvolved q_phi, u_phi, pi, and psf
        pdi_data_i = data_arg[6] 
        pdi_data_h = data_arg[7]

    simulation_name = workdir.name

    sed_path = workdir / "data_th" / "sed_rt.fits.gz"
    if not sed_path.exists():
        print(f"SED file {sed_path} not found.")
        # trial ran but produced no SED -> invalid config or earlier failure
        return 1e99
    
    chi2_sed, chi2_reduced_sed, loglike_sed= obs.chi2_SED_with_reddening(str(workdir.name), str(workdir.parent)+'/', data_wave=data_sed[0], data_flux=data_sed[1],data_err=data_sed[2],
                                       plot=True, description=simulation_name)
    
    if fidelity["stage"] in ["F1", "F2", "F3"]:
        chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.monochromatic_chi(str(workdir), img_dir="data_1.63/", container_data=container_data_pionier, vistype='vis2', plot=True, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER 1.63", log_plotv=False)
        chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.monochromatic_chi(str(workdir), img_dir="data_2.2/", container_data=container_data_gravity, vistype='vis2', plot=True, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY 2.2", log_plotv=False)
        chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.monochromatic_chi(str(workdir), img_dir="data_3.5/", container_data=container_data_matisse_l,vistype='vis2', plot=True, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L 3.5", log_plotv=True)
        chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.monochromatic_chi(str(workdir), img_dir="data_10.0/", container_data=container_data_matisse_n, vistype='vis', plot=True, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N 10.0", log_plotv=False)
        
    if fidelity["stage"] in ['F2','F3', "F4", "F5"]:
        results_i=obp.polarimetric_analysis(str(workdir), 0.55, distance_pc= 1220.0, camera='zimpol',convolution_mode='file', psf_array=pdi_data_i['psf'],psf_cut=100, 
                                                                                                    image_scale='asinh', radial_limit_mas=500.0,
                                                                                                    deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                    theta0=0.0, plot=True, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Iband')
        data_cropped_i, model_cropped_i= obp.crop_to_same_size(pdi_data_i['pi'], results_i['mcfost_convolved_unresolved_corrected']['pi']) 
        
        metrics_i = obp.full_image_metrics_noshift(
            data_cropped_i, model_cropped_i,
            normalize="zscore",          # good default for morphology
            ssim_win=11,                 # 7–15 is typical
            return_pixel_chi2=True
        )
        
        obp.plot_polarimetric_image(metrics_i["ssim_image"], 3.6, title='ssim', save=str(workdir)+'/figures'+'/ssim_image_I.png', image_scale='linear', roi_half_size=50)

        
        results_v=obp.polarimetric_analysis(str(workdir), 0.82, distance_pc= 1220.0, camera='zimpol',convolution_mode='file', psf_array=pdi_data_v['psf'],psf_cut=100, 
                                                                                                    image_scale='asinh', radial_limit_mas=500.0,
                                                                                                    deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                    theta0=0.0, plot=True, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Vband')
        data_cropped_v, model_cropped_v= obp.crop_to_same_size(pdi_data_v['pi'], results_v['mcfost_convolved_unresolved_corrected']['pi']) 
        
        metrics_v = obp.full_image_metrics_noshift(
            data_cropped_v, model_cropped_v,
            normalize="zscore",          # good default for morphology
            ssim_win=11,                 # 7–15 is typical
            return_pixel_chi2=True
        )
        
        obp.plot_polarimetric_image(metrics_v["ssim_image"], 3.6, title='ssim', save=str(workdir)+'/figures'+'/ssim_image_V.png', image_scale='linear', roi_half_size=50)


        results_h=obp.polarimetric_analysis(str(workdir), 1.63, distance_pc= 1220.0, camera='irdis',convolution_mode='file', psf_array=pdi_data_h['psf'],psf_cut=100, 
                                                                                                    image_scale='asinh', radial_limit_mas=500.0,
                                                                                                    deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                    theta0=0.0, plot=True, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name+'_Hband')
        
        obp.plot_polarimetric_image(pdi_data_h['pi'], 12.27, title='IRAS08544-4431 H-band PI_check', save=str(workdir)+'/pi_h_band_check.png', image_scale='asinh', roi_half_size=30)

        data_cropped_h, model_cropped_h= obp.crop_to_same_size(pdi_data_h['pi'], results_h['mcfost_convolved_unresolved_corrected']['pi']) 
        
        metrics_h = obp.full_image_metrics_noshift(
            data_cropped_h, model_cropped_h,
            normalize="zscore",          # good default for morphology
            ssim_win=11,                 # 7–15 is typical
            return_pixel_chi2=True
        )
        
        obp.plot_polarimetric_image(metrics_h["ssim_image"], 12.27, title='ssim', save=str(workdir)+'/figures'+'/ssim_image_H.png', image_scale='linear', roi_half_size=30)
        if fidelity["stage"] in ["F2", "F3" ]:
            chi2_sum_pdi_h, chi2_red_pdi_h, loglike_pdi_h, n_data_points_pdi_h= obp.profiles_chi2(pdi_data_h['pi'], results_h['mcfost_convolved_unresolved_corrected']['pi'], ps=12.27, profile_type='both', mode='sum', plot=True, 
                                                                                                save=str(workdir)+'/figures/'+simulation_name+'_Hband', az_nbins=18)
            chi2_sum_pdi_v, chi2_red_pdi_v, loglike_pdi_v, n_data_points_pdi_v= obp.profiles_chi2(pdi_data_v['pi'], results_v['mcfost_convolved_unresolved_corrected']['pi'], ps=3.6, profile_type='both', mode='sum', plot=True, 
                                                                                                save=str(workdir)+'/figures/'+simulation_name+'_Vband', az_nbins=18)
            chi2_sum_pdi_i, chi2_red_pdi_i, loglike_pdi_i, n_data_points_pdi_i= obp.profiles_chi2(pdi_data_i['pi'], results_i['mcfost_convolved_unresolved_corrected']['pi'], ps=3.6, profile_type='both', mode='sum', plot=True, 
                                                                                                save=str(workdir)+'/figures/'+simulation_name+'_Iband', az_nbins=18)   
            
        
        loss_v=1-metrics_v['ssim']
        loss_i=1-metrics_i['ssim']
        loss_h=1-metrics_h['ssim']
        print(f"PDI SSIM losses: I-band: {loss_i}, V-band: {loss_v}, H-band: {loss_h}")


    chi_total= chi2_sed 
    num_points_total= len(data_sed[0]) 
    loglike_total=loglike_sed
    i_num=1
    if fidelity["stage"] in ["F1", "F2", "F3"]:
        chi_total+= chi2_pionier + chi2_gravity + chi2_matisse_l + chi2_matisse_n
        num_points_total+=num_points_pionier + num_points_gravity + num_points_matisse_l + num_points_matisse_n
        loglike_total+=loglike_pionier+loglike_gravity+loglike_matisse_l+loglike_matisse_n
        i_num+=4
    if fidelity["stage"] in ['F2','F3']:
        chi_total+= chi2_sum_pdi_i + chi2_sum_pdi_v + chi2_sum_pdi_h
        num_points_total+= n_data_points_pdi_i + n_data_points_pdi_v + n_data_points_pdi_h
        loglike_total+= loglike_pdi_i + loglike_pdi_v + loglike_pdi_h
        i_num+=3
         
    
    chi2_red_total = chi_total/(num_points_total-i_num)  # reduced chi2 - not sure about number of free parameters here
    print(f"Total reduced chi2: {chi2_red_total}, loglike: {loglike_total}")

    if fidelity["stage"] in ["F5"]:
        chi2_red_total+= (loss_h + loss_v + loss_i)*100  # weighting factor to bring SSIM losses to similar scale as chi2
   
    return chi2_red_total
