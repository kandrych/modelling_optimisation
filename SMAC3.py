"""
SMAC3 + Dask(SLURM) + Multi‑Fidelity skeleton 

- Cluster launcher (Dask + dask_jobqueue.SLURMCluster)
- ConfigSpace with conditional ("flexible") hyperparameters
- Multi-fidelity mapping (budget -> photon packets / resolution / stage F0→F3)
- Objective wrapper that shells out to MCFOST and returns a scalar loss
- SMAC3 MultiFidelityFacade orchestration (parallel, async-friendly)

Test locally first, then switch to SLURM.
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
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

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

from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from ConfigSpace.conditions import InCondition, EqualsCondition
import pymcfost as mcfost  # MCFOST Python bindings
import distroi

sys.path.append(os.path.abspath(".."))  # parent of current working dir
#import lib.Katya_func as kf

import lib.obriy_general as obg
import lib.obriy_sed as obs
import lib.obriy_interferometry as obi
import lib.obriy_polarimetry as obp
import lib.obriy_mcfost as obm





# -----------------------------------------------------------------------------
# Cluster setup
# -----------------------------------------------------------------------------

def start_cluster(n_workers: int, processes_per_worker: int, use_slurm: bool) -> Client:
    """
    Start a Dask cluster. For quick local testing, set use_slurm=False and
    rely on default LocalCluster via `Client()`.

    For SLURM, adjust walltime/cores/memory/partition/etc. 🔧
    """
    if not use_slurm:
        client = Client()  # LocalCluster default
        print("[cluster] Local Dask cluster started:", client)
        return client

    cluster = SLURMCluster(
        queue="normal",  # partition
        cores=processes_per_worker,
        processes=1,
        memory="10GB",
        walltime="02:00:00",
        job_extra=[
            "--export=ALL",
            "--job-name=smac_postAGB",
            "--cpus-per-task=16",
            "--output=job.in.qout",
            "--mail-type=BEGIN",
            "--mail-type=END",
            "--mail-type=FAIL",
            "--mail-user=kateryna.andrych@mq.edu.au"
        ],
        interface=None,  # or e.g. "ib0" if needed
        python=sys.executable,
        job_script_prologue=['echo "HOSTNAME = $HOSTNAME"','echo "HOSTTYPE = $HOSTTYPE"', "echo Time is `date`", "echo Directory is `pwd`"]
    )

    cluster.adapt(minimum=n_workers, maximum=n_workers)
    client = Client(cluster)
    print("[cluster] SLURM Dask cluster started:", client)
    return client

# -----------------------------------------------------------------------------
# ConfigSpace (flexible / conditional hyperparameters)
# -----------------------------------------------------------------------------

def build_configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace()

    # --- Example core physical knobs (change here parameters) ---
    alpha = Float("alpha_viscosity", (0.0000001, 0.1), default=0.01, log=True)
    Rc = Float("zone_1_Rc", (7.3, 50.0), default=10.0)
    surface_density_exp = Float("zone_1_surface_density_exp", (-1.5, 1.5), default=1.0)
    gamma_exp = Float("zone_1_gamma_exp", (-2.0, -0.2), default=-1.0)   

    # Flexible dust law choice unlocks different parameters
    # dust_law = Categorical("dust_law", ["MRN", "powerlaw"], default="MRN")
   
    # q = Float("q", (-4.5, -2.0), default=-3.5)  # only if powerlaw

    cs.add([alpha, Rc, surface_density_exp, gamma_exp])

    # --- Optional conditions (uncomment if you want them active) ---
    # Only use a_min/a_max when dust_law == "MRN"
    # Only use q when dust_law == "powerlaw"
    # cs.add([
    #     EqualsCondition(a_min, dust_law, "MRN"),
    #     EqualsCondition(a_max, dust_law, "MRN"),
    #     EqualsCondition(q,      dust_law, "powerlaw"),
    # ])

    # --- Optional forbiddens example ---
    # forb = ForbiddenAndConjunction(
    #     ForbiddenEqualsClause(dust_law, "powerlaw"),
    #     ...
    # )
    # cs.add_forbidden_clause(forb)

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
    # Example budget range: min=0.25, max=3.0 (F0=0.25, F1=1.0, F2=2.0, F3=3.0)
    stage = "F0"
    if budget >= 1.0:
        stage = "F1"
    if budget >= 2.0:
        stage = "F2"
    if budget >= 3.0:
        stage = "F3"

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
        products = ["sed", "vis2_1perband", "pdi_radial"]
    else:  # F3
        img_res = 2
        products = ["sed", "vis2_chromatic", "pdi_radial", "pdi_colour"]

    return {
        "stage": stage,
        # "nbr_photons_eq_th": nbr_photons_eq_th,
        # "nbr_photons_lambda": nbr_photons_lambda,
        # "nbr_photons_image": nbr_photons_image,
        "image_res": img_res,
        "products": products,
    }

# -----------------------------------------------------------------------------
# External simulator glue (MCFOST)
# -----------------------------------------------------------------------------

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
    pf = obm.ParaFile(str(outdir.parent/"simulation.para"))
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


def run_mcfost(fidelity: Dict[str, Any], param_path: Path, workdir: Path) -> None:
    """
    Execute MCFOST. 
    If your cluster nodes have the binary on PATH, this will run inline
    within the Dask worker. Otherwise, prepend module loads or call a wrapper script.
    
    """
    workdir=str(workdir)+"/"
    param_path=str(param_path)

    pf = obm.ParaFile(param_path)
    
    os.makedirs(workdir, exist_ok=True)  # no error if it already exists

    # Save the modified file
    pf.save(workdir+"simulation.para")
    try:
        print('run mcfost in '+workdir)
        os.chdir(workdir) #this is to change directory to where the simulation.para file is and then run mcfost there
        mcfost.run(workdir+'/simulation.para', delete_previous=False, silent=True) #for all stages 

        if fidelity["stage"] in ["F1", "F2", "F3"]:
            for wave in [1.63, 2.20, 3.50, 10.0]:
                mcfost.run(workdir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
        if fidelity["stage"] == "F2":
            for wave in [0.55, 0.82]: #V and I bands for polarimetric images, H band is run above
                mcfost.run(workdir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
        if fidelity["stage"] == "F3":
            for wave in [1.5, 1.55, 1.6,1.65,  1.7, 1.75,  1.8, 1.85, 1.9]: #spectral vis2 for PIONIER
                mcfost.run(workdir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
            for wave in [1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4, 2.45, 2.5]: #spectral vis2 for GRAVITY
                mcfost.run(workdir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
            for wave in [2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3]: #spectral vis2 for MATISSE L band
                mcfost.run(workdir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
            for wave in [7, 8, 9, 10, 11, 12, 13, 14]: #spectral vis2 for MATISSE N band
                mcfost.run(workdir+'/simulation.para',options = "-img "+str(wave), delete_previous=False, silent=True)
    except:
        print(f"MCFOST run failed for fidelity: {fidelity['stage']} and parameter file {param_path}")
        raise RuntimeError(f"MCFOST run failed for fidelity: {fidelity['stage']} and parameter file {param_path}")


def load_and_score_outputs(fidelity: Dict[str, Any], workdir: Path, data_arg:Dict[str, Any]) -> float:
    """
    Read MCFOST outputs and compute a single scalar loss.
    Recommended: Gaussian-error negative log-likelihood combining SED/vis2/PDI.
    

    Return the likelihood (lower is better).
    """

    data_sed = data_arg[0]
    container_data_pionier = data_arg[1]
    container_data_gravity = data_arg[2]
    container_data_matisse_l = data_arg[3]
    container_data_matisse_n = data_arg[4]
    pdi_data_v = data_arg[5] #each disc with data not deconvolved q_phi, u_phi, pi, and psf
    pdi_data_i = data_arg[6] 
    pdi_data_h= data_arg[7]
    simulation_name=workdir.name


    chi2_sed, chi2_reduced_sed, loglike_sed= obs.chi2_SED_with_reddening(str(workdir.name), str(workdir.parent)+'/', data_wave=data_sed[0], data_flux=data_sed[1],data_err=data_sed[2],
                                       plot=True, description=simulation_name)
    
    if fidelity["stage"] in ["F1", "F2", "F3"]:
        chi2_pionier, chi2_red_pionier, loglike_pionier, num_points_pionier= obi.monochromatic_chi(str(workdir), img_dir="data_1.63/", container_data=container_data_pionier, vistype='vis2', plot=True, fig_dir=str(workdir)+'/figures/', extra_title="PIONIER 1.63", log_plotv=False)
        chi2_gravity, chi2_red_gravity, loglike_gravity, num_points_gravity= obi.monochromatic_chi(str(workdir), img_dir="data_2.2/", container_data=container_data_gravity, vistype='vis2', plot=True, fig_dir=str(workdir)+'/figures/', extra_title="GRAVITY 2.2", log_plotv=False)
        chi2_matisse_l, chi2_red_matisse_l, loglike_matisse_l, num_points_matisse_l= obi.monochromatic_chi(str(workdir), img_dir="data_3.5/", container_data=container_data_matisse_l,vistype='vis2', plot=True, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE L 3.5", log_plotv=True)
        chi2_matisse_n, chi2_red_matisse_n, loglike_matisse_n, num_points_matisse_n= obi.monochromatic_chi(str(workdir), img_dir="data_10.0/", container_data=container_data_matisse_n, vistype='vis', plot=True, fig_dir=str(workdir)+'/figures/', extra_title="MATISSE N 10.0", log_plotv=False)
        
    if fidelity["stage"] in ['F2','F3']:
        results_i=obp.polarimetric_analysis(str(workdir), 0.55, distance_pc= 1220.0, camera='zimpol',convolution_mode='file', psf_array=pdi_data_i['psf'],psf_cut=100, 
                                                                                                    image_scale='asinh', radial_limit_mas=150.0,
                                                                                                    deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                    theta0=0.0, plot=True, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name)
        chi2_sum_pdi_i, chi2_red_pdi_i, loglike_pdi_i, n_data_points_pdi_i= obp.profiles_chi2(pdi_data_i['pi'], results_i['mcfost_convolved_unresolved_corrected']['pi'], ps=3.6, profile_type='both', mode='sum', plot=True,
                                                                                            save=str(workdir)+'/figures/'+simulation_name, az_nbins=18)
        
        results_v=obp.polarimetric_analysis(str(workdir), 0.82, distance_pc= 1220.0, camera='zimpol',convolution_mode='file', psf_array=pdi_data_v['psf'],psf_cut=100, 
                                                                                                    image_scale='asinh', radial_limit_mas=150.0,
                                                                                                    deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                    theta0=0.0, plot=True, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name)
        chi2_sum_pdi_v, chi2_red_pdi_v, loglike_pdi_v, n_data_points_pdi_v= obp.profiles_chi2(pdi_data_v['pi'], results_v['mcfost_convolved_unresolved_corrected']['pi'], ps=3.6, profile_type='both', mode='sum', plot=True, 
                                                                                             save=str(workdir)+'/figures/'+simulation_name, az_nbins=18)
        
        results_h=obp.polarimetric_analysis(str(workdir), 1.63, distance_pc= 1220.0, camera='irdis',convolution_mode='file', psf_array=pdi_data_h['psf'],psf_cut=100, 
                                                                                                    image_scale='asinh', radial_limit_mas=150.0,
                                                                                                    deprojection=(0, 0), azimuthal_r_in_mas=0.0, azimuthal_r_out_mas=500.0, azimuthal_nbins=18,
                                                                                                    theta0=0.0, plot=True, roi_size_half=30, fig_dir=str(workdir)+'/figures/', extra_title=simulation_name)
        chi2_sum_pdi_h, chi2_red_pdi_h, loglike_pdi_h, n_data_points_pdi_h= obp.profiles_chi2(pdi_data_h['pi'], results_h['mcfost_convolved_unresolved_corrected']['pi'], ps=12.27, profile_type='both', mode='sum', plot=True, 
                                                                                             save=str(workdir)+'/figures/'+simulation_name, az_nbins=18)


    
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

    return chi2_red_total

# -----------------------------------------------------------------------------
# Objective callable for SMAC (supports multi-fidelity via `budget` argument)
# -----------------------------------------------------------------------------


def load_data(data_root: str) -> Dict[str, Any]:
    #filename of SED catalogue data file
    if data_root =='demo_mac':
        data_filename = '/Users/katerynaandrych/Work/lin/Postdoc/Data/interferometry/IRAS08544-4431/SED/IRAS08544-4431.phot'
        data_wave, data_flux, data_err = obs.load_sed_data(data_filename)
            
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

         #real PSF from observations
        star_psf='HD83878'
        figfolder_psf='/Users/katerynaandrych/Work/lin/PhD/SPHERE_reduction_data/paper2/mean_combined/'+star_psf+'/'
        file_psf=star_psf+'_'+'V'+'_'+'I'+'_meancombined.fits'
        psf_v=obp.Loadimage(figfolder_psf,file_psf)
        file_psf=star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits'
        psf_i=obp.Loadimage(figfolder_psf,file_psf)

        #polarimetric observations
        pdi_folder_v = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/V_band/'
        pdi_file_v = 'IRAS08544-4431_dc_notnorm_V_PI_corr_tel+unres.fits'
        pdi_v= obp.Loadimage(pdi_folder_v,pdi_file_v)
        pdi_folder_i = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/I_band/'
        pdi_file_i = 'IRAS08544-4431_dc_notnorm_I_PI_corr_tel+unres.fits'
        pdi_i= obp.Loadimage(pdi_folder_i,pdi_file_i)
        pdi_folder_h = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/H_band/'
        pdi_file_h = 'iras08544-4431_calib_H_PI_corr_tel+unres.fits'
        pdi_h= obp.Loadimage(pdi_folder_h,pdi_file_h)
        file_psf='iras08544-4431_calib_H_I_meancombined.fits'
        psf_h=obp.Loadimage(pdi_folder_h,file_psf)

        
        pdi_data_v={'psf': psf_v, 'pi': pdi_v}
        pdi_data_i={'psf': psf_i, 'pi': pdi_i}
        pdi_data_h={'psf': psf_h, 'pi': pdi_h}
        data_sed = [data_wave, data_flux, data_err]
        data_arrays = [data_sed, container_data_pionier, container_data_gravity, container_data_matisse_l, container_data_matisse_n,pdi_data_v, pdi_data_i, pdi_data_h]
        print('data loaded')

    return data_arrays

def make_unique_trial_dir(scratch_root: Path, seed: int, budget: float) -> Path:
    """
    Create a unique trial directory like:
      trial_seed42_budget1.50_20251029-204211
      trial_seed42_budget1.50_20251029-204211_001
      ...
    Uses atomic mkdir to avoid races.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    base = Path(scratch_root) / f"trial_seed{seed}_budget{budget:.2f}_{ts}"
    trial_dir = base
    idx = 0
    while True:
        try:
            trial_dir.mkdir(parents=True, exist_ok=False)  # atomic: raises if exists
            return trial_dir
        except FileExistsError:
            idx += 1
            trial_dir = base.with_name(f"{base.name}_{idx:03d}")

def objective(cfg: Dict[str, Any], seed: int, budget: float, data_arg: Dict[str, Any], scratch_root: str) -> float:
    fidelity = map_budget_to_fidelity(budget)

    # Each trial gets a private scratch dir
    # trial_dir = Path(scratch_root) / f"trial_seed{seed}_budget{budget:.2f}" /time.strftime("%Y%m%d-%H%M%S")
    # trial_dir.mkdir(parents=True, exist_ok=True)
    trial_dir = make_unique_trial_dir(Path(scratch_root), seed, budget)

    # Convert cfg (ConfigSpace.Configuration) to dict
    if hasattr(cfg, "get_dictionary"):
        cfg = dict(cfg)

    # Write param file and run MCFOST
    par_path = write_mcfost_paramfile(cfg, fidelity, trial_dir)
    run_mcfost(fidelity,par_path, trial_dir)

    # Score outputs

    loss = load_and_score_outputs(fidelity, trial_dir, data_arg)

    return float(loss)

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
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    client = start_cluster(args.n_workers, args.procs_per_worker, args.use_slurm)

    cs = build_configspace()

    scenario = Scenario(
        configspace=cs,
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
    
    data_arg = load_data(args.data_root)
    # SMAC Objective wrapper with extra kwargs via lambda/closure
    def smac_objective(cfg, seed: int, budget: float) -> float:
        return objective(cfg, seed, budget, data_arg=data_arg, scratch_root=work_root)

    smac = MultiFidelityFacade(
        scenario=scenario,
        target_function=smac_objective,
        # intensifier defaults to Hyperband; override via intensifier if desired
        overwrite=False,
        dask_client=client,
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
    fidelity_result['stage']='F1'
    # Write param file and run MCFOST
    par_path = write_mcfost_paramfile(incumbent, fidelity_result, results_dir)
    run_mcfost(fidelity_result,par_path, results_dir)
    # Score outputs
    loss = load_and_score_outputs(fidelity_result, results_dir, data_arg)




    # Trajectory / RunHistory (optional deep dive)
    runhistory = smac.runhistory
    traj = smac.intensifier.trajectory
    print(f"Trials run: {len(runhistory)}")

    # Persist runhistory in JSON for later analysis
    try:
        from smac.utils.logging import save_configspace
        save_configspace(cs, results_dir / "configspace.json")
    except Exception:
        pass


if __name__ == "__main__":
    main()
