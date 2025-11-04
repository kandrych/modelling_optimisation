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
        client = Client()#(n_workers=n_workers, threads_per_worker=1)  # LocalCluster default
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

    cs = ConfigurationSpace.from_yaml("/Users/katerynaandrych/Work/lin/python scripts/modelling_optimisation/config/space.yaml")


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
# Objective callable for SMAC (supports multi-fidelity via `budget` argument)
# -----------------------------------------------------------------------------


def load_data(data_root: str, work_root: str) -> Dict[str, Any]:
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
        obp.plot_polarimetric_image(psf_v, 3.6, title='IRAS08544-4431 V-band PSF', save=work_root+'/psf_v_band.png', image_scale='asinh', roi_half_size=30)

        file_psf=star_psf+'_'+'I'+'_'+'I'+'_meancombined.fits'
        psf_i=obp.Loadimage(figfolder_psf,file_psf)
        obp.plot_polarimetric_image(psf_i, 3.6, title='IRAS08544-4431 I-band PSF', save=work_root+'/psf_i_band.png', image_scale='asinh', roi_half_size=30)


        #polarimetric observations
        pdi_folder_v = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/V_band/'
        pdi_file_v = 'IRAS08544-4431_dc_notnorm_V_PI_corr_tel+unres.fits'
        pdi_v= obp.Loadimage(pdi_folder_v,pdi_file_v)
        obp.plot_polarimetric_image(pdi_v, 3.6, title='IRAS08544-4431 V-band PI', save=work_root+'/pi_v_band.png', image_scale='asinh', roi_half_size=50)
        
        pdi_folder_i = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/I_band/'
        pdi_file_i = 'IRAS08544-4431_dc_notnorm_I_PI_corr_tel+unres.fits'
        pdi_i= obp.Loadimage(pdi_folder_i,pdi_file_i)
        obp.plot_polarimetric_image(pdi_i, 3.6, title='IRAS08544-4431 I-band PI', save=work_root+'/pi_i_band.png', image_scale='asinh', roi_half_size=50)

        pdi_folder_h = '/Users/katerynaandrych/Work/lin/Postdoc/Data/polarimetry/IRAS08544-4431_for_modelling/H_band/'
        pdi_file_h = 'iras08544-4431_calib_H_PI_corr_tel+unres.fits'
        pdi_h= obp.Loadimage(pdi_folder_h,pdi_file_h)
        obp.plot_polarimetric_image(pdi_h, 12.27, title='IRAS08544-4431 H-band PI', save=work_root+'/pi_h_band.png', image_scale='asinh', roi_half_size=30)

        file_psf='iras08544-4431_calib_H_I_meancombined.fits'
        psf_h=obp.Loadimage(pdi_folder_h,file_psf)
        obp.plot_polarimetric_image(psf_h, 12.27, title='IRAS08544-4431 H-band PSF', save=work_root+'/psf_h_band.png', image_scale='asinh', roi_half_size=30)

        
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
    par_path = obm.write_mcfost_paramfile(cfg, fidelity, trial_dir)
    try:
        obm.run_mcfost(fidelity,par_path, trial_dir)
    except Exception:
        # Trial failed; return a high loss
        return 1e99

    # Score outputs

    loss = obm.load_and_score_outputs(fidelity, trial_dir, data_arg)

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

    data_arg = load_data(args.data_root, str(work_root))
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
    # Verify template parameter file for mcfost exists
    template_para= Path(results_dir.parent/"simulation.para")
    assert template_para.exists(), f"Missing template .para at {template_para}"

    # Write param file and run MCFOST
    par_path = obm.write_mcfost_paramfile(incumbent, fidelity_result, results_dir)
    obm.run_mcfost(fidelity_result,par_path, results_dir)
    # Score outputs
    loss = obm.load_and_score_outputs(fidelity_result, results_dir, data_arg)




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
