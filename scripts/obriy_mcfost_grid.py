
from typing import Literal, Tuple, Dict, Optional, Union, Any, List

import os
import matplotlib.pyplot as plt

from IPython.display import display
import glob
import subprocess


import json

from textwrap import wrap

import shutil, subprocess
from pathlib import Path

from ConfigSpace import ConfigurationSpace
from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical
from typing import Dict, Any, Iterable, Optional
from ConfigSpace.conditions import InCondition, EqualsCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace import Configuration
from itertools import product

import re, time






#constants.set_matplotlib_params()  # set project matplotlib parameters
os.environ.setdefault("MCFOST_NO_UPDATE", "1") # prevent MCFOST from checking for updates every time it is run within this script


# If mcfost is not working from script, Ensure MCFOST is found in PATH and MCFOST_UTILS is set. 
#You can modify and uncomment following 2 lines as needed.
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



def write_mcfost_paramfile(cfg: Dict[str, Any], outdir: Path) -> Path:
    """
    Materialize an MCFOST parameter file in `outdir` from the sampled configuration.
    Returns the path to the written .para file.
    """
    print(f"write mcfost param file in {outdir}")

    outdir.mkdir(parents=True, exist_ok=True)
    param_path = outdir / "model.para"

    # Json dump of config + fidelity for record-keeping
    with open(outdir / "config_used.json", "w") as f:
        json.dump({"cfg": cfg}, f, indent=2)
    

    # Load a base .para file template from folder that was passed as working root
    pf = ParaFile(str(outdir.parent/"simulation.para"))
    for key in cfg.keys():
        if key in pf.params:
            pf.set_param(key, cfg[key])
    # Set fidelity-related params
    # pf.set_param("nbr_photons_eq_th", fidelity["nbr_photons_eq_th"])
    # pf.set_param("nbr_photons_lambda", fidelity["nbr_photons_lambda"])
    # pf.set_param("nbr_photons_image", fidelity["nbr_photons_image"])

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

#########################################
# End of MCFOST
##########################################



def build_configspace(config_file: str) -> ConfigurationSpace:

    #cs = ConfigurationSpace.from_yaml('/fred/oz061/kandrych/modelling_optimisation/config/space.yaml')#("/Users/katerynaandrych/Work/lin/python scripts/modelling_optimisation/config/space.yaml")
    cs = ConfigurationSpace.from_yaml(config_file)

    return cs

def categorical_grid(
    cs: ConfigurationSpace,
    *,
    as_dict: bool = True,
    respect_forbiddens: bool = True,
) -> Iterable[Dict[str, Any] | ConfigurationSpace]:
    """
    Enumerate all combinations of hyperparameters declared in the YAML,
    assuming those are exactly the *non-fixed* variables.

    Only Categorical and Ordinal are supported (explicit grids).
    Conditions/forbiddens in the YAML are respected.

    Yields dictionaries by default; set as_dict=False to yield ConfigSpace.Configuration.
    """
    hp_defs = []
    for hp in list(cs.values()):
        if isinstance(hp, CategoricalHyperparameter):
            hp_defs.append((hp.name, tuple(hp.choices)))
        elif isinstance(hp, OrdinalHyperparameter):
            hp_defs.append((hp.name, tuple(hp.sequence)))
        else:
            raise TypeError(
                f"Unsupported hyperparameter '{hp.name}' of type {type(hp).__name__}. "
                "This grid assumes only categorical/ordinal variables are present in the YAML."
            )

    # If YAML is empty (no variables), yield a single empty config
    if not hp_defs:
        yield {} if as_dict else Configuration(cs, values={})
        return

    names = [n for n, _ in hp_defs]
    choices = [c for _, c in hp_defs]

    for vals in product(*choices):
        cfg_dict = dict(zip(names, vals))
        # Validate against conditions/forbiddens
        try:
            cfg = Configuration(cs, values=cfg_dict)
        except Exception:
            continue
        
        yield cfg_dict if as_dict else cfg

def dict_from_cfg(cfg: Any) -> Dict[str, Any]:
    if isinstance(cfg, Configuration):
        return dict(cfg)
    if hasattr(cfg, "get_dictionary"):
        return dict(cfg)
    return dict(cfg)
    


def slug(s: str) -> str:
    """Filesystem-safe slug: letters, digits, _ and - only."""
    s = str(s)
    s = s.replace("/", "-").replace("\\", "-")
    s = re.sub(r"[^\w\-.]+", "_", s).strip("_")
    return s or "x"

def fmt_val(v: Any, float_digits: int = 3, trim_zeros: bool = True) -> str:
    """Human-friendly value formatting for folder names."""
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int,)):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.{float_digits}g}"  # short, scientific if needed
        if trim_zeros:
            s = re.sub(r"(\d)0+($|e)", r"\1\2", s)
            s = re.sub(r"\.(?=e|$)", "", s)  # drop trailing dot before 'e' or EOL
        return s
    return slug(v)

def folder_from_cfg(
    cfg: Dict[str, Any],
    *,
    include: Optional[Iterable[str]] = None,   # which keys to include (None -> all)
    aliases: Optional[Dict[str, str]] = None,  # rename keys for brevity
    max_pairs: Optional[int] = None,           # cap number of key=val pairs
    sep: str = "_",
    kv_sep: str = "",
    float_digits: int = 3,
    ) -> str:
    """
    Build a deterministic, compact, readable folder name from parameter dict.
    Example: 'Rc48.3_H1.73_gtd50_mass1e-3'
    """
    aliases = aliases or {}
    items = sorted(cfg.items(), key=lambda kv: kv[0])  # stable ordering

    # Optionally filter to a subset (e.g., only categoricals or key params)
    if include is not None:
        include = set(include)
        items = [kv for kv in items if kv[0] in include]

    
    toks = []
    for k, v in items[: max_pairs or len(items)]:
        k2 = aliases.get(k, k)
        k2 = slug(k2)
        v2 = fmt_val(v, float_digits=float_digits)
        toks.append(f"{k2}{kv_sep}{v2}")

    name = sep.join(toks) if toks else "cfg"
   
    return name or "cfg"


def make_unique_trial_dir(scratch_root: Path, folder: str) -> Path:
    """
    Create a unique trial directory under scratch_root/folder by appending
    an index if needed.
    Returns the Path to the created directory.
    """
    
    base = Path(scratch_root) / f"{folder}"
    trial_dir = base
    idx = 0
    while True:
        try:
            trial_dir.mkdir(parents=True, exist_ok=False)  # atomic: raises if exists
            return trial_dir
        except FileExistsError:
            idx += 1
            trial_dir = base.with_name(f"{base.name}_{idx:03d}")




def main():
    grid_path='/fred/oz061/kandrych/obriy_grid/' #your path to where configuration yaml file is located and where to run all simulations
    fidelity={"stage":"F2"}  #choose from "F1", "F2", "F3" based on needed fidelity level if you use run_mcfost function as is. You can modify fidelity dictionary as needed if you modify run_mcfost function.
            
    WORK_ROOT = Path(grid_path).resolve()

    print(f"[main] Working root: {WORK_ROOT}")


    config_candidates = list(WORK_ROOT.rglob("*.yaml")) + list(WORK_ROOT.rglob("*.yml"))
    if not config_candidates:
        raise FileNotFoundError(f"No *.yaml found under {WORK_ROOT}")
    config_file = sorted(config_candidates)[0]

    # Verify template parameter file for mcfost exists
    template_para= Path(WORK_ROOT/"simulation.para") #make sure you have a template mcfost simulation.para file in your working root. It will be used as a base to create new para files for each model run.
    assert template_para.exists(), f"Missing template .para at {template_para}"

    cs = build_configspace(config_file)
   
    # serial or light parallel (avoid big parallelism unless you tuned RAM!)
    for cfg in categorical_grid(cs, as_dict=True):
        # cfg contains only the variable keys from YAML.
        # Merge with your external fixed params when writing model.para:
        # full_cfg = {**fixed_params, **cfg}
        folder = folder_from_cfg(
                cfg,
                float_digits=3,          # tweak precision for floats
                sep="_",
                kv_sep="",               # yields 'Rc48.3' instead of 'Rc=48.3'
                )
        trial_dir = make_unique_trial_dir(WORK_ROOT, folder)
        # Write param file and run MCFOST
        par_path = write_mcfost_paramfile(cfg, trial_dir)
        try:
            #You can use this function and modify it as needed to run mcfost with different wavelengths based on fidelity stage. 
            run_mcfost(fidelity,par_path, trial_dir) 
            #Or you can use run_mcfost_safe(param_path, workdir, options=[], logfile="mcfost_base.log") for tempereture structure and run_mcfost_image(wavelength, folder) for any wavelength you want.
            
        except Exception:
            # Trial failed; return a high loss
            print(f"[objective] Trial failed for cfg={cfg}, dir={trial_dir}")
            continue
    
    
    pass


if __name__ == "__main__":
    main()
