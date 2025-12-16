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
from pathlib import Path
from typing import Any, Dict, Tuple


sys.path.append(os.path.abspath(".."))  # parent of current working dir
#import lib.Katya_func as kf

import lib.obriy_general as obg
import lib.obriy_sed as obs
import lib.obriy_interferometry as obi
import lib.obriy_polarimetry as obp
import lib.obriy_mcfost as obm
import SMAC3 as obriy_smac


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "DejaVu Serif",         # shipped with matplotlib
    "Liberation Serif",
    "Nimbus Roman",
    "TeX Gyre Termes",
    "Times",                # generic fallback
]


p = argparse.ArgumentParser(description="SMAC3 + Dask(SLURM) + Multi-fidelity for MCFOST")
p.add_argument("--seed", type=int, default=-1)# Random seed for SMAC
p.add_argument("--correct-unresolved-polarimetry", action="store_true", help="Apply correction for unresolved central source polarimetry")

args = p.parse_args()
work_root=f'/fred/oz061/kandrych/smac/polarimetry_sed/full_chromatic_for_best/' #distance_sublimationF #distance660  #original
fidelity={}
fidelity["stage"]='F3'


WORK_ROOT = Path(work_root).resolve()
os.environ["SMAC_WORK_ROOT"] = str(WORK_ROOT)
print(f"[main] Working root: {WORK_ROOT}")


results_dir = Path(work_root)
results_dir.mkdir(exist_ok=True)
obm.run_mcfost(fidelity, WORK_ROOT/'model.para', results_dir)

# Score outputs

data_arg = obriy_smac.load_data('demo_ozstar', str(work_root),fidelity["stage"])

loss = obm.load_and_score_outputs(fidelity, results_dir, data_arg, args)

