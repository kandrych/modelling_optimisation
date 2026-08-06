
import pandas as pd

import argparse
import json
import math
import os
import matplotlib

import shutil
import subprocess
import sys
import textwrap
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple

# --- SMAC3 / ConfigSpace ---
from smac import Scenario
from smac.facade.multi_fidelity_facade import MultiFidelityFacade
from smac.runhistory import TrialValue
from smac import HyperparameterOptimizationFacade as HPO  # for callbacks, utils

from ConfigSpace import ConfigurationSpace, Float, Integer, Categorical, Configuration
from ConfigSpace.conditions import InCondition, EqualsCondition

from smac.runhistory.dataclasses import TrialInfo, TrialValue




import distroi

sys.path.append(os.path.abspath(".."))  # parent of current working dir


def runhistory_to_df(smac, cs) -> pd.DataFrame:
    rh = smac.runhistory
    rows = []
    for k, v in rh.items():
        # k has: config_id, seed, budget, instance
        cfg = rh.get_config(k.config_id)  # Configuration object

        row = {}
        row.update(cfg.get_dictionary())
        row["config_id"] = k.config_id
        row["seed"] = k.seed
        row["budget"] = k.budget
        row["cost"] = v.cost
        row["time"] = getattr(v, "time", None)
        row["status"] = getattr(v, "status", None)
        row["additional_info"] = v.additional_info if isinstance(v, TrialValue) else None
        
        rows.append(row)

    df = pd.DataFrame(rows)
    # handy sorting
    df = df.sort_values(["budget", "cost"], ascending=[True, True]).reset_index(drop=True)
    return df
