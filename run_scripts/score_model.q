#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=score_model
#SBATCH --cpus-per-task=1
#SBATCH --time=0-1:00:00
#SBATCH --mem=32G
#SBATCH --output=scoring.%j.out
#SBATCH --error=scoring.%j.err

set -euo pipefail

# Submit from the optimisation run folder containing fidelity.txt and smac_results/.
# Override these paths with environment variables or arguments after the script name.
project_root=${PROJECT_ROOT:-/fred/oz061/kandrych/modelling_optimisation}
work_root=${WORK_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$work_root"
model_dir=${MODEL_DIR:-$PWD/smac_results}
fidelity_config=${FIDELITY_CONFIG:-$PWD/fidelity.txt}
data_root=${DATA_ROOT:-demo_ozstar}

# Reuse the normal Python environment. Scoring runs in one process without MCFOST.
n_workers=1
source "$project_root/run_scripts/setup_environment.sh"
export MPLBACKEND=Agg

# Plots are enabled by default. Pass the original run's correction options if needed.
# "$@" forwards additional options, for example --stage NAME or --no-plot-intermediate.
exec python -u "$project_root/score_model.py" \
  --model-dir "$model_dir" \
  --fidelity-config "$fidelity_config" \
  --data-root "$data_root" \
  "$@"
