#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=mcfost_single_model
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=kateryna.andrych@mq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --time=0-0:30:00
#SBATCH --mem=10G
#SBATCH --output=driver.%j.out
#SBATCH --error=driver.%j.err

set -euo pipefail

# Compatibility filename for the old AR Pup model check. The batch resources
# above still apply; the shared launcher does not submit another Slurm job.
project_root=${PROJECT_ROOT:-/fred/oz061/kandrych/modelling_optimisation}
export DATA_ROOT=${DATA_ROOT:-ar_pup_ozstar}
exec bash "$project_root/run_scripts/run_model.q" "$@"
