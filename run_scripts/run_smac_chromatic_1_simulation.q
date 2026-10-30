#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=mcfost_chromatic_model
#SBATCH --cpus-per-task=64
#SBATCH --mail-user=kateryna.andrych@mq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --time=0-2:00:00
#SBATCH --mem=50G
#SBATCH --output=driver.%j.out
#SBATCH --error=driver.%j.err

set -euo pipefail

# Compatibility filename. Use the supported single-model driver and select
# vis2_chromatic in fidelity.txt. Keep the original correction request, but
# require its aperture explicitly rather than assuming an observational value.
project_root=${PROJECT_ROOT:-/fred/oz061/kandrych/modelling_optimisation}
correction_radius=${UNRESOLVED_CORRECTION_RADIUS_PX:?Set the correction radius used for the observations.}
exec bash "$project_root/run_scripts/run_model.q" \
  --correct-unresolved-polarimetry \
  --unresolved-correction-radius-px "$correction_radius" \
  "$@"
