#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=1all_unres
#SBATCH --cpus-per-task=60
#SBATCH --mail-user=kateryna.andrych@mq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END
#SBATCH --time=0-15:00:00
#SBATCH --mem=80G
#SBATCH --output=driver.%j.out
#SBATCH --error=driver.%j.err

set -euo pipefail
echo "HOSTNAME = $HOSTNAME"
echo "HOSTTYPE = $HOSTTYPE"
echo Time is `date`
echo Directory is `pwd`

export OMP_NUM_THREADS=12
export OMP_STACKSIZE=1024m
export OMP_SCHEDULE=dynamic


# --- Make 'module' available on compute nodes (portable across sites) ---
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
elif [ -f /usr/share/Modules/init/bash ]; then
  source /usr/share/Modules/init/bash
fi

module --force purge
module load python-scientific/3.11.5-foss-2023b


source /home/kandrych/venvs/obriy311/bin/activate


base_path=`pwd`

# Run your script with --use-slurm
python /fred/oz061/kandrych/modelling_optimisation/SMAC3.py \
  --data-root demo_ozstar \
  --working-root "$base_path"/ \
  --n-trials 1000 --n-workers 5 \
  --seed 358239 \
  --2ZONE_CONT_LHC --share-zone-composition \
  --plot-intermediate --tapered-edge-p1-eq-p2 \
  --correct-unresolved-polarimetry --unresolved-correction-radius-px 3 \
  --initial-design-seed 101 \
  --trial-walltime-limit 3600 --timeout-cost none
  