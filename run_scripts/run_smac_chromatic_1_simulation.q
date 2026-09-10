#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=feature_test_smac
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
echo "HOSTNAME = $HOSTNAME"
echo "HOSTTYPE = $HOSTTYPE"
echo Time is `date`
echo Directory is `pwd`

export OMP_NUM_THREADS=64
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


python /fred/oz061/kandrych/modelling_optimisation/scripts/mcfost_chromatic_simulation_for_1_model.py --correct-unresolved-polarimetry