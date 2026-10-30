#!/bin/bash
# Shared environment for the supported Slurm examples. Source this file after
# setting n_workers. Each worker launches one MCFOST process with OpenMP threads.
# Keep the Python worker count consistent with the thread allocation below.
for argument in "$@"; do
    case "$argument" in
        --n-workers|--n-workers=*)
            echo "Set N_WORKERS before submission instead of passing --n-workers." >&2
            exit 1
            ;;
    esac
done
allocated_cpus=${SLURM_CPUS_PER_TASK:-1}
if [[ ! "$n_workers" =~ ^[1-9][0-9]*$ ]] || (( n_workers > allocated_cpus )); then
    echo "n_workers must be between 1 and the allocated CPUs ($allocated_cpus)." >&2
    exit 1
fi
export OMP_NUM_THREADS=$((allocated_cpus / n_workers))
export OMP_STACKSIZE=1024m
export OMP_SCHEDULE=dynamic
# NumPy/BLAS should not start another full set of threads inside each worker.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Working directory: $PWD"
echo "Workers: $n_workers; MCFOST threads per worker: $OMP_NUM_THREADS"

if ! type module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/z00_lmod.sh ]; then
        source /etc/profile.d/z00_lmod.sh
    elif [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
        source /usr/share/Modules/init/bash
    fi
fi
module --force purge
module load python-scientific/3.11.5-foss-2023b
source /home/kandrych/venvs/obriy311/bin/activate
