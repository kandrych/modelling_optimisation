# Supported launch examples

Submit these jobs from your run directory. Each job uses SMAC workers or one
MCFOST process inside the allocation granted by Slurm; it does not submit
additional worker jobs. Review the `#SBATCH` time, memory and CPU requests for
your run before submitting.

| Script | Purpose |
| --- | --- |
| `run_smac.q` | Optimisation using `SMAC3.py` |
| `run_smac_warmstart.q` | Optimisation with a supplied warm-start run and puffed-rim options |
| `run_model.q` | Evaluate one supplied `.para` through `run_model.py` |
| `score_model.q` | Score existing MCFOST outputs and regenerate plots through `score_model.py` |

`setup_environment.sh` is sourced by these scripts. It sets up the Python
environment and assigns `allocated CPUs / workers` OpenMP threads to each
MCFOST process, rounded down. For example, 21 CPUs and 10 workers give two
threads per worker. BLAS threads are set to one to avoid nested parallelism.
Change `N_WORKERS` to choose a different balance; the launchers reject a later
`--n-workers` argument so it cannot invalidate the thread allocation. A single-model job uses all
allocated CPUs for that model.

The default project path is `/fred/oz061/kandrych/modelling_optimisation`.
Set `PROJECT_ROOT` if the repository is elsewhere. `WORK_ROOT` defaults to
the directory from which the job was submitted.

## Optimisation

The working directory must contain `simulation.para`, a ConfigSpace YAML and
an observable-stage file. The default names are `configuration.yaml` and
`fidelity.txt`; set `CONFIG_SPACE` and `FIDELITY_CONFIG` for other paths.
Use absolute paths when setting these variables.

```bash
cd /path/to/run
CONFIG_SPACE="$PWD/configuration_tapered.yaml" N_WORKERS=5 \
  sbatch /fred/oz061/kandrych/modelling_optimisation/run_scripts/run_smac.q
```

Budgets come from the stages in `fidelity.txt`. The removed `--min-budget`
and `--max-budget` options must not be supplied. See
[fidelity configuration](../config/fidelity_usage.md) for the file format.
Other arguments after the script name are passed to `SMAC3.py` unchanged.
The examples request 1000 trials; a later `--n-trials` argument can override it.

For a warm start, set `WARMSTART` to the previous SMAC run directory. That
example retains `--puffed-up-rim`, so its ConfigSpace must include the required
rim parameters. Review observation, template and scoring compatibility before
reusing previous results; see [warm starts](../manual/warmstart_and_continuation.md).

## One supplied model

For a model that has already been calculated, submit the scoring-only job:

```bash
cd /path/to/run
sbatch /fred/oz061/kandrych/modelling_optimisation/run_scripts/score_model.q
```

This reads `smac_results/model.para`, the existing model outputs and `fidelity.txt`
from the submission directory. It uses the last fidelity stage and enables plots.
It writes `smac_results/scoring.json` and comparison figures without running MCFOST.
Override `MODEL_DIR`, `FIDELITY_CONFIG`, `DATA_ROOT` or `WORK_ROOT` as needed.
Additional arguments go directly to `score_model.py`, for example:

```bash
MODEL_DIR=/path/to/another/model \
  sbatch /fred/oz061/kandrych/modelling_optimisation/run_scripts/score_model.q \
  --stage NAME --correct-unresolved-polarimetry --unresolved-correction-radius-px 3
```

Choose the stage and correction settings used by the original run. The job
requests one CPU, 32 GB of memory and one hour; adjust these for your image sizes.

`run_model.q` reads `model.para` and `fidelity.txt` from the working directory
and writes to a new `model_results` directory. Override these with `PARA_FILE`,
`FIDELITY_CONFIG` and `OUTPUT_DIR`. `DATA_ROOT` selects the observation preset
and defaults to `demo_ozstar`. Extra arguments are passed to `run_model.py`.

The last fidelity stage is used unless `--stage NAME` is supplied. Choose
`vis2_chromatic` in that stage for chromatic interferometry; the script name
does not select observables. See [single-model evaluation](../docs/run_model.md)
for physical options, including unresolved polarisation and puffed-rim settings.

Two older filenames remain as small launch wrappers:

- `run_smac_chromatic_1_simulation.q` uses `run_model.q` and retains the original
  unresolved-correction request. Set `UNRESOLVED_CORRECTION_RADIUS_PX` to the
  aperture used for your observations before submission.
- `scripts/run_smac.q` uses `run_model.q` with `ar_pup_ozstar` as its default
  observation preset. Its existing Slurm resource requests still apply.

Neither wrapper executes the old scripts under `scripts/testing/`.
