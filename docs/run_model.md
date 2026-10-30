# Evaluate one existing MCFOST model

To score a model whose MCFOST calculation has **already finished**, use
`score_model.py` (the script equivalent of `scripts/score_model.ipynb`):

```bash
python /fred/oz061/kandrych/modelling_optimisation/score_model.py \
  --model-dir /path/to/run/smac_results \
  --fidelity-config /path/to/run/fidelity.txt \
  --data-root demo_ozstar
```

The model folder must contain its generated `model.para` and the `data_*`
outputs required by the selected stage. This script does not run MCFOST.
Both paths above are relative to the current directory unless absolute.
The last fidelity stage is selected by default; use `--stage NAME` to select
an earlier stage for a trial that generated fewer observables.

Scores are saved to `scoring.json` in the model folder, including the total
cost, individual contributions, settings and success/failure status. Repeated
scoring replaces this report and regenerates comparison plots in their usual
locations. The original `evaluation.json` is preserved. Plots are enabled by
default; use `--no-plot-intermediate` to disable model diagnostic plots.
The shared observation loader still writes `data_figures/`.

Use the same observation preset and comparison options as the original run:
`--correct-unresolved-polarimetry`, `--unresolved-correction-radius-px`,
`--pdi-radial-bin-mas` and `--overresolved_flux_fit_for_interferometry`.
The scoring code and weights are shared with SMAC3. Simulation flags such as
`--puffed-up-rim` are unnecessary when reading already generated outputs.

## Run MCFOST before scoring

`run_model.py` reads a supplied `.para` and `fidelity.txt`, runs MCFOST, and
compares the selected products with observations using `SMAC3.load_data` and
`obriy_mcfost.load_and_score_outputs`. It does not launch an optimiser or need
a ConfigSpace, seed, or runhistory. Use the same Python/module environment and
MCFOST installation as for `SMAC3.py`.

For the observation settings in the example optimisation job:

```bash
python /fred/oz061/kandrych/modelling_optimisation/run_model.py \
  --working-root /path/to/model_folder \
  --para-file model.para \
  --data-root demo_ozstar \
  --correct-unresolved-polarimetry \
  --unresolved-correction-radius-px 3
```

The working folder must contain the selected `.para` and `fidelity.txt` in the
same named-stage format used by `SMAC3.py`. Omit `--para-file` if exactly one
`.para` exists directly in the folder. Relative file and output paths are
resolved against `--working-root`; absolute paths are also accepted.
Use `--fidelity-config /path/to/fidelity.txt` to select another fidelity file.

The last stage is selected by default, matching final-model generation in
`SMAC3.py`. Use `--stage NAME` for a different stage. Only the selected stage's
products are loaded and scored; stages are not automatically combined.
Fidelity selects observables, not photon counts or the supplied physical model.

The script copies the `.para` unchanged to `model_results/model.para`. Local
relative wavelength, optical-index and stellar-spectrum files referenced by
the parameter file are copied too. Local references containing `..` must be
made absolute in the input file first. Files supplied by MCFOST's normal utility
lookup remain subject to that lookup. As in the optimiser, ALMA generates a
separate `model_alma.para` with image geometry matched to the observations.

Outputs are saved under `model_results/`:

- `evaluation.json`: total `cost`, per-observable `additional_info`, selected
  fidelity, options, runtime and success/failure status. Non-finite diagnostic
  values are written as JSON `null`; a non-finite total cost is an error.
- `figures/`: model comparison and diagnostic plots, enabled by default.
- `data_figures/`: observation plots from the shared data loader.
- MCFOST data directories and command logs.

An existing output directory is rejected to avoid scoring stale model images.
For another evaluation, choose a fresh directory with `--output-dir check2`.
Failures return a nonzero exit code; failures after output creation also leave
an `evaluation.json` error record and any available MCFOST logs.

Use the same comparison options as the original optimisation. Unresolved
polarimetry correction is off unless explicitly requested. The radial bin size
defaults to 25 mas. The optional
`--overresolved_flux_fit_for_interferometry WAVELENGTH` has the same meaning and
micron units as in `SMAC3.py`. `--no-plot-intermediate` disables model diagnostic
plots, though the shared observation loader still produces its own plots.
Scoring formulas, uncertainties and weights are unchanged.

Geometry/composition flags such as `--2ZONE_CONT_LHC`,
`--share-zone-composition` and `--tapered-edge-p1-eq-p2` are unnecessary here:
their effects must already be represented in the supplied `.para`. An original
template `simulation.para` evaluates that template's values; it does not recover
an optimiser's best configuration. To evaluate a saved best model, supply its
generated `model.para`.

Puffed-rim settings are passed to MCFOST separately and cannot be recovered
from `.para` alone. If the model needs them, specify
`--puffed-up-rim H_OVER_R R_AU DELTA_R` with the original MCFOST arguments
(`puffed_h_rim_over_h0`, `puffed_r_rim`, `puffed_delta_r`).

Run this command inside a compute allocation. The existing Slurm module/venv
setup can be reused; replace its `python SMAC3.py ...` command with the command
above. Set `OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"` to match the allocation.
This driver runs one model and inherits the environment; it does not impose
the optimiser's per-trial timeout. The Slurm job time limit still applies.
