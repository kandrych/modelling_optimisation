# Observable stages

Copy the repository's `fidelity.txt` to your run directory. This text file uses
YAML syntax, but is read independently of the ConfigSpace YAML.

```bash
python SMAC3.py --data-root demo_ozstar --working-root /path/to/run/ \
  --config-space /path/to/run/parameters.yaml \
  --fidelity-config /path/to/run/fidelity.txt \
  --n-trials 1000 --n-workers 10
```

Add the existing geometry, dust-sharing and plotting flags as needed.
Remove `--min-budget` and `--max-budget` from launch scripts: supplying them
now produces an explanatory error. File order defines stages; default eta=3
assigns budgets 1, 3, 9, etc. Optional top-level `eta: 2` assigns 1, 2, 4, etc.
A single stage uses min_budget=max_budget=1. Stage names must be unique.

`--fidelity-config` defaults to `fidelity.txt` directly in the working root.
If `--config-space` is omitted, exactly one YAML configuration must be found
under that root; ambiguous selection fails instead of choosing the first file.
An explicitly selected fidelity file is excluded from ConfigSpace discovery.

Stages accept sed, pdi_V, pdi_I, pdi_H, alma, vis2_1perband or vis2_chromatic.
Choose only one interferometry mode per stage. Data loading uses the union of
all stages, even when the last stage does not include all earlier products.
Observable scoring formulas and weights are unchanged. Image resolution
metadata is 2, matching the prior joint stages; the parameter writer still
does not change image resolution or photon counts based on budget.

Hyperband still chooses promotions and can start some trials directly at a
higher stage. The final model uses the last stage. Stage names, products and
budgets are logged and stored in trial additional_info and config_used.json.
The resolved schedule is saved as fidelity_resolved.json; a changed schedule
requires a new working root so automatic resume cannot mix stage meanings.

Warm starts skip trials without compatible per-trial fidelity metadata,
including histories from the old numeric stage mapping. Matching metadata
does not establish compatibility of observations, physical templates or cost
code; those must still match. Previous map_budget_to_fidelity code remains
commented in SMAC3.py for reference.
