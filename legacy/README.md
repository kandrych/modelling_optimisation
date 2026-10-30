# Historical and experimental code

These modules are outside the supported optimisation path. Nothing in
`SMAC3.py`, `run_model.py` or the active scoring code imports them.
Importing them does not start workers or submit jobs.

## Archived cluster launcher

`cluster_setup.py` contains the former `SMAC3.start_cluster()` implementation.
Its site-specific settings are retained for reference. Its imports are now
self-contained, including `socket`, but its job-launch behaviour has not been
validated. Calling it can submit Slurm jobs; do not treat it as a supported
alternative to the scripts in `run_scripts/`.

Supported optimisation lets SMAC manage local workers within the existing
allocation. It does not require a direct `dask_jobqueue` import in `SMAC3.py`.

## Historical ALMA diagnostics

The following implementations moved from `lib.obriy_alma` without changing
their numerical definitions:

- `rescale_alma`: shape-based image resampling.
- `cut_down_alma`: central cropping based on array dimensions.
- `chi2_ALMA`: the former normalised-profile comparison.

For old analyses that specifically need these definitions, import them explicitly:

```python
from legacy import alma_diagnostics as legacy_alma
```

`scripts/testing/checking_whether_file_ok.ipynb` uses this explicit import for
its historical profile check. Other external notebooks must update their old
`obriy_alma` imports before calling these functions.

The supported ALMA path remains in `lib.obriy_alma`:
`load_mcfost_image_alma_casa` → `convolve_alma_to_jybeam` →
`align_alma_to_observation` → `compare_alma_images`.
The profile diagnostic and the active image loss are different quantities;
neither is silently substituted for the other. The shared data loader no
longer prepares legacy ALMA profiles; historical callers must supply them.

## Other historical files

`lib/Katya_func.py` already identifies itself as unmaintained. It remains in
its original location in this cleanup; supported callers use the separate
`lib/obriy_*` modules. Older scripts under `scripts/testing/` are exploratory
analyses, not supported launch entry points. Use `run_model.py` for evaluating
a supplied model.
