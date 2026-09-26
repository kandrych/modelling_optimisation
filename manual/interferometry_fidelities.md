# SED and interferometry fidelity stages

Use `config/fidelity_sed_interferometry.txt` as the run's `fidelity.txt`, or pass
its path with `--fidelity-config`. It defines:

1. SED (budget 1).
2. SED + one wavelength window per interferometric band (budget 3).
3. SED + chromatic interferometry (budget 9).

SMAC promotes selected configurations between stages. The budget numbers select
stages; they do not multiply the objective terms.

Observations are loaded independently for each requested interferometry mode.
Scoring, optional overresolved-flux fitting, and `--plot-intermediate` use the
observations for the current trial's stage. The final incumbent uses the last
stage's observations.

| Instrument | One-per-band selection (µm) | Chromatic selection (µm) |
| --- | --- | --- |
| PIONIER | 1.63–1.64 | No explicit wavelength cut |
| GRAVITY | 2.199–2.201 | No explicit wavelength cut |
| MATISSE L | 3.48–3.52 | 2.95–3.95 |
| MATISSE N | 9.9–10.10 | 8–13 |

Existing FITS file patterns and visibility filters are retained. MATISSE N uses
correlated flux; the other instruments use squared visibility. Chromatic means
the selected observational channels, not every generated model image wavelength.

The fix preserves cost formulas, uncertainties, and explicit weights. For mixed
stages, the corrected chromatic cost now includes observations previously lost
through narrow-band loading, so its magnitude and balance with SED can change.
Start this setup in a new working root. Do not import old mixed-mode chromatic
results that were scored against narrow-band observations; the warmstart
metadata check cannot detect that historical data-selection error.

For Python callers, `load_data()` now returns a named dictionary with keys
`sed`, `interferometry`, `pdi_V`, `pdi_I`, `pdi_H`, and `alma`.
Access observations as `data["interferometry"][mode][instrument]`. The other
observable payloads are unchanged; for example, replace `data[5]` with
`data["pdi_V"]` and `data[8]` with `data["alma"]`.
