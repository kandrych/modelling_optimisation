# Draft status and evidence

The LaTeX draft assumes IRAS 08544-4431 from the supplied paper, stellar
template and previous parameter discussion. The user confirmed two zones,
an exponentially tapered second zone, observable order SED → interferometry
→ polarimetry → ALMA, and uncertainty work in progress using repeated runs
and corner plots. Four stages are assumed in the prose; confirm whether
polarimetry and ALMA instead enter together in one stage.

The current repository fidelity.txt still uses SED → SED+PDI → joint.
It was not changed as part of writing the manuscript. Match the actual
production schedule to the description before submitting the paper.

## Corporaal et al. (2023): locations in the supplied PDF

- Section 3, p. 3: MCMax3D; two-zone surface-density construction; shared
  physical properties; scale-height and grain-size prescriptions.
- Section 4.1, pp. 3–4; Table 1: central source, distance, companion
  post-processing, reference dust opacity and parameter exploration.
- Table 2, p. 9: grid refined after the parameter study.
- Section 5.1, pp. 9–10: staged photometric/interferometric screening,
  SED selection restricted to 20 microns, thirteen acceptance criteria.
- Sections 5.2 and 6: family of ten retained models; residual over-resolved
  emission and band-dependent tensions. Do not describe this as a posterior.
- Abstract/conclusions: motivation for combining VLTI, SPHERE and ALMA.

Source: the supplied local Corporaal_2023.pdf, extracted directly with PDFKit.
Published article: https://doi.org/10.1051/0004-6361/202245689

## Code supporting the draft

| Topic | Source |
|---|---|
| Surface-density mass continuity and tapered integral | lib/obriy_mcfost.py: `_2zone_cont_calc_dmass` |
| Shared populations and one-way height tie | lib/obriy_mcfost.py: `ParaFile.save`, `write_mcfost_paramfile` |
| Stage file, geometric budgets, product validation | lib/obriy_fidelity.py |
| Optimisation and Hyperband setup | SMAC3.py: `main`, `objective` |
| SED N−1 normalisation and bounded reddening | lib/obriy_sed.py: `chi2reddened`, `fit_sed_reddening` |
| 4000 K, 3.9% companion contribution | lib/obriy_sed.py: `chi2_SED_with_reddening`, `add_blackbody_component` |
| Interferometric N−1 normalisation | lib/obriy_interferometry.py: `oi_container_chi2` |
| PDI measurements and empirical weights | lib/obriy_polarimetry.py: `measure_pdi_constraints`, `pdi_constraint_loss`, `compare_pdi_constraints` |
| ALMA residual energy | lib/obriy_alma.py: `compare_alma_images` |
| Final summed score | lib/obriy_mcfost.py: `load_and_score_outputs` |
| Corner plot selection and inverse-cost weighting | supplied Downloads/after_optimisation.ipynb, cells 20, 22–23 |

Important implementation details:

- PDI weights are 1e5 (fraction), 1e4 (mean radial residual), 5e3 (sum of
  six differential-sector residuals). The function returns their **sum**.
  Its stale `loss_definition` string mentions a mean; the draft follows the
  executable calculation, not that metadata string.
- The observed positive-Qphi fraction is clipped in its numerator; radial
  profiles retain signed Qphi and are divided by the aperture's signed sum.
- ALMA uses 100 times residual energy divided by observed energy, not an
  uncertainty-normalised pixel chi-square. Correlated beam noise matters
  for any future statistical interpretation.
- The Python seed configures SMAC; the current subprocess wrapper does not
  explicitly pass it to MCFOST. Specify an MCFOST seed policy for repeat tests.
- `deterministic=False` does not itself implement repeated-model experiments.
- SED wavelength selection must be obtained from the actual data file and
  loader; do not inherit Corporaal's 20-micron cutoff by assumption.
- The supplied template enables viscous heating and dust radial migration.
  Verify these choices and the adopted settling prescription; do not call
  the model purely passive without checking the production template.
- The height tie only supports zone 2 following zone 1. Equal reference
  radii and flaring indices are separately needed for shared H(r).

## Parameter comparison to finalise

Values below for the proposed work are from the user's pasted configuration,
not verified final-run settings. They should not be presented as final priors
without checking the actual run YAML and effective parameter files.

| Quantity | Corporaal reference/exploration | User's recent proposed setup |
|---|---|---|
| Outer radius | 175 au fixed (Table 1) | 300 au |
| Inner radius | 7.2 au | 7.2 au |
| Transition radius | 2.5–3.5 Rin explored | 18 au = 2.5 Rin |
| Inner density slope | p_in = −1, −1.5, −2 explored | s1 = +1, +1.5, +2; equivalent sign convention |
| Outer profile | power law, p_out=1 | tapered; s2=−1; Rc needs confirmation |
| Dust mass | refined grid 1, 1.5, 2, 3 × 10^-3 solar masses | 1, 5, 10 × 10^-3 solar masses |
| Porosity | reference 0%; explored 0%, 25% | 0%, 50%, 80% |
| Scale height at Rin | reference 0.93 au; explored 0.93, 1.40, 1.87 | 1.4 au, shared |
| Flaring exponent | refined grid 1.2, 1.3, 1.4 | 1.3, shared |

The recent discrete configuration has only 27 distinct combinations before
any additional parameters are freed. Repeated evaluations do not create a
continuous posterior. Describe categorical distributions as such; smoothed
corner contours can suggest support between values that were never sampled.

## Uncertainty analysis in progress

The inspected notebook filters the best 90% by cost before selecting the best
30% again inside the corner function. For distinct costs that is approximately
27% of successful highest-budget trials, not necessarily the intended fraction.
It weights samples by 1/cost and displays 16/50/84 percentiles. These are
descriptive weighted ensemble summaries, not calibrated credible intervals.
The notebook also reclassifies the maximum cost as a crash; preserve original
statuses instead. No notebook changes were made in this drafting task.

For repeated runs, compare final-stage best costs and individual observables,
use a consistent acceptance rule, distinguish repeated identical configurations
from independently supported parameter regions, and retain multimodality.
Separate optimiser-seed variation, MCFOST numerical noise, observational
uncertainty and alternative physical assumptions. Do not claim uncertainty
coverage or global convergence from software tests or corner plots alone.

## Checks and remaining inputs

The one-way height tie is implemented and the 13 existing automated tests
pass. These checks validate software behaviour, not scientific convergence.
No optimisation or MCFOST radiative-transfer simulation was launched.

The draft uses natbib citation commands and amsmath. A TeX compiler is not
available locally, so compilation was not verified. Supply the actual priors,
interferometry mode, corrected-polarimetry setting, MCFOST version and numerical
settings, final stage schedule, evaluation counts, repeat counts, and measured
stability/uncertainty results before treating this as a completed Methods section.
