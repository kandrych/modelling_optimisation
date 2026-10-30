# Trial walltime limits

Add these options to the existing `SMAC3.py` command (times are seconds):

```bash
# Stop after 45 minutes and leave the objective unknown:
--trial-walltime-limit 2700 --timeout-cost none

# Stop after 45 minutes and train SMAC with an explicit penalty:
--trial-walltime-limit 2700 --timeout-cost 1e99
```

Without `--trial-walltime-limit`, evaluation remains unlimited. With a limit,
`--timeout-cost none` is the default. A numeric penalty must be finite and positive;
choose it relative to the objective scale. A penalty smaller than valid losses
would encourage the surrogate to explore expensive regions.

The deadline covers the entire objective: parameter generation, all MCFOST
commands, diagnostic plotting and scoring. SMAC's existing pynisher integration
terminates the objective process and its descendants. Cleanup can take a few
seconds beyond the configured limit. Queuing time is not included. The final
incumbent regeneration also has the limit for its MCFOST and scoring steps; a
timeout there does not overwrite the incumbent's original measured score.

Every limited evaluation writes `trials/config_<id>_budget_<budget>/trial_status.json`.
Repeated evaluations use `_retrial1`, `_retrial2`, etc., choosing the first unused
folder name and preserving existing outputs. The record includes the
sampled configuration, config ID, seed, budget/fidelity, time limit, elapsed time,
policy, cost and status. On timeout it contains a message such as
`Trial exceeded its 2700 s walltime limit.` Existing parameter files, logs and
partial outputs are retained. Final regeneration records its status in
`smac_results/trial_status.json`.

Each final regeneration starts with an empty `smac_results` directory. Before
writing the new incumbent, an existing results directory is moved intact to
`smac_results_history/run_<unique suffix>/smac_results`. This preserves previous
figures, inputs, logs and partial outputs while preventing old PDI or
interferometry images from being reused for a new final model. A failed or
timed-out regeneration leaves its own partial outputs in `smac_results`; the next
attempt archives those too. The optimisation history and trial directories are
not moved. Do not run multiple drivers concurrently in the same working root.

## What each policy means

| Policy | Saved cost | Surrogate training | Scientific interpretation |
| --- | --- | --- | --- |
| `--timeout-cost 1e99` | `1e99` | Included as a penalty | Prefer models that can complete within the time limit; fit quality itself was not measured. |
| `--timeout-cost none` | JSON `null`, blank CSV cell | Excluded | Fit quality remains unknown. Nearby configurations may still be proposed. |

Both policies record SMAC status `TIMEOUT`, count toward `--n-trials`, release the
worker and allow exploration to continue. Timed-out evaluations cannot become
incumbents or be promoted to a higher fidelity. If all trials time out, SMAC
continues with sampling and finishes without an incumbent; it does not fabricate
a best-fit model. Neither policy automatically retries a timed-out configuration.
The saved configuration and `model.para`, if generated, identify what to rerun.

`none` prevents learning a fabricated bad fit, but slow regions will still have
fewer measured scores. A representative set of longer reruns is needed to assess
whether promising fits were missed. A numeric penalty deliberately makes runtime
part of model selection and can exclude scientifically good but expensive models.
Completed objective values and the relative weighting of observational constraints
are unchanged by either policy.

## SMAC compatibility

SMAC 2.3 requires numeric costs and ordinarily includes timeout rows in its
encoder. Returning Python `None` from a normal target function instead marks a
crash. The custom runner therefore retains a finite bookkeeping value (`1e99`)
inside SMAC for the `none` policy. The custom encoder explicitly excludes those
rows, including SMAC's separate timeout collection. `trial_status.json` and
`additional_info.trial_timeout.cost` contain the actual requested value (`null`),
and the final `trials.csv` export leaves unscored costs blank. When directly
analysing SMAC's native `runhistory.json`, check status and timeout metadata rather
than interpreting its numeric bookkeeping cost as a measured loss.

The custom intensifier excludes unfinished evaluations from incumbent selection
and promotion. Real finite penalties are still used for surrogate training.
Use the same timeout settings when resuming a run. For different limits or cost
policies, use a new working root; the existing `--warmstart` imports successful
evaluations, not timeout records. Editing scripts does not retrofit a running job;
the options take effect on the next invocation.

Tests use short synthetic objectives and child processes; they do not run MCFOST.
