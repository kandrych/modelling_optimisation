# Warmstart and saved-state continuation

The commands below show the relevant options. Replace `...` with the other
arguments required for your run, including its data and model settings.

## Warmstart

Import previous trial results into a run:

```bash
python SMAC3.py ... --working-root /new/work/ \
  --warmstart /previous/work/optimization/<old_seed>/
```

The current code filters eligible trials, sorts them by cost, and imports them
with `smac.tell()`. It does not restore the previous optimiser state.

**Original configuration IDs are not preserved.** Unseen configurations receive
IDs in import order; configurations already in the destination history reuse
their destination IDs. Match models across runs using their parameter values,
not their configuration IDs.

## Continue from saved state

Resume an existing run:

```bash
python SMAC3.py ... --working-root /existing/work/ --seed <existing_seed>
```

Omit `--warmstart`. Keep the original configuration space, fidelity settings,
model and scoring settings, worker count, and total trial limit. Use the same
software environment where possible. `overwrite=False` is already configured
in `SMAC3.py`.

Use the seed recorded in the **current run's**
`optimization/<existing_seed>/scenario.json`, including when that run originally
started through warmstart. Set `--seed` explicitly: the default `-1` requests a
random seed. The working root is the directory containing `optimization`, not
the seed directory itself.

Keep the saved run directory intact, including `scenario.json`,
`configspace.json`, `optimization.json`, `runhistory.json`, and
`intensifier.json`. SMAC automatically restores a matching run's history,
intensifier, and optimisation statistics. **Saved configuration IDs are
preserved.** This preserves the current history's IDs; it does not undo any
renumbering from an earlier warmstart.

Successful restoration logs:

```text
Continuing from previous run.
```

If SMAC reports a scenario mismatch, resolve the differences before continuing;
choosing to overwrite or rename starts a new run. Start the continuation only
after the previous optimiser has stopped.

`--n-trials` is the **total trial limit**, including previous trials, not the
number of additional trials. Extending an already completed run beyond its
original limit may require version-specific changes and is separate from
resuming an interrupted run.

With the current code, budgets come from `fidelity.txt` (or
`--fidelity-config`). Do not pass the obsolete `--min-budget` or `--max-budget`
options.

Reference: [SMAC saved-state continuation documentation](https://automl.github.io/SMAC3/latest/advanced_usage/10_continue/).
