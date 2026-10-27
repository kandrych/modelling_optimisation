"""Read-only analysis of saved SMAC/MCFOST histories; never launch simulations.

Used by run_summary.ipynb. Every result is conditional on the histories read in
this execution. "Final budget" means a configured fidelity stage, not a finished
search or a final model choice. Costs and observational uncertainties are unchanged.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from parameter_labels import parameter_label

SCORE_COLUMNS = {
    "SED": "score_sed", "PIONIER": "score_pionier", "GRAVITY": "score_gravity",
    "MATISSE L": "score_matisse_l", "MATISSE N": "score_matisse_n",
    "PDI V": "score_pdi_V", "PDI I": "score_pdi_I", "PDI H": "score_pdi_H",
    "ALMA": "score_alma",
}
INTERFEROMETRY = ["score_" + k for k in ("pionier", "gravity", "matisse_l", "matisse_n")]
PDI = ["score_pdi_" + b for b in "VIH"]
SUBSETS = {
    "SED": ["score_sed"], "SED+ALMA": ["score_sed", "score_alma"],
    "SED+interferometry": ["score_sed", *INTERFEROMETRY],
    "SED+PDI": ["score_sed", *PDI], "All": list(SCORE_COLUMNS.values()),
    "All minus ALMA": ["score_sed", *INTERFEROMETRY, *PDI],
    "All minus PDI": ["score_sed", *INTERFEROMETRY, "score_alma"],
    "All minus interferometry": ["score_sed", *PDI, "score_alma"],
    "All minus SED": [*INTERFEROMETRY, *PDI, "score_alma"],
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def discover_histories(folders):
    """Accept labelled parents, run folders, seed folders, or runhistory.json files.

    Bounded globs avoid walking the very large trials/ trees. Overlapping inputs
    are deduplicated by resolved history path; the first label wins.
    """
    if not isinstance(folders, dict):
        folders = {str(Path(p).expanduser()): p for p in folders}
    found, seen = [], set()
    for label, location in folders.items():
        root = Path(location).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        candidates = [root] if root.name == "runhistory.json" else []
        if root.is_dir():
            # Prune BEFORE descending: bounded wildcard globs can still enter
            # millions of files in trials/ even with a small maximum depth.
            queue, visited = [(root, 0)], set()
            excluded = {"trials", "smac_results", "data_figures", "dask-tmp",
                        "dask-logs", "run_summary_outputs", "__pycache__"}
            while queue:
                directory, depth = queue.pop()
                resolved = directory.resolve()
                if resolved in visited:
                    continue
                visited.add(resolved)
                if (directory / "runhistory.json").is_file():
                    candidates.append(directory / "runhistory.json")
                    continue
                if (directory / "optimization").is_dir():
                    candidates.extend(directory.glob("optimization/*/runhistory.json"))
                    continue
                if depth < 4:
                    queue.extend((child, depth+1) for child in directory.iterdir()
                                 if child.name not in excluded and not child.name.startswith((".", "data_"))
                                 and child.is_dir())
        for path in sorted(set(candidates)):
            path = path.resolve()
            if path in seen:
                continue
            if path.parent.parent.name != "optimization":
                raise ValueError(f"Expected RUN/optimization/SEED/runhistory.json: {path}")
            seen.add(path)
            run_root = path.parent.parent.parent
            try:
                relative = str(run_root.relative_to(root))
            except ValueError:
                relative = run_root.name
            run = str(label) if relative == "." else f"{label}/{relative}"
            found.append({"path": path, "run_root": run_root, "run": run,
                          "history": f"{run} [seed {path.parent.name}]"})
    if not found:
        raise FileNotFoundError("No RUN/optimization/SEED/runhistory.json found in the selected folders.")
    return found


def number(value):
    try:
        return float(value) if value is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def successful(status):
    return status == 1 or str(status).upper() in {"SUCCESS", "STATUSTYPE.SUCCESS"}


def score_terms(info):
    result = {c: np.nan for c in SCORE_COLUMNS.values()}
    result["score_sed"] = number(info.get("sed", {}).get("chi2_reduced"))
    mode = "vis2_chromatic" if "vis2_chromatic" in info else "vis2_1perband"
    for instrument in ("pionier", "gravity", "matisse_l", "matisse_n"):
        result["score_" + instrument] = number(info.get(mode, {}).get(instrument, {}).get("chi2_reduced"))
    for band in "VIH":
        result["score_pdi_" + band] = number(info.get("pdi", {}).get(band, {}).get("loss"))
    result["score_alma"] = number(info.get("alma", {}).get("loss"))
    return result


def audit_cost(info, cost, rtol=1e-8, atol=1e-8):
    """Check the additive recorded objective without treating absent terms as zero.

    Products are authoritative when recorded; otherwise infer them from payloads.
    A numerical coincidence cannot establish completeness of a partial record.
    """
    terms = score_terms(info)
    mapping = {"sed": ["score_sed"], "alma": ["score_alma"],
               "vis2_chromatic": INTERFEROMETRY, "vis2_1perband": INTERFEROMETRY,
               **{"pdi_"+b: ["score_pdi_"+b] for b in "VIH"}}
    products = products_for(info)
    expected = sorted({c for product in products for c in mapping.get(product, [])})
    unknown = sorted(set(products)-set(mapping))
    missing = [c for c in expected if not np.isfinite(terms[c])]
    unexpected = [c for c, v in terms.items() if np.isfinite(v) and c not in expected]
    finite = [v for v in terms.values() if np.isfinite(v)]
    available_sum = math.fsum(finite) if finite else np.nan
    complete = bool(expected) and not (missing or unknown or unexpected)
    reconstructed = math.fsum(terms[c] for c in expected) if complete else np.nan
    delta = reconstructed-cost
    match = bool(complete and np.isfinite(cost) and np.isclose(reconstructed, cost, rtol=rtol, atol=atol))
    status = ("incomplete_terms" if not complete else "non_finite_cost" if not np.isfinite(cost)
              else "match" if match else "mismatch")
    return {"available_score_sum": available_sum, "reconstructed_cost": reconstructed,
            "cost_terms_complete": complete, "cost_terms_match": match, "cost_audit_status": status,
            "cost_difference": delta, "cost_absolute_difference": abs(delta),
            "cost_relative_difference": delta/abs(cost) if np.isfinite(cost) and cost != 0 else np.nan,
            "expected_score_terms": ", ".join(expected), "missing_score_terms": ", ".join(missing),
            "unknown_products": ", ".join(unknown), "unexpected_score_terms": ", ".join(unexpected),
            "products_source": "recorded" if info.get("fidelity", {}).get("products") else "inferred"}


def products_for(info):
    recorded = info.get("fidelity", {}).get("products")
    if recorded:
        return tuple(sorted(recorded))
    products = [k for k in ("sed", "vis2_chromatic", "vis2_1perband", "alma") if k in info]
    products.extend("pdi_" + b for b in info.get("pdi", {}))
    return tuple(sorted(products))


def objective_label(products):
    """Readable identifier retaining interferometry mode and individual PDI bands."""
    products = set(products)
    parts = []
    if "sed" in products:
        parts.append("SED")
    for product, label in (("vis2_1perband", "INT(band)"), ("vis2_chromatic", "INT(chrom)")):
        if product in products:
            parts.append(label)
    bands = [b for b in "VIH" if "pdi_"+b in products]
    if bands:
        parts.append("PDI("+",".join(bands)+")")
    if "alma" in products:
        parts.append("ALMA")
    known = {"sed", "vis2_1perband", "vis2_chromatic", "alma", *["pdi_"+b for b in "VIH"]}
    unknown = sorted(products-known)
    if unknown:
        # JSON escaping preserves distinctions even for unusual future product names.
        parts.append("other="+json.dumps(unknown, separators=(",", ":")))
    return "+".join(parts) if parts else "No products"


def scoring_fingerprint(info):
    """Known scoring/data descriptors for a common fully-scored pool.

    Observed quadrants vary with trial PA and must not be used as data hashes.
    This is evidence of compatibility, not a proof of historical data identity.
    """
    parts = {"products": products_for(info), "pdi": {},
             "numerical_fidelity": {k: v for k, v in info.get("fidelity", {}).items()
                                    if k not in {"products", "stage", "budget"}}}
    for band, payload in info.get("pdi", {}).items():
        constraints = payload.get("constraints", {})
        observed, model = constraints.get("observed", {}), constraints.get("model", {})
        weights = {}
        for term, key in (("fraction", "positive_qphi_over_i"), ("radial", "qphi_profile"), ("quadrant", "quadrants")):
            try:
                squared = (np.asarray(model[key], float) - np.asarray(observed[key], float)) ** 2
                raw = float(np.mean(squared) if term == "radial" else np.sum(squared))
                weights[term] = float(f"{number(constraints[term]) / raw:.8g}") if raw > 0 else None
            except (KeyError, ValueError, TypeError):
                weights[term] = None
        parts["pdi"][band] = {
            "corrected": payload.get("model_unresolved_corrected", "unknown"),
            "radius_px": payload.get("unresolved_correction_radius_px"),
            "definition": constraints.get("loss_definition", "unknown"),
            "error_model": constraints.get("error_model", "unknown"), "inferred_weights": weights,
            "observed": {k: observed.get(k) for k in ("positive_qphi_over_i", "qphi_profile", "radial_edges_mas", "aperture_radius_mas")},
        }
    alma = info.get("alma", {}).get("comparison", {})
    parts["alma_observed"] = {k: alma.get(k) for k in ("observed_flux_jy", "observed_energy", "aperture_radius_mas", "aperture_pixel_count")}
    mode = "vis2_chromatic" if "vis2_chromatic" in info else "vis2_1perband"
    parts["interferometry"] = {k: v.get("num_points") for k, v in info.get(mode, {}).items() if isinstance(v, dict)}
    return parts


def load_runs(folders, penalty_limit=1e90):
    """Return flat evaluations, inventory, raw records and ConfigSpace definitions."""
    rows, inventory, records, spaces = [], [], {}, {}
    for entry in discover_histories(folders):
        path, root, history = entry["path"], entry["run_root"], entry["history"]
        raw = path.read_bytes()
        payload = json.loads(raw)
        if any(not isinstance(r, dict) for r in payload.get("data", [])):
            raise ValueError(f"Unsupported legacy runhistory layout: {path}. Expected dictionary trial rows.")
        scenario_path = path.parent / "scenario.json"
        scenario = read_json(scenario_path) if scenario_path.exists() else {}
        space_path = path.parent / "configspace.json"
        space = read_json(space_path) if space_path.exists() else {}
        spaces[history] = space
        config_paths = sorted([*root.glob("*.yaml"), *root.glob("*.yml")])
        config_paths = [p for p in config_paths if "fidelity" not in p.stem]
        physical_space = {k: space.get(k) for k in ("hyperparameters", "conditions", "forbiddens")} if space else {
            p.name: file_hash(p) for p in config_paths}
        physical_hash = digest(physical_space) if space or config_paths else "unknown"
        template_hash = file_hash(root / "simulation.para") or "unknown"
        stage_path = root / "fidelity_resolved.json"
        stages = read_json(stage_path).get("stages", []) if stage_path.exists() else []
        budgets = [number(r.get("budget")) for r in payload.get("data", [])]
        declared_max = number(scenario.get("max_budget"))
        if not np.isfinite(declared_max):
            declared_max = max((number(s.get("budget")) for s in stages), default=np.nan)
        if not np.isfinite(declared_max):
            declared_max = max((b for b in budgets if np.isfinite(b)), default=np.nan)
        launch = (root / "run_smac.q").read_text() if (root / "run_smac.q").exists() else ""
        photons = []
        if (root / "simulation.para").exists():
            photons = [line.strip() for line in (root / "simulation.para").read_text().splitlines() if "nbr_photons_" in line]
        inventory.append({"history": history, "run": entry["run"], "run_root": str(root),
                          "history_path": str(path), "optimizer_seed": scenario.get("seed", path.parent.name),
                          "configured_max_budget": declared_max, "requested_trials": scenario.get("n_trials"),
                          "recorded_trials": len(payload.get("data", [])), "configspace_hash": physical_hash,
                          "template_hash": template_hash, "history_sha256": hashlib.sha256(raw).hexdigest(),
                          "launch_requests_correction": "--correct-unresolved-polarimetry" in launch,
                          "photon_settings": " | ".join(photons), "stages": json.dumps(stages)})
        for index, trial in enumerate(payload.get("data", [])):
            info = trial.get("additional_info") or {}
            cfg = payload.get("configs", {}).get(str(trial.get("config_id")))
            cost = number(trial.get("cost"))
            failure = info.get("failure") or {}
            reason = ("recorded_failure" if failure else "unfinished" if trial.get("status") == 0 else
                      "non_success" if not successful(trial.get("status")) else
                      "non_finite_cost" if not np.isfinite(cost) else "penalty_cost" if cost >= penalty_limit else
                      "missing_configuration" if not isinstance(cfg, dict) else "usable")
            cfg = cfg if isinstance(cfg, dict) else {}
            evaluation = f"{history}/row_{index}"
            terms = score_terms(info)
            products = products_for(info)
            scoring = scoring_fingerprint(info)
            budget = number(trial.get("budget"))
            stage = info.get("fidelity", {}).get("stage")
            if not stage:
                stage = next((s.get("stage") for s in stages if number(s.get("budget")) == budget), "unknown")
            row = {"evaluation_id": evaluation, "history": history, "run": entry["run"],
                   "config_id": trial.get("config_id"), "seed": trial.get("seed"),
                   "instance": json.dumps(trial.get("instance"), sort_keys=True),
                   "budget": budget, "stage": stage, "products": ", ".join(products),
                   "objective_key": objective_label(products), "configured_max_budget": declared_max,
                   "status": trial.get("status"), "usable": reason == "usable", "exclusion": reason,
                   "failure_reason": failure.get("reason", ""), "cost": cost,
                   "time_seconds": number(trial.get("time")), "endtime": number(trial.get("endtime")),
                   "record_index": index, "parameter_hash": digest(cfg),
                   **audit_cost(info, cost),
                   "comparison_key": digest({"space": physical_hash, "template": template_hash, "scoring": scoring}),
                   "known_physical_metadata": physical_hash != "unknown" and template_hash != "unknown",
                   "sed_ebminv": number(info.get("sed", {}).get("ebminv")), **terms}
            for band in "VIH":
                band_info = info.get("pdi", {}).get(band, {})
                row[f"pdi_{band}_corrected"] = band_info.get("model_unresolved_corrected")
                row[f"pdi_{band}_correction_radius_px"] = band_info.get("unresolved_correction_radius_px")
                row[f"pdi_{band}_ssim"] = number(band_info.get("ssim"))
            row.update({"param__" + k: v for k, v in cfg.items()})
            rows.append(row)
            records[evaluation] = {"trial": trial, "info": info, "cfg": cfg, "run_root": root,
                                   "history_path": path, "scoring_metadata": scoring}
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Histories were found but contain no trials.")
    return frame, pd.DataFrame(inventory), records, spaces


def coverage_table(evaluations):
    rows = []
    for keys, group in evaluations.groupby(["history", "budget", "stage", "products"], dropna=False, sort=False):
        good = group[group.usable]
        rows.append(dict(zip(("history", "budget", "stage", "products"), keys),
                         recorded=len(group), usable=len(good), excluded=len(group)-len(good),
                         failures=int(group.exclusion.isin(["recorded_failure", "penalty_cost", "non_success"]).sum()),
                         unfinished=int(group.exclusion.eq("unfinished").sum()),
                         best_cost=good.cost.min(), evaluation_wall_hours=group.time_seconds.sum()/3600))
    return pd.DataFrame(rows)


def select_solutions(evaluations, relative_tolerance=0.10, absolute_tolerance=0.0):
    """Use configured final budget, not merely the highest successful budget.

    Competitive selection is descriptive; it is not a likelihood threshold.
    Repeated evaluations remain visible in the source table; representative
    model tables keep the lowest observed score per parameter hash and objective.
    The allowance is max(abs(best) * relative_tolerance, absolute_tolerance) in
    score units. The two returned frames are current minima and competitive rows.
    """
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("Nonnegative competitive tolerances are required.")
    final = evaluations[evaluations.usable & np.isclose(evaluations.budget, evaluations.configured_max_budget)].copy()
    keys = ["history", "objective_key"]
    unique = final.sort_values("cost").drop_duplicates([*keys, "parameter_hash"])
    best, competitive = [], []
    for _, group in unique.groupby(keys, sort=False):
        group = group.sort_values("cost")
        winner = group.iloc[0]
        threshold = winner.cost + max(absolute_tolerance, abs(winner.cost)*relative_tolerance)
        best.append(group.iloc[:1])
        competitive.append(group[group.cost <= threshold])
    empty = evaluations.iloc[:0].copy()
    return tuple(pd.concat(parts, ignore_index=True) if parts else empty.copy() for parts in (best, competitive))


def parameter_summary(competitive, spaces):
    rows = []
    for history, group in competitive.groupby("history", sort=False):
        definitions = {p["name"]: p for p in spaces.get(history, {}).get("hyperparameters", [])}
        for column in (c for c in group if c.startswith("param__")):
            name = column.removeprefix("param__")
            definition = definitions.get(name, {})
            if definition.get("type") == "constant":
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            lower, upper = definition.get("lower"), definition.get("upper")
            near_lower = near_upper = np.nan
            if lower is not None and upper is not None:
                if definition.get("log") and lower > 0 and (values > 0).all():
                    positions = (np.log(values)-math.log(lower))/(math.log(upper)-math.log(lower))
                else:
                    positions = (values-lower)/(upper-lower)
                near_lower, near_upper = float((positions <= .02).mean()), float((positions >= .98).mean())
            rows.append({"history": history, "parameter": name, "models": len(group),
                         "sample_min": values.min(), "sample_median": values.median(), "sample_max": values.max(),
                         "lower_bound": lower, "upper_bound": upper,
                         "fraction_near_lower_bound": near_lower, "fraction_near_upper_bound": near_upper,
                         "interpretation": "sampled competitive models; not a confidence interval"})
    return pd.DataFrame(rows)


def parameter_pair_analysis(evaluations, competitive, spaces, parameters=None,
                            min_models=20, min_unique=3, periodic_parameters=("disk_pa",)):
    """Quantify all independent parameter pairs within each final-stage objective.

    Spearman rho and Kendall tau-b describe association in the same finite
    paired configurations; tau-b explicitly accounts for ties on either axis.
    Neither is a posterior correlation or a significance test. Compute both the
    competitive subset and all usable final-stage configurations as a control.
    Fixed/tied ConfigSpace parameters are excluded; circular and nonnumeric
    pairs are retained in the audit with an explicit exclusion reason.
    """
    if min_models < 3 or min_unique < 2:
        raise ValueError("Pair analysis requires min_models >= 3 and min_unique >= 2.")
    columns = ["history", "objective_key", "products", "parameter_x", "parameter_y", "direct_configspace_constraint"]
    for scope in ("all_final", "competitive"):
        columns += [f"{name}_{scope}" for name in ("n", "unique_x", "unique_y", "rho", "tau_b", "eligible", "reason")]
    final = evaluations[evaluations.usable & np.isclose(evaluations.budget, evaluations.configured_max_budget)]
    final = final.sort_values("cost").drop_duplicates(["history", "objective_key", "parameter_hash"])

    def linked_pairs(value):
        links = set()
        if isinstance(value, dict):
            if isinstance(value.get("left"), str) and isinstance(value.get("right"), str):
                links.add(frozenset((value["left"], value["right"])))
            for nested in value.values():
                links.update(linked_pairs(nested))
        elif isinstance(value, list):
            for nested in value:
                links.update(linked_pairs(nested))
        return links

    def statistics(group, x, y):
        names = ["param__"+x, "param__"+y]
        original = group.reindex(columns=names)
        numeric = original.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        nonnumeric = (original.notna() & numeric.isna()).any().any()
        pair = numeric.dropna()
        n = len(pair)
        ux, uy = (pair[c].nunique() for c in names)
        periodic = x in periodic_parameters or y in periodic_parameters
        rho = tau_b = np.nan
        if n >= 3 and ux > 1 and uy > 1 and not periodic and not nonnumeric:
            # Use exactly the same complete rows for both metrics. Request b
            # explicitly so ties in categorical physical values are normalised.
            rho = float(pair[names[0]].rank().corr(pair[names[1]].rank()))
            tau_b = float(kendalltau(pair[names[0]].to_numpy(), pair[names[1]].to_numpy(),
                                    variant="b", nan_policy="raise").statistic)
        reason = ("periodic_parameter_requires_circular_analysis" if periodic else
                  "nonnumeric_or_nonfinite_values" if nonnumeric else
                  "too_few_models" if n < min_models else
                  "too_few_distinct_values" if min(ux, uy) < min_unique else "eligible")
        return {"n": n, "unique_x": ux, "unique_y": uy, "rho": rho, "tau_b": tau_b,
                "eligible": reason == "eligible" and np.isfinite(rho) and np.isfinite(tau_b), "reason": reason}

    rows = []
    for (history, objective), group in final.groupby(["history", "objective_key"], sort=False):
        space = spaces.get(history, {})
        definitions = {p["name"]: p for p in space.get("hyperparameters", [])}
        candidates = parameters if parameters is not None else [c.removeprefix("param__") for c in group if c.startswith("param__")]
        names = []
        for name in dict.fromkeys(candidates):
            column = "param__"+name
            if column not in group or not group[column].notna().any():
                continue
            definition = definitions.get(name, {})
            if definition.get("type") == "constant":
                continue
            if not definition and group[column].dropna().nunique() <= 1:
                continue
            # A symbolic reference represents a tied parameter, not a new axis.
            if group[column].dropna().map(lambda v: isinstance(v, str) and "param__"+v in group.columns).any():
                continue
            names.append(name)
        constrained = linked_pairs(space.get("forbiddens", []))
        subset = competitive[(competitive.history == history) & (competitive.objective_key == objective)]
        subset = subset.drop_duplicates("parameter_hash")
        for x, y in combinations(sorted(names), 2):
            row = {"history": history, "objective_key": objective,
                   "products": group.products.iloc[0] if "products" in group else "unknown",
                   "parameter_x": x, "parameter_y": y,
                   "direct_configspace_constraint": frozenset((x, y)) in constrained}
            for scope, sample in (("all_final", group), ("competitive", subset)):
                row.update({f"{k}_{scope}": v for k, v in statistics(sample, x, y).items()})
            rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def select_parameter_pairs(pair_analysis, scope="competitive", min_abs_rho=.5,
                           max_plots=6, max_per_history=2, include_constrained=False):
    """Select a bounded set of eligible pairs by absolute within-history rho.

    Round-robin selection across histories avoids filling every panel with one
    optimiser's solution family. The complete pair table remains available.
    """
    if scope not in ("competitive", "all_final"):
        raise ValueError("Pair scope must be 'competitive' or 'all_final'.")
    if not 0 <= min_abs_rho <= 1 or max_plots < 1 or max_per_history < 1:
        raise ValueError("Require 0 <= min_abs_rho <= 1 and positive plot limits.")
    if pair_analysis.empty:
        return pair_analysis.assign(selection_scope=pd.Series(dtype=str), abs_rho=pd.Series(dtype=float),
                                    rank_within_history=pd.Series(dtype=int))
    selected = pair_analysis.loc[pair_analysis[f"eligible_{scope}"].eq(True)].copy()
    if not include_constrained:
        selected = selected[~selected.direct_configspace_constraint]
    selected["selection_scope"] = scope
    selected["abs_rho"] = selected[f"rho_{scope}"].abs()
    selected = selected[selected.abs_rho >= min_abs_rho]
    selected = selected.sort_values(["abs_rho", f"n_{scope}", "parameter_x", "parameter_y"],
                                    ascending=[False, False, True, True], kind="stable")
    selected["rank_within_history"] = selected.groupby("history", sort=False).cumcount()+1
    selected = selected[selected.rank_within_history <= max_per_history]
    return selected.sort_values(["rank_within_history", "abs_rho"], ascending=[True, False], kind="stable").head(max_plots).reset_index(drop=True)


CORRELATION_COLORS = (
    "#053061", "#2166ac", "#4393c3", "#92c5de", "#f4f4f4",
    "#f4a582", "#d6604d", "#b2182b", "#67001f",
)
CORRELATION_LEVELS = ("Negligible", "Weak", "Moderate", "Strong", "Very strong")
PAIR_CORRELATION_METRICS = {
    "spearman": ("rho", "Spearman", "ρ"),
    "kendall_tau_b": ("tau_b", "Kendall Tau-b", "τ-b"),
}


def correlation_metric(metric):
    if metric not in PAIR_CORRELATION_METRICS:
        raise ValueError("Metric must be 'spearman' or 'kendall_tau_b'.")
    return PAIR_CORRELATION_METRICS[metric]


def correlation_band(rho):
    """Signed colour bin, 0..8; thresholds describe magnitude, not significance."""
    if not np.isfinite(number(rho)):
        return None
    rho = number(rho)
    if abs(rho) > 1 + 1e-12:
        raise ValueError("Correlation coefficient must lie between -1 and 1.")
    # Avoid colouring an exact theoretical boundary (e.g. tau-b=0.8) in the
    # lower bin merely because floating arithmetic returns 0.7999999999999999.
    level = sum(abs(rho) >= threshold or math.isclose(abs(rho), threshold, rel_tol=0, abs_tol=1e-12)
                for threshold in (.2, .4, .6, .8))
    return 4 + (level if rho >= 0 else -level)


def correlation_strength(rho):
    band = correlation_band(rho)
    return "Unavailable" if band is None else CORRELATION_LEVELS[abs(band-4)]


def parameter_correlation_matrices(pair_analysis, scope="competitive", metric="spearman"):
    """Symmetric coefficient, eligibility and constraint matrices for ONE history/objective.

    Keep raw coefficients in the returned matrix even when ineligible, so they
    remain auditable. The plot masks ineligible cells. Diagonals are omitted.
    """
    if scope not in ("competitive", "all_final"):
        raise ValueError("Pair scope must be 'competitive' or 'all_final'.")
    prefix, _, _ = correlation_metric(metric)
    if not pair_analysis.empty and len(pair_analysis[["history", "objective_key"]].drop_duplicates()) != 1:
        raise ValueError("A correlation matrix must contain exactly one history and objective.")
    names = sorted(set(pair_analysis.parameter_x) | set(pair_analysis.parameter_y))
    rho = pd.DataFrame(np.nan, index=names, columns=names)
    eligible = pd.DataFrame(False, index=names, columns=names)
    constrained = eligible.copy()
    for _, row in pair_analysis.iterrows():
        x, y = row.parameter_x, row.parameter_y
        rho.loc[x, y] = rho.loc[y, x] = row[f"{prefix}_{scope}"]
        eligible.loc[x, y] = eligible.loc[y, x] = bool(row[f"eligible_{scope}"])
        constrained.loc[x, y] = constrained.loc[y, x] = bool(row.direct_configspace_constraint)
    return rho, eligible, constrained


def plot_parameter_correlation_heatmap(pair_analysis, scope="competitive", metric="spearman"):
    """Annotated signed/strength-binned rank-correlation heatmap for one history."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Rectangle
    if pair_analysis.empty:
        return None
    _, metric_label, symbol = correlation_metric(metric)
    rho, eligible, constrained = parameter_correlation_matrices(pair_analysis, scope, metric)
    names = list(rho.index)
    bins = np.full(rho.shape, np.nan)
    for i in range(len(names)):
        for j in range(len(names)):
            if eligible.iloc[i, j] and np.isfinite(rho.iloc[i, j]):
                bins[i, j] = correlation_band(rho.iloc[i, j])
    cmap = ListedColormap(CORRELATION_COLORS).with_extremes(bad="#dddddd")
    norm = BoundaryNorm(np.arange(-.5, 9.5), 9)
    fig, ax = plt.subplots(figsize=(max(11, .58*len(names)+4), max(9, .58*len(names)+3)))
    im = ax.imshow(np.ma.masked_invalid(bins), cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(range(len(names)), [parameter_label(name) for name in names], rotation=65, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), [parameter_label(name) for name in names], fontsize=8)
    ax.set_xticks(np.arange(-.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    font_size = max(6, min(9, 9-.12*len(names)))
    for i in range(len(names)):
        for j in range(len(names)):
            value = rho.iloc[i, j]
            if i == j:
                text, color = "—", "#555555"
            elif not eligible.iloc[i, j] or not np.isfinite(value):
                text, color = "NA", "#666666"
            else:
                text = f"{value:+.2f}" + ("†" if constrained.iloc[i, j] else "")
                color = "white" if abs(value) >= .6 else "#222222"
                if constrained.iloc[i, j]:
                    ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False,
                                           edgecolor="#7b3294", linewidth=1.6))
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=font_size)
    colorbar = fig.colorbar(im, ax=ax, ticks=range(9), fraction=.035, pad=.03)
    colorbar.ax.set_yticklabels(["Very strong −", "Strong −", "Moderate −", "Weak −", "Negligible",
                                "Weak +", "Moderate +", "Strong +", "Very strong +"], fontsize=8)
    colorbar.set_label(f"{metric_label} {symbol}: sign and strength", fontsize=9)
    history = pair_analysis.history.iloc[0]
    counts = pair_analysis[f"n_{scope}"]
    count_text = str(int(counts.iloc[0])) if counts.nunique() == 1 else f"{int(counts.min())}–{int(counts.max())}"
    ax.set_title(f"{history}\n{metric_label} correlation: {scope}; paired n={count_text}", pad=15, fontsize=11)
    fig.text(.02, .015,
             f" — = diagonal omitted; † = directly linked by ConfigSpace.\n", fontsize=font_size)
    fig.tight_layout(rect=(0, .085, 1, 1))
    return fig


def plot_selected_parameter_pairs(selected, evaluations, competitive, log_parameters=(), metric="spearman"):
    import matplotlib.pyplot as plt
    if selected.empty:
        return None
    prefix, metric_label, symbol = correlation_metric(metric)
    ncols = min(2, len(selected))
    fig, axes = plt.subplots(math.ceil(len(selected)/ncols), ncols,
                             figsize=(14, 4.6*math.ceil(len(selected)/ncols)), squeeze=False)
    for ax, (_, row) in zip(axes.flat, selected.iterrows()):
        full = evaluations[evaluations.usable & (evaluations.history == row.history)
                           & (evaluations.objective_key == row.objective_key)
                           & np.isclose(evaluations.budget, evaluations.configured_max_budget)]
        full = full.sort_values("cost").drop_duplicates("parameter_hash")
        subset = competitive[(competitive.history == row.history) & (competitive.objective_key == row.objective_key)]
        subset = subset.drop_duplicates("parameter_hash")
        x, y = "param__"+row.parameter_x, "param__"+row.parameter_y
        scope = row.get("selection_scope", "competitive")
        coefficient = row[f"{prefix}_{scope}"]
        band = correlation_band(coefficient)
        color = CORRELATION_COLORS[band] if band is not None else "grey"
        level = correlation_strength(coefficient)
        # Point appearance identifies sample membership; only the border encodes correlation.
        ax.scatter(pd.to_numeric(full[x], errors="coerce"), pd.to_numeric(full[y], errors="coerce"),
                   marker="o", s=24, alpha=0.4, color="black", linewidths=0,
                   label="All usable final-stage models", zorder=1)
        ax.scatter(pd.to_numeric(subset[x], errors="coerce"), pd.to_numeric(subset[y], errors="coerce"),
                   marker="o", s=27, alpha=.85, color="#2166ac", linewidths=0,
                   label="Competitive models", zorder=2)
        ax.set(xlabel=parameter_label(row.parameter_x), ylabel=parameter_label(row.parameter_y))
        sign = "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero"
        history_title = f"{row.history}\n" if selected.history.nunique() > 1 else ""
        ax.set_title(f"{history_title}{level} {sign} association ({metric_label}, {scope})\n"
                     f"Competitive: ρ={row.rho_competitive:+.2f}, τ-b={row.tau_b_competitive:+.2f}, n={row.n_competitive}\n"
                     f"All final: ρ={row.rho_all_final:+.2f}, τ-b={row.tau_b_all_final:+.2f}, n={row.n_all_final}", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1 + .4*abs(band-4) if band is not None else 1)
        for name, axis in ((row.parameter_x, "x"), (row.parameter_y, "y")):
            values = pd.to_numeric(full["param__"+name], errors="coerce").dropna()
            if name in log_parameters and len(values) and values.gt(0).all():
                getattr(ax, f"set_{axis}scale")("log")
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)
    title = (f"{metric_label} parameter associations: {selected.history.iloc[0]}" if selected.history.nunique() == 1
             else f"{metric_label} parameter associations selected from quantified within-history pairs")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return fig


def stage_correlations(evaluations):
    rows = []
    for history, group in evaluations[evaluations.usable].groupby("history", sort=False):
        budgets = sorted(group.budget.dropna().unique())
        for low, high in zip(budgets, budgets[1:]):
            keys = ["parameter_hash", "seed", "instance"]
            a = group[group.budget == low].groupby(keys, dropna=False).cost.mean().rename("low")
            b = group[group.budget == high].groupby(keys, dropna=False).cost.mean().rename("high")
            paired = pd.concat([a, b], axis=1).dropna()
            rho = paired.low.rank().corr(paired.high.rank()) if len(paired) > 2 else np.nan
            rows.append({"history": history, "low_budget": low, "high_budget": high,
                         "paired_configurations": len(paired), "spearman_rho": rho,
                         "interpretation": "selected paired evaluations; repeated key scores averaged only for this diagnostic"})
    return pd.DataFrame(rows)


def common_pool_comparison(evaluations):
    """Reconstruct subset objectives only from models already scored on ALL groups.

    Partition by known physical/scoring/data metadata; never fill missing scores
    with zero or silently combine incompatible physical spaces/templates.
    """
    all_columns = SUBSETS["All"]
    pool = evaluations[evaluations.usable & evaluations.known_physical_metadata & evaluations.cost_terms_match].dropna(subset=all_columns).copy()
    pool = pool.sort_values("cost").drop_duplicates(["comparison_key", "parameter_hash", "seed", "instance", *all_columns])
    if pool.empty:
        return pool, pd.DataFrame()
    pool["pool_group"] = pool.comparison_key.str[:12]
    rows = []
    for group_id, group in pool.groupby("pool_group", sort=False):
        for name, columns in SUBSETS.items():
            scores = group[columns].sum(axis=1, min_count=len(columns))
            winner = group.loc[scores.idxmin()]
            rows.append({"pool_group": group_id, "objective": name, "models": group.parameter_hash.nunique(),
                         "evaluations": len(group), "evaluation_id": winner.evaluation_id, "history": winner.history,
                         "config_id": winner.config_id, "budget": winner.budget,
                         "subset_score": scores.min(), "all_observation_score": winner.cost})
    return pool, pd.DataFrame(rows)


def prediction_tables(selection, records):
    pdi_rows, alma_rows = [], []
    for _, row in selection.iterrows():
        info = records[row.evaluation_id]["info"]
        for band, band_info in info.get("pdi", {}).items():
            constraints = band_info.get("constraints", {})
            observed = number(constraints.get("observed", {}).get("positive_qphi_over_i"))
            predicted = number(constraints.get("model", {}).get("positive_qphi_over_i"))
            pdi_rows.append({"evaluation_id": row.evaluation_id, "history": row.history, "band": band,
                             "observed_fraction": observed, "model_fraction": predicted,
                             "model_over_observed": predicted/observed if observed > 0 else np.nan,
                             "corrected": band_info.get("model_unresolved_corrected"),
                             "correction_radius_px": band_info.get("unresolved_correction_radius_px"),
                             "fraction_loss": constraints.get("fraction"), "radial_loss": constraints.get("radial"),
                             "quadrant_loss": constraints.get("quadrant"), "band_loss": band_info.get("loss"),
                             "ssim_diagnostic": band_info.get("ssim")})
        comparison = info.get("alma", {}).get("comparison", {})
        if comparison:
            alma_rows.append({"evaluation_id": row.evaluation_id, "history": row.history,
                              **{k: comparison.get(k) for k in ("observed_flux_jy", "model_flux_jy", "model_to_observed_flux", "residual_energy_loss", "aperture_radius_mas")}})
    return pd.DataFrame(pdi_rows), pd.DataFrame(alma_rows)


def resolve_model_files(row, records):
    """Match actual saved configurations, avoiding ID collisions across histories."""
    record = records[row.evaluation_id]
    cfg, root = record["cfg"], record["run_root"]
    matches = []
    for folder in sorted((root / "trials").glob(f"config_{int(row.config_id)}_budget_{row.budget:.2f}*")):
        path = folder / "config_used.json"
        if not path.exists():
            continue
        saved = read_json(path)
        actual = saved.get("cfg", {})
        same = True
        for key, value in cfg.items():
            expected = cfg.get(value, value) if isinstance(value, str) else value
            actual_value = actual.get(key)
            if isinstance(expected, (int, float)):
                same &= np.isclose(number(actual_value), expected, rtol=1e-10, atol=0)
            else:
                same &= actual_value == expected
        saved_fidelity = saved.get("fidelity", {})
        same &= np.isclose(number(saved_fidelity.get("budget")), row.budget)
        same &= tuple(sorted(saved_fidelity.get("products", []))) == products_for(record["info"])
        if same:
            matches.append({"folder": folder, "effective_cfg": actual})
    return matches


def plot_costs(best):
    import matplotlib.pyplot as plt
    if best.empty:
        return None
    fig, ax = plt.subplots(figsize=(13, max(4, .55*len(best))))
    left = np.zeros(len(best))
    colors = plt.get_cmap("tab10").colors
    for (label, column), color in zip(SCORE_COLUMNS.items(), colors):
        values = best[column].fillna(0).to_numpy(float)
        ax.barh(np.arange(len(best)), values, left=left, label=label, color=color)
        left += values
    for i, total in enumerate(best.cost):
        ax.annotate(f"{total:.2f}", (left[i], i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(np.arange(len(best)), best.history, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left.max(), 1)*1.14)
    ax.set_xlabel("Recorded composite objective; missing groups are unscored")
    ax.set_title("Current lowest-score target-budget candidates: observation contributions")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.13))
    fig.tight_layout()
    return fig


def plot_convergence(evaluations):
    import matplotlib.pyplot as plt
    good = evaluations[evaluations.usable]
    histories = good.history.unique()
    if not len(histories):
        return None
    ncols = min(2, len(histories))
    fig, axes = plt.subplots(math.ceil(len(histories)/ncols), ncols,
                             figsize=(13, 3.7*math.ceil(len(histories)/ncols)), squeeze=False)
    for ax, history in zip(axes.flat, histories):
        for budget, group in good[good.history == history].groupby("budget"):
            # Completion order when timestamps exist; otherwise saved row order.
            group = group.sort_values("endtime" if group.endtime.gt(0).all() else "record_index", kind="stable")
            ax.step(np.arange(1, len(group)+1), group.cost.cummin(), where="post", label=f"budget {budget:g}")
        ax.set(title=history, xlabel="Usable evaluations at this budget", ylabel="Best recorded cost so far")
        if good.loc[good.history == history, "cost"].gt(0).all():
            ax.set_yscale("log")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    for ax in axes.flat[len(histories):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_parameters(competitive, best, names, log_parameters=()):
    import matplotlib.pyplot as plt
    names = [p for p in names if "param__" + p in competitive]
    if competitive.empty or not names:
        return None
    histories = competitive.history.unique()
    fig, axes = plt.subplots(math.ceil(len(names)/3), 3, figsize=(16, 4.5*math.ceil(len(names)/3)), squeeze=False)
    rng = np.random.default_rng(42)
    for ax, name in zip(axes.flat, names):
        for i, history in enumerate(histories):
            values = pd.to_numeric(competitive.loc[competitive.history == history, "param__"+name], errors="coerce").dropna()
            ax.scatter(i+rng.uniform(-.16, .16, len(values)), values, s=15, alpha=.45, color=f"C{i%10}")
            winner = pd.to_numeric(best.loc[best.history == history, "param__"+name], errors="coerce").dropna()
            ax.scatter(np.repeat(i, len(winner)), winner, marker="*", s=100, color="black", zorder=4)
        ax.set_xticks(range(len(histories)), histories, rotation=75, ha="right", fontsize=7)
        ax.set_title(parameter_label(name))
        all_values = pd.to_numeric(competitive["param__"+name], errors="coerce").dropna()
        if name in log_parameters and len(all_values) and all_values.gt(0).all():
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=.2)
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Competitive sampled models; black stars = best recorded models", y=1.01)
    fig.tight_layout()
    return fig


def plot_amplitudes(pdi, alma):
    import matplotlib.pyplot as plt
    if pdi.empty and alma.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    if not pdi.empty:
        for history, group in pdi.groupby("history", sort=False):
            group = group.set_index("band").reindex(list("VIH"))
            values = group.model_over_observed.where(group.model_over_observed > 0)
            axes[0].plot(list("VIH"), values, "o-", label=history)
        axes[0].axhline(1, color="black", ls="--")
        axes[0].set_yscale("log")
        axes[0].set(ylabel="Model / observed positive-Qphi-to-I fraction", title="PDI amplitude (zero/negative ratios omitted)")
        axes[0].legend(fontsize=7)
    else:
        axes[0].text(.5, .5, "No saved PDI constraints", ha="center")
    if not alma.empty:
        axes[1].barh(alma.history, pd.to_numeric(alma.model_to_observed_flux, errors="coerce"))
        axes[1].axvline(1, color="black", ls="--")
        axes[1].set(xlabel="Model / observed aperture flux", title="ALMA amplitude")
        axes[1].tick_params(axis="y", labelsize=7)
    else:
        axes[1].text(.5, .5, "No saved ALMA aperture comparison", ha="center")
    fig.tight_layout()
    return fig


def plot_saved_profiles(selection, records, observable="pdi"):
    import matplotlib.pyplot as plt
    count = 3 if observable == "pdi" else 1
    fig, axes = plt.subplots(1, count, figsize=(5*count, 4.5), squeeze=False)
    drawn = False
    for _, row in selection.iterrows():
        info = records[row.evaluation_id]["info"]
        for i, band in enumerate("VIH" if observable == "pdi" else ["alma"]):
            ax = axes[0, i]
            payload = info.get("pdi", {}).get(band, {}).get("constraints", {}) if observable == "pdi" else info.get("alma", {}).get("comparison", {})
            if not payload:
                continue
            if observable == "pdi":
                observed, model = payload.get("observed", {}), payload.get("model", {})
                edges = np.asarray(observed.get("radial_edges_mas", []), float)
                x = (edges[:-1]+edges[1:])/2
                data, prediction = observed.get("qphi_profile", []), model.get("qphi_profile", [])
                ylabel = "Normalised annular Qphi contribution"
            else:
                x = payload.get("radius_mas", [])
                data, prediction = payload.get("observed_profile_jybeam", []), payload.get("model_profile_jybeam", [])
                ylabel = "Annular mean intensity [Jy/beam]"
            if len(x) and len(x) == len(data) == len(prediction):
                line, = ax.plot(x, prediction, label=row.history)
                ax.plot(x, data, "--", color=line.get_color(), alpha=.55)
                ax.set(xlabel="Radius [mas]", ylabel=ylabel, title=band + ": solid model, dashed observation")
                drawn = True
    if not drawn:
        plt.close(fig)
        return None
    for ax in axes.flat:
        if ax.lines:
            ax.legend(fontsize=6)
    fig.tight_layout()
    return fig


def plot_tradeoffs(pool):
    import matplotlib.pyplot as plt
    if pool.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for history, group in pool.groupby("history", sort=False):
        pdi = group[PDI].sum(axis=1, min_count=3)
        interf = group[INTERFEROMETRY].sum(axis=1, min_count=4)
        axes[0].scatter(group.score_sed, pdi, alpha=.4, s=12, label=history)
        axes[1].scatter(interf, pdi, alpha=.4, s=12, label=history)
    for ax, xlabel in zip(axes, ("SED contribution", "Interferometry contribution")):
        ax.set(xlabel=xlabel, ylabel="PDI contribution")
        ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Already scored common-pool evaluations (one compatibility group)")
    fig.tight_layout()
    return fig


def final_configurations(evaluations):
    """One best observed realisation per configuration/history/final objective."""
    return (evaluations.loc[evaluations.usable & np.isclose(
        evaluations.budget, evaluations.configured_max_budget)]
        .sort_values("cost").drop_duplicates(["history", "objective_key", "parameter_hash"]))


def competitive_comparison(evaluations, competitive, pair_analysis, parameters=None):
    """Descriptive retained fractions, numeric distributions and paired correlations.

    Competitive rows are a nested subset, not an independent statistical sample.
    Quantiles describe the optimiser's sampled models, not posterior uncertainty.
    """
    final = final_configurations(evaluations)
    counts, distributions = [], []
    for (history, objective), full in final.groupby(["history", "objective_key"], sort=False):
        subset = competitive[(competitive.history == history) & (competitive.objective_key == objective)]
        subset = full[full.parameter_hash.isin(subset.parameter_hash)]
        counts.append({"history": history, "objective_key": objective, "products": full.products.iloc[0],
                       "n_all_final": len(full), "n_competitive": len(subset),
                       "retained_fraction": len(subset)/len(full), "best_cost": full.cost.min(),
                       "max_competitive_cost": subset.cost.max()})
        names = parameters if parameters is not None else [c[7:] for c in full if c.startswith("param__")]
        for name in names:
            column = "param__"+name
            if column not in full:
                continue
            row = {"history": history, "objective_key": objective, "parameter": name}
            for scope, sample in (("all_final", full), ("competitive", subset)):
                original = sample[column].dropna()
                values = pd.to_numeric(original, errors="coerce").replace([np.inf, -np.inf], np.nan)
                # Do not partially summarise nominal categories as numbers.
                values = values.dropna() if values.notna().all() else pd.Series(dtype=float)
                row[f"n_{scope}"] = len(values)
                for label, q in (("min", 0), ("q10", .1), ("median", .5), ("q90", .9), ("max", 1)):
                    row[f"{label}_{scope}"] = values.quantile(q) if len(values) else np.nan
            distributions.append(row)
    pairs = pair_analysis.copy()
    pairs["both_eligible"] = pairs.eligible_all_final & pairs.eligible_competitive
    for prefix in ("rho", "tau_b"):
        delta = pairs[f"{prefix}_competitive"]-pairs[f"{prefix}_all_final"]
        pairs[f"delta_{prefix}"] = delta.where(pairs.both_eligible)
        pairs[f"abs_delta_{prefix}"] = pairs[f"delta_{prefix}"].abs()
        pairs[f"sign_flip_{prefix}"] = ((pairs[f"{prefix}_competitive"] *
                                           pairs[f"{prefix}_all_final"]) < 0).where(pairs.both_eligible)
    pairs["max_abs_delta"] = pairs[["abs_delta_rho", "abs_delta_tau_b"]].max(axis=1)
    return (pd.DataFrame(counts, columns=["history", "objective_key", "products", "n_all_final",
                "n_competitive", "retained_fraction", "best_cost", "max_competitive_cost"]),
            pd.DataFrame(distributions, columns=["history", "objective_key", "parameter",
                *[f"{name}_{scope}" for scope in ("all_final", "competitive")
                  for name in ("n", "min", "q10", "median", "q90", "max")]]), pairs)


def threshold_sensitivity(evaluations, spaces, relative_allowances, absolute_allowance=0., **pair_options):
    """Hold absolute allowance fixed; recompute nested sets on one final pool.

    The absolute floor can make different relative allowances select identical
    sets. Effective thresholds and retained counts are returned explicitly.
    """
    allowances = sorted(set(float(v) for v in relative_allowances))
    if not allowances or not all(np.isfinite(v) and v >= 0 for v in allowances):
        raise ValueError("Supply finite, nonnegative relative allowances.")
    if not np.isfinite(absolute_allowance) or absolute_allowance < 0:
        raise ValueError("Absolute allowance must be finite and nonnegative.")
    final = final_configurations(evaluations)
    count_rows, pair_tables = [], []
    for allowance in allowances:
        _, competitive = select_solutions(final, allowance, absolute_allowance)
        for (history, objective), full in final.groupby(["history", "objective_key"], sort=False):
            subset = competitive[(competitive.history == history) & (competitive.objective_key == objective)]
            best = full.cost.min()
            count_rows.append({"history": history, "objective_key": objective,
                               "relative_allowance": allowance, "absolute_allowance": absolute_allowance,
                               "cost_threshold": best+max(abs(best)*allowance, absolute_allowance),
                               "n_all_final": len(full), "n_competitive": len(subset),
                               "retained_fraction": len(subset)/len(full)})
        pairs = parameter_pair_analysis(final, competitive, spaces, **pair_options)
        pairs["relative_allowance"] = allowance
        pairs["absolute_allowance"] = absolute_allowance
        pair_tables.append(pairs)
    return pd.DataFrame(count_rows), pd.concat(pair_tables, ignore_index=True)


def plot_parameter_distributions(evaluations, competitive, history, objective, parameters, log_parameters=()):
    """ECDFs compare distributions without making the larger sample look denser."""
    import matplotlib.pyplot as plt
    full = final_configurations(evaluations)
    full = full[(full.history == history) & (full.objective_key == objective)]
    subset = competitive[(competitive.history == history) & (competitive.objective_key == objective)]
    subset = full[full.parameter_hash.isin(subset.parameter_hash)]
    if full.empty or not parameters:
        return None
    fig, axes = plt.subplots(math.ceil(len(parameters)/2), 2,
                             figsize=(12, 3.5*math.ceil(len(parameters)/2)), squeeze=False)
    for ax, name in zip(axes.flat, parameters):
        column = "param__"+name
        for label, sample, color in (("All final", full, "#888888"), ("Competitive", subset, "#2166ac")):
            raw = sample[column].dropna()
            values = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.empty or not values.notna().all():
                continue
            values = np.sort(values.to_numpy())
            ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", color=color,
                    label=f"{label} (n={len(values)})")
        if name in log_parameters:
            values = pd.to_numeric(full[column], errors="coerce").dropna()
            if len(values) and values.gt(0).all():
                ax.set_xscale("log")
        ax.set(xlabel=parameter_label(name), ylabel="Empirical cumulative fraction", ylim=(0, 1.02))
        if ax.lines:
            ax.legend(fontsize=7)
    for ax in axes.flat[len(parameters):]:
        ax.set_visible(False)
    fig.suptitle(f"Sampled parameter distributions: {history}\nObjective {objective}")
    fig.tight_layout()
    return fig


def plot_threshold_sensitivity(selected, sensitivity):
    """Use fixed selected pairs; mask coefficients failing eligibility at each threshold."""
    import matplotlib.pyplot as plt
    if selected.empty:
        return None
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3.2*len(selected)), squeeze=False)
    for (left, right), (_, pair) in zip(axes, selected.iterrows()):
        rows = sensitivity[(sensitivity.history == pair.history) &
                           (sensitivity.objective_key == pair.objective_key) &
                           (sensitivity.parameter_x == pair.parameter_x) &
                           (sensitivity.parameter_y == pair.parameter_y)].sort_values("relative_allowance")
        for prefix, label, color in (("rho", "Spearman", "#2166ac"), ("tau_b", "Kendall tau-b", "#b2182b")):
            values = rows[f"{prefix}_competitive"].where(rows.eligible_competitive)
            left.plot(rows.relative_allowance, values, 'o-', color=color, label=label)
            baseline = rows[f"{prefix}_all_final"].where(rows.eligible_all_final).dropna()
            if len(baseline):
                left.axhline(baseline.iloc[0], color=color, linestyle='--', alpha=.6)
        left.set(xlabel="Relative cost allowance", ylabel="Rank correlation", ylim=(-1.05, 1.05),
                 title=f"{parameter_label(pair.parameter_x)} / {parameter_label(pair.parameter_y)}")
        left.legend(fontsize=7)
        right.plot(rows.relative_allowance, rows.n_competitive, 'o-', label="Complete paired models")
        right.set(xlabel="Relative cost allowance", ylabel="Paired sample size")
        right.legend(fontsize=7)
    fig.suptitle(f"Threshold sensitivity: {selected.history.iloc[0]}\n"
                 "Dashed = all-final reference; gaps = ineligible; absolute allowance held fixed")
    fig.tight_layout()
    return fig

