"""Scientific bookkeeping checks for the saved-run summary notebook."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/post-processing/run_summary_utils.py"
sys.path.insert(0, str(MODULE_PATH.parent))
spec = importlib.util.spec_from_file_location("run_summary_utils", MODULE_PATH)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def full_info(corrected=False, weight=1.0):
    info = {"sed": {"chi2_reduced": 2.0}, "vis2_1perband": {}, "pdi": {},
            "alma": {"loss": 3.0, "comparison": {"observed_flux_jy": 1.0}},
            "fidelity": {"stage": "all", "products": ["sed", "vis2_1perband", "pdi_V", "pdi_I", "pdi_H", "alma"], "budget": 3}}
    for instrument in ("pionier", "gravity", "matisse_l", "matisse_n"):
        info["vis2_1perband"][instrument] = {"chi2_reduced": 1.0, "num_points": 10}
    for band in "VIH":
        info["pdi"][band] = {
            "loss": 3*weight, "model_unresolved_corrected": corrected,
            "unresolved_correction_radius_px": 3 if corrected else None,
            "constraints": {"fraction": weight, "radial": weight, "quadrant": weight,
                            "loss_definition": "test", "error_model": "none",
                            "observed": {"positive_qphi_over_i": 2, "qphi_profile": [2],
                                         "radial_edges_mas": [0, 25], "quadrants": [2]},
                            "model": {"positive_qphi_over_i": 1, "qphi_profile": [1], "quadrants": [1]}}}
    return info


def trial(config_id=1, budget=3, cost=18, info=None, status=1):
    return {"config_id": config_id, "budget": budget, "cost": cost,
            "seed": 7, "instance": None, "status": status, "time": 2,
            "additional_info": full_info() if info is None else info}


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, name="run", rows=None, seed="111", configs=None, max_budget=3):
        run = self.root / name
        history = run / "optimization" / seed
        history.mkdir(parents=True, exist_ok=True)
        cfg = configs or {"1": {"mass": 1.0}, "2": {"mass": 2.0}}
        (history / "runhistory.json").write_text(json.dumps({"configs": cfg, "data": rows or [trial()]}))
        (history / "scenario.json").write_text(json.dumps({"max_budget": max_budget, "seed": int(seed)}))
        (history / "configspace.json").write_text(json.dumps({"hyperparameters": [{"name": "mass", "type": "uniform_float", "lower": .1, "upper": 10}]}))
        (run / "simulation.para").write_text("same physical template")
        return run, history

    def test_penalties_and_failures_are_excluded_even_with_success_status(self):
        rows = [trial(), trial(cost=1e99), trial(cost=0.1, info={"failure": {"reason": "killed"}}),
                trial(cost=float("inf")), trial(status=0)]
        run, _ = self.make_run(rows=rows)
        frame, *_ = summary.load_runs({"test": run})
        self.assertEqual(frame.usable.sum(), 1)
        self.assertEqual(set(frame.exclusion), {"usable", "penalty_cost", "recorded_failure", "non_finite_cost", "unfinished"})

    def test_readable_objective_labels_preserve_product_distinctions(self):
        self.assertEqual(summary.objective_label(["alma", "sed"]), "SED+ALMA")
        self.assertEqual(summary.objective_label(["sed", "pdi_I", "pdi_V", "pdi_H"]), "SED+PDI(V,I,H)")
        self.assertNotEqual(summary.objective_label(["vis2_chromatic"]),
                            summary.objective_label(["vis2_1perband"]))
        self.assertNotEqual(summary.objective_label(["pdi_V"]), summary.objective_label(["pdi_I"]))
        self.assertNotEqual(summary.objective_label(["x+y"]), summary.objective_label(["x", "y"]))
        run, _ = self.make_run()
        frame, *_ = summary.load_runs({"test": run})
        self.assertEqual(frame.iloc[0].objective_key, "SED+INT(band)+PDI(V,I,H)+ALMA")

    def test_cost_audit_checks_completeness_not_just_numeric_sum(self):
        info = full_info()
        self.assertEqual(summary.audit_cost(info, 18)["cost_audit_status"], "match")
        mismatch = summary.audit_cost(info, 17)
        self.assertEqual(mismatch["cost_difference"], 1)
        self.assertEqual(mismatch["cost_audit_status"], "mismatch")
        del info["pdi"]["V"]["loss"]
        partial = summary.audit_cost(info, 15)
        self.assertEqual(partial["available_score_sum"], 15)
        self.assertFalse(partial["cost_terms_match"])
        self.assertTrue(np.isnan(partial["reconstructed_cost"]))
        self.assertIn("score_pdi_V", partial["missing_score_terms"])
        info["fidelity"]["products"].append("unknown_observation")
        self.assertEqual(summary.audit_cost(info, 15)["unknown_products"], "unknown_observation")

    def test_cost_audit_zero_infinity_and_inferred_products(self):
        zero = summary.audit_cost({"sed": {"chi2_reduced": 0}}, 0)
        self.assertTrue(zero["cost_terms_match"])
        self.assertTrue(np.isnan(zero["cost_relative_difference"]))
        self.assertEqual(zero["products_source"], "inferred")
        info = full_info()
        info["alma"]["loss"] = float("inf")
        self.assertFalse(summary.audit_cost(info, 15)["cost_terms_complete"])

    def test_numerical_fidelity_partitions_common_pool(self):
        first, second = full_info(), full_info()
        first["fidelity"]["npix"] = 101
        second["fidelity"]["npix"] = 201
        run, _ = self.make_run(rows=[trial(info=first), trial(info=second)])
        frame, *_ = summary.load_runs({"test": run})
        pool, _ = summary.common_pool_comparison(frame)
        self.assertEqual(pool.pool_group.nunique(), 2)

    def test_final_stage_is_not_replaced_with_a_successful_lower_stage(self):
        sed = {"sed": {"chi2_reduced": 2}, "fidelity": {"stage": "sed", "products": ["sed"]}}
        run, _ = self.make_run(rows=[trial(budget=1, cost=2, info=sed), trial(cost=1e99)])
        frame, *_ = summary.load_runs({"test": run})
        best, competitive = summary.select_solutions(frame)
        self.assertTrue(best.empty and competitive.empty)

    def test_distinct_model_selection_keeps_repeats_in_evaluation_table(self):
        run, _ = self.make_run(rows=[trial(cost=18), trial(cost=19), trial(config_id=2, cost=20)])
        frame, *_ = summary.load_runs({"test": run})
        best, competitive = summary.select_solutions(frame, relative_tolerance=.1)
        self.assertEqual(len(frame), 3)
        self.assertEqual(best.iloc[0].cost, 18)
        self.assertEqual(len(competitive), 1)

    def test_discovery_deduplicates_overlapping_inputs_and_accepts_json(self):
        run, history = self.make_run()
        found = summary.discover_histories({"parent": self.root, "run": run, "file": history/"runhistory.json"})
        self.assertEqual(len(found), 1)
        self.assertEqual(len(summary.discover_histories({"file": history/"runhistory.json"})), 1)

    def test_missing_observables_are_nan_and_do_not_enter_common_pool(self):
        sed = {"sed": {"chi2_reduced": 2}, "fidelity": {"products": ["sed"]}}
        run, _ = self.make_run(rows=[trial(cost=2, info=sed)])
        frame, *_ = summary.load_runs({"test": run})
        self.assertTrue(frame.score_pdi_I.isna().all())
        pool, winners = summary.common_pool_comparison(frame)
        self.assertTrue(pool.empty and winners.empty)

    def test_correction_and_weights_partition_common_pools(self):
        rows = [trial(), trial(config_id=2, info=full_info(True)),
                trial(config_id=2, info=full_info(False, 2), cost=27)]
        run, _ = self.make_run(rows=rows)
        frame, *_ = summary.load_runs({"test": run})
        self.assertTrue(frame.cost_terms_match.all())
        pool, winners = summary.common_pool_comparison(frame)
        self.assertEqual(pool.pool_group.nunique(), 3)
        self.assertEqual(len(winners), 3*len(summary.SUBSETS))
        self.assertEqual(winners[winners.objective == "SED"].subset_score.tolist(), [2, 2, 2])

    def test_pairing_uses_parameters_rather_than_local_config_ids(self):
        rows = []
        cfg = {}
        for i, cost in enumerate([1, 2, 3], 1):
            cfg[str(i)] = {"mass": i}
            cfg[str(i+3)] = {"mass": i}
            rows += [trial(i, 1, cost), trial(i+3, 3, 4-cost)]
        run, _ = self.make_run(rows=rows, configs=cfg)
        frame, *_ = summary.load_runs({"test": run})
        result = summary.stage_correlations(frame).iloc[0]
        self.assertEqual(result.paired_configurations, 3)
        self.assertAlmostEqual(result.spearman_rho, -1)

    def test_saved_trial_matching_checks_parameters_and_budget(self):
        run, _ = self.make_run(configs={"1": {"mass": 1.0, "height": 2.0, "height_tie": "height"}})
        for suffix, mass in [("", 9.0), ("_001", 1.0)]:
            folder = run / "trials" / ("config_1_budget_3.00"+suffix)
            folder.mkdir(parents=True)
            (folder / "config_used.json").write_text(json.dumps({
                "cfg": {"mass": mass, "height": 2.0, "height_tie": 2.0},
                "fidelity": full_info()["fidelity"]}))
        frame, _, records, _ = summary.load_runs({"test": run})
        matches = summary.resolve_model_files(frame.iloc[0], records)
        self.assertEqual(len(matches), 1)
        self.assertTrue(matches[0]["folder"].name.endswith("_001"))

    def pair_samples(self, count=30, history="sample", sign=1):
        x = np.arange(1, count+1, dtype=float)
        return pd.DataFrame({"history": history, "objective_key": "sed", "products": "sed",
                             "usable": True, "budget": 1., "configured_max_budget": 1.,
                             "cost": x, "parameter_hash": [f"{history}_{i}" for i in range(count)],
                             "param__x": x, "param__y": sign*x*x})

    def test_competitive_comparison_uses_nested_distinct_final_models(self):
        full = self.pair_samples()
        full["param__y"] = np.r_[np.arange(10)[::-1], np.arange(10, 30)]
        _, competitive = summary.select_solutions(full, relative_tolerance=9)
        pairs = summary.parameter_pair_analysis(full, competitive, {}, min_models=3)
        counts, parameters, comparison = summary.competitive_comparison(
            pd.concat([full, full.iloc[:2]], ignore_index=True), competitive, pairs)
        self.assertEqual(counts.iloc[0].n_all_final, 30)
        self.assertEqual(counts.iloc[0].n_competitive, 10)
        self.assertAlmostEqual(counts.iloc[0].retained_fraction, 1/3)
        row = comparison.iloc[0]
        self.assertTrue(row.both_eligible)
        self.assertTrue(row.sign_flip_rho)
        self.assertAlmostEqual(row.delta_rho, row.rho_competitive-row.rho_all_final)
        self.assertEqual(parameters.set_index("parameter").loc["x", "max_competitive"], 10)
        sparse = summary.parameter_pair_analysis(full, competitive, {}, min_models=20)
        _, _, audit = summary.competitive_comparison(full, competitive, sparse)
        self.assertTrue(audit.delta_rho.isna().all())

    def test_threshold_sensitivity_preserves_absolute_floor_and_eligibility(self):
        full = self.pair_samples()
        counts, pairs = summary.threshold_sensitivity(full, {}, [0, 1, 9, 29],
                                                     absolute_allowance=4, min_models=6)
        self.assertEqual(counts.n_competitive.tolist(), [5, 5, 10, 30])
        self.assertEqual(counts.cost_threshold.tolist(), [5, 5, 10, 30])
        self.assertFalse(pairs.iloc[0].eligible_competitive)
        self.assertTrue(pairs.iloc[-1].eligible_competitive)
        self.assertAlmostEqual(pairs.iloc[-1].rho_competitive, pairs.iloc[-1].rho_all_final)
        with self.assertRaises(ValueError):
            summary.threshold_sensitivity(full, {}, [float("nan")])

    def test_comparison_plots_and_empty_final_pool(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        full = self.pair_samples()
        _, competitive = summary.select_solutions(full, relative_tolerance=19)
        pairs = summary.parameter_pair_analysis(full, competitive, {})
        _, sensitivity = summary.threshold_sensitivity(full, {}, [.1, 19, 29])
        fig = summary.plot_threshold_sensitivity(pairs, sensitivity)
        self.assertTrue(np.isnan(fig.axes[0].lines[0].get_ydata()[0]))
        plt.close(fig)
        fig = summary.plot_parameter_distributions(full, competitive, "sample", "sed", ["x", "y"])
        self.assertEqual(len(fig.axes[0].lines), 2)
        self.assertEqual(fig.axes[0].lines[0].get_ydata()[-1], 1)
        plt.close(fig)
        full["configured_max_budget"] = 3
        pairs = summary.parameter_pair_analysis(full, competitive.iloc[:0], {})
        counts, parameters, comparison = summary.competitive_comparison(full, competitive.iloc[:0], pairs)
        self.assertTrue(counts.empty and parameters.empty and comparison.empty)
        self.assertIn("history", parameters)

    def test_pair_associations_are_within_history_with_full_sample_control(self):
        frame = pd.concat([self.pair_samples(history="positive"), self.pair_samples(history="negative", sign=-1)], ignore_index=True)
        competitive = frame.groupby("history", sort=False).head(20)
        pairs = summary.parameter_pair_analysis(frame, competitive, {}, min_models=20)
        self.assertEqual(len(pairs), 2)
        indexed = pairs.set_index("history")
        self.assertAlmostEqual(indexed.loc["positive", "rho_competitive"], 1)
        self.assertAlmostEqual(indexed.loc["negative", "rho_competitive"], -1)
        self.assertEqual(indexed.loc["positive", "n_all_final"], 30)
        self.assertEqual(indexed.loc["positive", "n_competitive"], 20)
        chosen = summary.select_parameter_pairs(pairs)
        self.assertEqual(len(chosen), 2)
        self.assertTrue(chosen.abs_rho.eq(1).all())

    def test_small_or_repeated_samples_do_not_qualify_for_pair_ranking(self):
        frame = self.pair_samples(count=10)
        repeated = pd.concat([frame]*4, ignore_index=True)
        pairs = summary.parameter_pair_analysis(repeated, repeated, {})
        self.assertEqual(pairs.iloc[0].n_competitive, 10)
        self.assertEqual(pairs.iloc[0].reason_competitive, "too_few_models")
        self.assertTrue(summary.select_parameter_pairs(pairs).empty)

    def test_fixed_tied_nominal_and_periodic_parameters_are_not_ranked(self):
        frame = self.pair_samples()
        frame["param__fixed"] = 4
        frame["param__tied"] = "x"
        frame["param__material"] = ["silicate", "iron"]*15
        frame["param__disk_pa"] = np.arange(30)
        definitions = [{"name": name, "type": "uniform_float"} for name in ("x", "y", "disk_pa")]
        definitions += [{"name": "fixed", "type": "constant"}, {"name": "tied", "type": "constant"},
                        {"name": "material", "type": "categorical"}]
        spaces = {"sample": {"hyperparameters": definitions}}
        pairs = summary.parameter_pair_analysis(frame, frame, spaces)
        self.assertFalse(pairs.parameter_x.isin(["fixed", "tied"]).any())
        self.assertFalse(pairs.parameter_y.isin(["fixed", "tied"]).any())
        periodic = pairs[pairs.parameter_x.eq("disk_pa") | pairs.parameter_y.eq("disk_pa")]
        self.assertTrue(periodic.reason_competitive.eq("periodic_parameter_requires_circular_analysis").all())
        nominal = pairs[(pairs.parameter_x.eq("material") | pairs.parameter_y.eq("material"))
                        & ~pairs.parameter_x.eq("disk_pa") & ~pairs.parameter_y.eq("disk_pa")]
        self.assertTrue(nominal.reason_competitive.eq("nonnumeric_or_nonfinite_values").all())
        self.assertEqual(len(summary.select_parameter_pairs(pairs)), 1)

    def test_direct_configspace_constraints_are_flagged_and_not_auto_selected(self):
        frame = self.pair_samples()
        spaces = {"sample": {"forbiddens": [{"type": "RELATION_LT", "left": "x", "right": "y"}]}}
        pairs = summary.parameter_pair_analysis(frame, frame, spaces)
        self.assertTrue(pairs.iloc[0].direct_configspace_constraint)
        self.assertTrue(summary.select_parameter_pairs(pairs).empty)
        self.assertEqual(len(summary.select_parameter_pairs(pairs, include_constrained=True)), 1)

    def test_pair_analysis_handles_empty_final_stage_and_sparse_discrete_axes(self):
        frame = self.pair_samples()
        frame["param__y"] = [0, 1]*15
        pairs = summary.parameter_pair_analysis(frame, frame, {})
        self.assertEqual(pairs.iloc[0].reason_competitive, "too_few_distinct_values")
        frame["configured_max_budget"] = 3
        empty_pairs = summary.parameter_pair_analysis(frame, frame, {})
        self.assertTrue(summary.select_parameter_pairs(empty_pairs).empty)

    def test_correlation_strength_bands_are_symmetric_at_boundaries(self):
        for magnitude, label in [(0, "Negligible"), (.199, "Negligible"), (.2, "Weak"),
                                 (.4, "Moderate"), (.6, "Strong"), (.8, "Very strong"), (1, "Very strong")]:
            self.assertEqual(summary.correlation_strength(magnitude), label)
            self.assertEqual(summary.correlation_strength(-magnitude), label)
            self.assertEqual(summary.correlation_band(magnitude)+summary.correlation_band(-magnitude), 8)
        self.assertEqual(summary.correlation_strength(np.nan), "Unavailable")
        self.assertEqual(summary.correlation_strength(.8-1e-16), "Very strong")
        with self.assertRaises(ValueError):
            summary.correlation_band(1.1)

    def test_heatmap_matrices_preserve_signed_coefficients_masks_and_constraints(self):
        frame = self.pair_samples(sign=-1)
        spaces = {"sample": {"forbiddens": [{"type": "RELATION_LT", "left": "x", "right": "y"}]}}
        pairs = summary.parameter_pair_analysis(frame, frame, spaces)
        rho, eligible, constrained = summary.parameter_correlation_matrices(pairs)
        self.assertAlmostEqual(rho.loc["x", "y"], -1)
        self.assertEqual(rho.loc["x", "y"], rho.loc["y", "x"])
        self.assertTrue(eligible.loc["x", "y"])
        self.assertTrue(constrained.loc["y", "x"])
        self.assertTrue(np.isnan(rho.loc["x", "x"]))
        self.assertFalse(eligible.loc["x", "x"])
        sparse = summary.parameter_pair_analysis(frame, frame.head(5), spaces)
        raw, mask, _ = summary.parameter_correlation_matrices(sparse)
        self.assertAlmostEqual(raw.loc["x", "y"], -1)
        self.assertFalse(mask.loc["x", "y"])

    def test_heatmap_rejects_pooling_multiple_modelling_histories(self):
        frame = pd.concat([self.pair_samples(history="one"), self.pair_samples(history="two")], ignore_index=True)
        pairs = summary.parameter_pair_analysis(frame, frame, {})
        with self.assertRaisesRegex(ValueError, "exactly one history"):
            summary.parameter_correlation_matrices(pairs)

    def test_heatmap_masks_unavailable_cells_instead_of_drawing_zero_correlations(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frame = self.pair_samples()
        pairs = summary.parameter_pair_analysis(frame, frame.head(5), {})
        fig = summary.plot_parameter_correlation_heatmap(pairs)
        image = fig.axes[0].images[0]
        self.assertTrue(np.ma.getmaskarray(image.get_array()).all())
        self.assertIn("NA", [text.get_text() for text in fig.axes[0].texts])
        plt.close(fig)

    def test_kendall_tau_b_accounts_for_ties_on_both_axes(self):
        frame = self.pair_samples(count=4)
        frame["param__x"] = [1, 1, 2, 3]
        frame["param__y"] = [1, 2, 2, 3]
        pairs = summary.parameter_pair_analysis(frame, frame, {}, min_models=3, min_unique=2)
        row = pairs.iloc[0]
        # Four concordant pairs, no discordance, one x-only tie and one y-only
        # tie: tau-b = 4/sqrt((4+1)*(4+1)) = 0.8; tau-a would be 4/6.
        self.assertAlmostEqual(row.tau_b_all_final, .8)
        self.assertAlmostEqual(row.tau_b_competitive, .8)
        self.assertAlmostEqual(row.rho_all_final, 5/6)
        self.assertNotAlmostEqual(row.tau_b_all_final, 4/6)
        self.assertTrue(row.eligible_competitive)

    def test_kendall_and_spearman_use_the_same_complete_rows_and_exclusions(self):
        frame = self.pair_samples(count=5)
        frame["param__x"] = [1, 1, 2, 3, np.nan]
        frame["param__y"] = [1, 2, 2, 3, -100]
        row = summary.parameter_pair_analysis(frame, frame, {}, min_models=3, min_unique=2).iloc[0]
        self.assertEqual(row.n_all_final, 4)
        self.assertAlmostEqual(row.tau_b_all_final, .8)
        self.assertAlmostEqual(row.rho_all_final, 5/6)
        frame["param__disk_pa"] = np.arange(5)
        pairs = summary.parameter_pair_analysis(frame, frame, {}, min_models=3, min_unique=2)
        periodic = pairs[pairs.parameter_x.eq("disk_pa") | pairs.parameter_y.eq("disk_pa")]
        self.assertTrue(periodic.tau_b_all_final.isna().all())
        self.assertTrue(periodic.rho_all_final.isna().all())

    def test_kendall_matrices_and_heatmap_label_the_metric_and_keep_the_pair_mask(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frame = self.pair_samples(count=4)
        frame["param__x"] = [1, 1, 2, 3]
        frame["param__y"] = [1, 2, 2, 3]
        pairs = summary.parameter_pair_analysis(frame, frame, {}, min_models=3, min_unique=2)
        spearman, sp_mask, _ = summary.parameter_correlation_matrices(pairs)
        kendall, tau_mask, _ = summary.parameter_correlation_matrices(pairs, metric="kendall_tau_b")
        self.assertAlmostEqual(kendall.loc["x", "y"], .8)
        self.assertAlmostEqual(spearman.loc["x", "y"], 5/6)
        pd.testing.assert_frame_equal(sp_mask, tau_mask)
        fig = summary.plot_parameter_correlation_heatmap(pairs, metric="kendall_tau_b")
        self.assertIn("Kendall Tau-b", fig.axes[0].get_title())
        self.assertIn("+0.80", [text.get_text() for text in fig.axes[0].texts])
        plt.close(fig)
        with self.assertRaises(ValueError):
            summary.parameter_correlation_matrices(pairs, metric="unknown")

    def test_both_association_metrics_render_identical_parameters_and_samples(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        frame = self.pair_samples()
        competitive = frame.head(20)
        pairs = summary.parameter_pair_analysis(frame, competitive, {})
        selected = summary.select_parameter_pairs(pairs, scope="all_final")
        self.assertEqual(len(selected), 1)
        first = summary.plot_selected_parameter_pairs(selected, frame, competitive)
        second = summary.plot_selected_parameter_pairs(selected, frame, competitive, metric="kendall_tau_b")
        self.assertEqual(first.axes[0].get_xlabel(), second.axes[0].get_xlabel())
        self.assertEqual(first.axes[0].get_ylabel(), second.axes[0].get_ylabel())
        for a, b in zip(first.axes[0].collections, second.axes[0].collections):
            np.testing.assert_allclose(a.get_offsets(), b.get_offsets())
        self.assertIn("Kendall Tau-b", second._suptitle.get_text())
        self.assertIn("τ-b=", second.axes[0].get_title())
        plt.close(first)
        plt.close(second)

    def test_kendall_tau_b_signed_agreement_and_insufficient_support(self):
        for sign in (1, -1):
            frame = self.pair_samples(sign=sign)
            pairs = summary.parameter_pair_analysis(frame, frame.head(5), {})
            row = pairs.iloc[0]
            self.assertAlmostEqual(row.tau_b_all_final, sign)
            self.assertAlmostEqual(row.tau_b_competitive, sign)
            self.assertFalse(row.eligible_competitive)
            raw, eligible, _ = summary.parameter_correlation_matrices(pairs, metric="kendall_tau_b")
            self.assertAlmostEqual(raw.loc["x", "y"], sign)
            self.assertFalse(eligible.loc["x", "y"])


if __name__ == "__main__":
    unittest.main()
