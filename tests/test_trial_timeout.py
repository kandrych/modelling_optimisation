"""Real process timeouts and small synthetic SMAC runs; no MCFOST simulations."""
import argparse
import ast
import json
import pickle
from functools import partial
from dataclasses import replace
from unittest.mock import patch
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from ConfigSpace import Configuration, ConfigurationSpace, Float
from smac import Scenario
from smac.facade.multi_fidelity_facade import MultiFidelityFacade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.runhistory import RunHistory, StatusType, TrialInfo, TrialValue
from smac.runhistory.dataclasses import InstanceSeedBudgetKey

from lib.obriy_trial_timeout import (
    TrialTimeoutRunner, TimeoutAwareEncoder, TimeoutAwareHyperband,
    UNSCORED_TIMEOUT_COST, positive_seconds, timeout_cost,
)

# Import just the existing directory allocator, without importing observational
# data or all of the plotting dependencies from the driver.
ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "SMAC3.py").read_text())
factory = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "make_unique_config_trial_dir")
exec(compile(ast.Module(body=[factory], type_ignores=[]), "<trial allocator>", "exec"))


def slow_model(config, seed, budget):
    directory = Path(config._obriy_trial_dir)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (directory / "child.pid").write_text(str(child.pid))
    # Simulate a finished MCFOST call followed by expensive scoring: the same
    # deadline must still stop this whole objective and its child process.
    (directory / "model.para").write_text("synthetic model\n")
    time.sleep(60)
    return 3.0


def fast_model(config, seed, budget, *, offset=0.0):
    return float(config["x"] ** 2 + offset), {"measured": True}


class TimeoutTests(unittest.TestCase):
    def test_script_entrypoint_uses_importable_targets(self):
        # Exercise the real __main__ entry point without launching optimisation.
        # Dask's cloudpickle round trip must leave targets standard-picklable
        # for the next boundary: pynisher's spawned process.
        probe = '''
import runpy
import pickle
import cloudpickle
from functools import partial
import SMAC3
def check_main():
    for function in (SMAC3.objective, SMAC3.final_model):
        assert function.__module__ == "SMAC3"
        target = cloudpickle.loads(cloudpickle.dumps(partial(function)))
        assert pickle.loads(pickle.dumps(target)).func is function
    print("canonical entrypoint checked")
SMAC3.main = check_main
runpy.run_path("SMAC3.py", run_name="__main__")
'''
        result = subprocess.run([sys.executable, "-c", probe], cwd=ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("canonical entrypoint checked", result.stdout)

    def test_driver_targets_are_picklable(self):
        from SMAC3 import objective, final_model

        targets = [
            partial(objective, data_arg={}, scratch_root=Path("trials"),
                    args=argparse.Namespace()),
            partial(final_model, fidelity_result={}, par_path=Path("model.para"),
                    results_dir=Path("results"), data_arg={},
                    args=argparse.Namespace(), incumbent={}),
        ]
        for target in targets:
            restored = pickle.loads(pickle.dumps(target))
            self.assertIs(restored.func, target.func)
            self.assertEqual(restored.keywords, target.keywords)

    def test_partial_targets_with_explicit_spawn(self):
        from pynisher import limit

        # Allow child imports to finish even on a busy shared filesystem.
        self.scenario = replace(self.scenario, trial_walltime_limit=10)
        with patch("smac.runner.target_function_runner.limit",
                   partial(limit, context="spawn")):
            _, value, record, _ = self.run_model(
                function=partial(fast_model, offset=2.0))
            self.assertEqual(value.status, StatusType.SUCCESS)
            self.assertEqual(value.cost, 2.25)
            self.assertEqual(record["status"], "success")
            config = Configuration(self.cs, values={"x": 0.5})
            config.config_id = 8
            _, timed_out = self.runner(function=partial(slow_model)).run_wrapper(
                TrialInfo(config, seed=1, budget=1.0))
            self.assertEqual(timed_out.status, StatusType.TIMEOUT)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trials = self.root / "trials"
        self.trials.mkdir()
        self.cs = ConfigurationSpace(seed=11)
        self.cs.add(Float("x", (0.0, 1.0)))
        self.scenario = Scenario(self.cs, output_directory=self.root / "optimization",
                                 n_trials=3, min_budget=1, max_budget=1,
                                 trial_walltime_limit=1, seed=11)

    def runner(self, penalty=None, function=slow_model):
        return TrialTimeoutRunner(self.scenario, function, trial_root=self.trials,
                                  directory_factory=make_unique_config_trial_dir,
                                  timeout_penalty=penalty)

    def run_model(self, penalty=None, function=slow_model):
        config = Configuration(self.cs, values={"x": 0.5})
        config.config_id = 7
        _, value = self.runner(penalty, function).run_wrapper(TrialInfo(config, seed=1, budget=1.0))
        record_path = next(self.trials.glob("*/trial_status.json"))
        record = json.loads(record_path.read_text())
        self.assertFalse(hasattr(config, "_obriy_trial_dir"))
        return config, value, record, record_path.parent

    def test_none_timeout_stops_process_tree_and_preserves_configuration(self):
        config, value, record, directory = self.run_model()
        self.assertEqual(value.status, StatusType.TIMEOUT)
        self.assertIsNone(record["cost"])
        self.assertIsNone(value.additional_info["trial_timeout"]["cost"])
        self.assertEqual(record["configuration"], dict(config))
        self.assertEqual(record["status"], "timeout")
        self.assertIn("exceeded", record["message"])
        self.assertGreaterEqual(value.time, 1.0)
        self.assertLess(value.time, 12.0)
        self.assertTrue((directory / "model.para").exists())
        pid = int((directory / "child.pid").read_text())
        try:
            process = psutil.Process(pid)
            self.assertIn(process.status(), (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD))
        except psutil.NoSuchProcess:
            pass

    def test_penalty_timeout_is_explicit_and_success_is_unchanged(self):
        _, value, record, _ = self.run_model(1e8)
        self.assertEqual(value.status, StatusType.TIMEOUT)
        self.assertEqual(value.cost, 1e8)
        self.assertEqual(record["cost"], 1e8)
        config = Configuration(self.cs, values={"x": 0.5})
        config.config_id = 8
        _, result = self.runner(1e8, fast_model).run_wrapper(TrialInfo(config, seed=1, budget=1.0))
        self.assertEqual(result.status, StatusType.SUCCESS)
        self.assertEqual(result.cost, 0.25)

    def test_encoder_ignores_none_but_learns_explicit_penalty(self):
        history = RunHistory()
        for x, status, cost, policy in [
            (0.1, StatusType.SUCCESS, 1.0, None),
            (0.2, StatusType.SUCCESS, 2.0, None),
            (0.3, StatusType.TIMEOUT, UNSCORED_TIMEOUT_COST, "none"),
            (0.4, StatusType.TIMEOUT, 1e8, "penalty"),
        ]:
            info = {} if policy is None else {"trial_timeout": {"policy": policy}}
            history.add(Configuration(self.cs, values={"x": x}), cost=cost,
                        status=status, seed=1, budget=1.0, additional_info=info)
        encoder = TimeoutAwareEncoder(self.scenario)
        encoder.runhistory = history
        X, y = encoder.transform([1.0])
        self.assertEqual(X.shape[0], 3)
        self.assertFalse(np.any(np.isclose(X[:, 0], 0.3)))
        self.assertTrue(np.any(np.isclose(X[:, 0], 0.4)))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertEqual(encoder.get_configurations([1.0]).shape[0], 3)

    def test_all_timeouts_continue_past_initial_design_without_an_incumbent(self):
        smac = MultiFidelityFacade(
            self.scenario, target_function=self.runner(), overwrite=True,
            initial_design=SobolInitialDesign(self.scenario, n_configs=1),
            runhistory_encoder=TimeoutAwareEncoder(self.scenario),
            intensifier=TimeoutAwareHyperband(self.scenario), logging_level=40)
        self.assertIsNone(smac.optimize())
        self.assertEqual(smac.runhistory.finished, 3)
        self.assertTrue(all(value.status == StatusType.TIMEOUT for value in smac.runhistory.values()))
        self.assertEqual(len(list(self.trials.glob("*/trial_status.json"))), 3)
        X, _ = smac._runhistory_encoder.transform([1.0])
        self.assertEqual(len(X), 0)

    def test_timeout_not_incumbent_or_promoted_even_when_penalty_is_low(self):
        scenario = Scenario(self.cs, output_directory=self.root / "multifidelity",
                            n_trials=6, min_budget=1, max_budget=3, seed=11)
        intensifier = TimeoutAwareHyperband(scenario, eta=3)
        smac = MultiFidelityFacade(scenario, fast_model, intensifier=intensifier,
                                  runhistory_encoder=TimeoutAwareEncoder(scenario),
                                  overwrite=True, logging_level=40)
        good = Configuration(self.cs, values={"x": 0.1})
        bad = Configuration(self.cs, values={"x": 0.9})
        smac.tell(TrialInfo(bad, seed=1, budget=1.0),
                  TrialValue(cost=0.01, status=StatusType.TIMEOUT,
                             additional_info={"trial_timeout": {"policy": "penalty"}}))
        self.assertIsNone(smac.intensifier.get_incumbent())
        smac.tell(TrialInfo(good, seed=1, budget=1.0), TrialValue(cost=10.0))
        self.assertEqual(smac.intensifier.get_incumbent(), good)
        intensifier.__post_init__()
        keys = [InstanceSeedBudgetKey(None, 1, 1.0)]
        self.assertEqual(intensifier._get_best_configs([good, bad], 0, 0, keys), [good])

    def test_cli_validation(self):
        self.assertEqual(positive_seconds("2700"), 2700)
        self.assertIsNone(timeout_cost("None"))
        self.assertEqual(timeout_cost("1e99"), 1e99)
        for value in ("0", "-1", "1.5", "nan"):
            with self.assertRaises(argparse.ArgumentTypeError):
                positive_seconds(value)
        for value in ("0", "-1", "inf", "nan", "invalid"):
            with self.assertRaises(argparse.ArgumentTypeError):
                timeout_cost(value)

    def test_csv_export_has_no_placeholder_loss_for_none_policy(self):
        from types import SimpleNamespace
        export_tree = ast.parse((ROOT / "lib/obriy_after_optimisation.py").read_text())
        function = next(node for node in export_tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "runhistory_to_df")
        namespace = {"pd": pd, "TrialValue": TrialValue}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<csv export>", "exec"), namespace)
        history = RunHistory()
        history.add(Configuration(self.cs, values={"x": 0.3}), cost=UNSCORED_TIMEOUT_COST,
                    status=StatusType.TIMEOUT, seed=1, budget=1.0,
                    additional_info={"trial_timeout": {"policy": "none", "cost": None}})
        frame = namespace["runhistory_to_df"](SimpleNamespace(runhistory=history), self.cs)
        self.assertTrue(pd.isna(frame.iloc[0]["cost"]))
        self.assertEqual(frame.iloc[0]["timeout_policy"], "none")

    def test_parallel_workers_report_timeout_and_release_slots(self):
        import dask
        from distributed import Client, LocalCluster
        from smac.runner.dask_runner import DaskParallelRunner

        with dask.config.set({"distributed.worker.daemon": False}):
            with LocalCluster(n_workers=2, threads_per_worker=1, processes=True,
                              dashboard_address=None, memory_limit=0) as cluster:
                with Client(cluster) as client:
                    runner = DaskParallelRunner(self.runner(), dask_client=client)
                    for config_id, x in enumerate([0.2, 0.7], 1):
                        config = Configuration(self.cs, values={"x": x})
                        config.config_id = config_id
                        runner.submit_trial(TrialInfo(config, seed=1, budget=1.0))
                    results = []
                    deadline = time.monotonic() + 30
                    while len(results) < 2 and time.monotonic() < deadline:
                        results.extend(runner.iter_results())
                        time.sleep(0.05)
                    self.assertEqual(len(results), 2)
                    self.assertTrue(all(value.status == StatusType.TIMEOUT for _, value in results))
                    self.assertEqual(runner.count_available_workers(), 2)
                    records = [json.loads(path.read_text())
                               for path in self.trials.glob("*/trial_status.json")]
                    self.assertEqual({record["config_id"] for record in records}, {1, 2})


if __name__ == "__main__":
    unittest.main()
