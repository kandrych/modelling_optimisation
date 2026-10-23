"""Whole-trial walltime limits and explicit timeout policies for SMAC 2.3.

SMAC requires numeric costs internally. For the ``none`` policy, trial_status.json
and additional_info contain a null cost; the numeric bookkeeping value is excluded
from surrogate training. Neither policy treats an unfinished trial as a fit.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from smac.intensifier.hyperband import Hyperband
from smac.runhistory import StatusType
from smac.runhistory.dataclasses import InstanceSeedBudgetKey, TrialKey
from smac.runhistory.encoder.log_scaled_encoder import RunHistoryLogScaledEncoder
from smac.runhistory.errors import NotEvaluatedError
from smac.runner.target_function_runner import TargetFunctionRunner

UNSCORED_TIMEOUT_COST = 1e99  # SMAC bookkeeping only, never a training target.


def positive_seconds(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Walltime must be a positive integer in seconds.") from error
    if seconds <= 0:
        raise argparse.ArgumentTypeError("Walltime must be a positive integer in seconds.")
    return seconds


def timeout_cost(value: str) -> float | None:
    if value.lower() == "none":
        return None
    try:
        cost = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Timeout cost must be 'none' or a finite positive number.") from error
    if not math.isfinite(cost) or cost <= 0:
        raise argparse.ArgumentTypeError("Timeout cost must be 'none' or a finite positive number.")
    return cost


def write_trial_status(trial_dir: Path, record: dict) -> None:
    """Publish a complete record, even when several Dask workers finish at once."""
    temporary = trial_dir / "trial_status.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(trial_dir / "trial_status.json")


def is_unscored_timeout(value) -> bool:
    timeout = value.additional_info.get("trial_timeout", {})
    return value.status == StatusType.TIMEOUT and timeout.get("policy") == "none"


class TrialTimeoutRunner(TargetFunctionRunner):
    """Use SMAC/pynisher's process-tree limit, retaining metadata in the parent.

    The limit covers parameter generation, all MCFOST commands, and scoring.
    File creation and final status recording happen outside the limited child,
    so even a timeout before model.para exists has an identifiable configuration.
    """

    def __init__(self, scenario, target_function, *, trial_root, directory_factory,
                 timeout_penalty=None, fidelity_settings=None):
        super().__init__(scenario, target_function, required_arguments=["seed", "budget"])
        self.trial_root = Path(trial_root)
        self.directory_factory = directory_factory
        self.timeout_penalty = timeout_penalty
        self.fidelity_settings = fidelity_settings

    @property
    def meta(self):
        return {**super().meta, "timeout_cost": self.timeout_penalty}

    def run(self, config, instance=None, budget=None, seed=None, **kwargs):
        from lib.obriy_fidelity import map_budget_to_fidelity

        config_id = config.config_id
        if config_id is None:
            raise ValueError("The trial must be registered with SMAC before execution.")
        trial_dir = self.directory_factory(self.trial_root, config_id, budget)
        # TargetFunctionRunner deep-copies Configuration before entering the child.
        # objective() consumes this attribute before converting the config to dict.
        config._obriy_trial_dir = str(trial_dir)
        record = {
            "status": "running", "config_id": config_id, "configuration": dict(config),
            "seed": seed, "instance": instance, "budget": budget,
            "trial_dir": str(trial_dir), "walltime_limit_seconds": self._algorithm_walltime_limit,
            "timeout_policy": "none" if self.timeout_penalty is None else "penalty",
            "timeout_cost": self.timeout_penalty, "cost": None,
        }
        fidelity = (map_budget_to_fidelity(budget, self.fidelity_settings)
                    if self.fidelity_settings is not None else None)
        if fidelity is not None:
            record["fidelity"] = fidelity
        write_trial_status(trial_dir, record)
        started = time.monotonic()
        try:
            status, cost, runtime, cpu_time, info = super().run(
                config, instance=instance, budget=budget, seed=seed, **kwargs)
        finally:
            del config._obriy_trial_dir
        # SMAC 2.3 reports zero runtime on a pynisher timeout. Preserve actual time,
        # including the short process cleanup interval after the deadline.
        runtime = time.monotonic() - started
        info = dict(info)
        info["trial_dir"] = str(trial_dir)
        if fidelity is not None:
            info["fidelity"] = fidelity
        if status == StatusType.TIMEOUT:
            cost = self.timeout_penalty if self.timeout_penalty is not None else UNSCORED_TIMEOUT_COST
            message = f"Trial exceeded its {self._algorithm_walltime_limit} s walltime limit."
            info["trial_timeout"] = {
                "policy": record["timeout_policy"], "cost": self.timeout_penalty,
                "walltime_limit_seconds": self._algorithm_walltime_limit,
                "elapsed_seconds": runtime, "message": message,
            }
            info["failure"] = {
                "code": "trial_walltime_exceeded", "stage": "objective",
                "reason": message, "trial_dir": str(trial_dir),
            }
            record.update(status="timeout", cost=self.timeout_penalty, message=message)
            print(f"[timeout] config_id={config_id}, budget={budget}, "
                  f"cost={self.timeout_penalty}, dir={trial_dir}: {message}", flush=True)
        else:
            record.update(status=status.name.lower(),
                          cost=float(cost) if math.isfinite(cost) else None)
        record["elapsed_seconds"] = runtime
        if status == StatusType.TIMEOUT:
            record["additional_info"] = info
        write_trial_status(trial_dir, record)
        return status, cost, runtime, cpu_time, info


class TimeoutAwareEncoder(RunHistoryLogScaledEncoder):
    """Include explicit penalties, but never fit the None-policy placeholder."""

    def _get_timeout_trials(self, budget_subset=None):
        return {key: value for key, value in super()._get_timeout_trials(budget_subset).items()
                if not is_unscored_timeout(value)}


class TimeoutAwareHyperband(Hyperband):
    """Unfinished models cannot be incumbents or advance to another fidelity."""

    def _timed_out(self, config, key):
        trial_key = TrialKey(self.runhistory.get_config_id(config), key.instance, key.seed, key.budget)
        return self.runhistory[trial_key].status == StatusType.TIMEOUT

    def get_instance_seed_budget_keys(self, config, compare=False):
        keys = [key for key in super().get_instance_seed_budget_keys(config, compare=False)
                if not self._timed_out(config, key)]
        if compare:
            keys = list({InstanceSeedBudgetKey(key.instance, key.seed, None) for key in keys})
        return keys

    def get_incumbents(self, sort_by=None):
        return [config for config in super().get_incumbents(sort_by=sort_by)
                if self.get_instance_seed_budget_keys(config)]

    def get_incumbent(self):
        if self._scenario.count_objectives() > 1:
            raise ValueError("Cannot get a single incumbent for multi-objective optimization.")
        incumbents = self.get_incumbents()
        return incumbents[0] if incumbents else None

    def _get_best_configs(self, configs, bracket, stage, from_keys):
        if stage + 1 >= len(self._n_configs_in_stage[bracket]):
            return []
        for config in configs:
            evaluated = self.runhistory.get_instance_seed_budget_keys(config)
            if not all(key in evaluated for key in from_keys):
                raise NotEvaluatedError
        eligible = [config for config in configs
                    if not any(self._timed_out(config, key) for key in from_keys)]
        needed = self._n_configs_in_stage[bracket][stage + 1]
        if len(eligible) <= needed:
            return eligible
        return super()._get_best_configs(eligible, bracket, stage, from_keys)
