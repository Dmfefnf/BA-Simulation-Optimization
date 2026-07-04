"""Bayesian optimization for the Highest Lateness Risk baseline agent.

This script tunes the two parameters that define the lateness-risk dispatching
rule while keeping the agent fixed to action 3.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RL_ROOT = Path(__file__).resolve().parents[2]
for path in (
    RL_ROOT / "RL_simulation",
    RL_ROOT / "experiments",
    RL_ROOT / "hyperparameter_tuning" / "RL_agent",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
except ImportError as exc:
    raise ImportError(
        "Ax Platform is required for BO_lateness_risk.py. Install it with "
        "`pip install ax-platform` in the project environment."
    ) from exc

from RL_agents import BaselineAgent
from RL_experiment import (
    BASE_SEED,
    CSV_FIELDS,
    RATE_MULTIPLIER,
    RUN_DURATION,
    SEED_STEP,
    STATION_CAPACITIES,
    append_csv_row,
    json_safe,
    make_eval_seeds,
    run_safely,
)
from lab_analysis_simulation_RL import DEFAULT_RISK_T1, DEFAULT_RISK_WINDOW, simulate_rl
from rl_tuning_common import FAILED_RUN_PENALTY, PARAMETER_BOUNDS, write_json

BASE_DIR = RL_ROOT
DEFAULT_OUTPUT_DIR = BASE_DIR / "results" / "lateness_risk_bo_results" / "lateness_risk_baseline"
BO_RANDOM_SEED = 24680
N_TRIALS = 50
N_REPLICATIONS = 30
FIXED_ACTION = 3
BASELINE_RULE = "Highest Lateness Risk"
PARAMETER_NAMES = ["risk_t1", "risk_window"]
PARAMETER_BOUNDS_LR = {
    name: PARAMETER_BOUNDS[name]
    for name in PARAMETER_NAMES
}

TRIAL_FIELDS = [
    "trial_index",
    *PARAMETER_NAMES,
    "risk_t2",
    "objective_mean",
    "objective_std",
    "best_objective_so_far",
    "total_reward_mean",
    "total_reward_std",
    "late_order_fraction_mean",
    "time_in_system_mean",
    "wip_mean",
    "n_valid_replications",
]
REPLICATION_FIELDS = [
    "trial_index",
    "objective_value",
    *CSV_FIELDS,
]

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--n-replications", type=int, default=N_REPLICATIONS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    parser.add_argument("--bo-random-seed", type=int, default=BO_RANDOM_SEED)
    parser.add_argument("--run-duration", type=float, default=RUN_DURATION)
    parser.add_argument("--rate-multiplier", type=float, default=RATE_MULTIPLIER)
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to existing CSV outputs instead of deleting previous files.",
    )
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return BASE_DIR / path


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "BO_lateness_risk.log", encoding="utf-8"),
        ],
        force=True,
    )


def reset_outputs(output_dir: Path) -> None:
    for file_name in [
        "lateness_risk_trials.csv",
        "lateness_risk_replications.csv",
        "lateness_risk_config.json",
        "lateness_risk_best_parameters.json",
        "BO_lateness_risk.log",
    ]:
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def value_as_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def sanitize_parameters(parameters: dict[str, Any]) -> dict[str, float]:
    sanitized: dict[str, float] = {}
    for name, (lower, upper) in PARAMETER_BOUNDS_LR.items():
        if name not in parameters:
            raise ValueError(f"Missing parameter '{name}'.")
        sanitized[name] = float(np.clip(float(parameters[name]), lower, upper))
    sanitized["risk_window"] = max(1.0, sanitized["risk_window"])
    return sanitized


def build_ax_parameters() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "type": "range",
            "bounds": [float(lower), float(upper)],
            "value_type": "float",
        }
        for name, (lower, upper) in PARAMETER_BOUNDS_LR.items()
    ]


def create_ax_client(random_seed: int) -> AxClient:
    random.seed(random_seed)
    np.random.seed(random_seed)
    try:
        ax_client = AxClient(random_seed=random_seed)
    except TypeError:
        ax_client = AxClient()
    try:
        ax_client.create_experiment(
            name="lateness_risk_baseline_bo",
            parameters=build_ax_parameters(),
            objectives={"objective": ObjectiveProperties(minimize=True)},
        )
    except TypeError:
        ax_client.create_experiment(
            name="lateness_risk_baseline_bo",
            parameters=build_ax_parameters(),
            objective_name="objective",
            minimize=True,
        )
    return ax_client


def complete_ax_trial(
    ax_client: AxClient,
    trial_index: int,
    objective_mean: float,
    objective_std: float,
    n_replications: int,
) -> None:
    objective_sem = objective_std / math.sqrt(n_replications) if n_replications else 0.0
    try:
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={"objective": (objective_mean, objective_sem)},
        )
    except Exception:
        ax_client.complete_trial(trial_index=trial_index, raw_data=objective_mean)


def simulation_kwargs(
    parameters: dict[str, float],
    run_duration: float,
    rate_multiplier: float,
) -> dict[str, Any]:
    return {
        "run_duration": run_duration,
        "rate_multiplier": rate_multiplier,
        "risk_t1": parameters["risk_t1"],
        "risk_window": parameters["risk_window"],
        **STATION_CAPACITIES,
    }


def evaluate_trial(
    parameters: dict[str, float],
    trial_index: int,
    output_dir: Path,
    eval_seeds: list[int],
    run_duration: float,
    rate_multiplier: float,
) -> dict[str, Any]:
    kwargs = simulation_kwargs(parameters, run_duration, rate_multiplier)
    objective_values: list[float] = []
    rewards: list[float] = []
    late_fractions: list[float] = []
    time_means: list[float] = []
    wip_means: list[float] = []
    n_valid = 0

    for rep, seed in enumerate(eval_seeds):
        row = run_safely(
            f"lateness_risk_trial_{trial_index}_replication_{rep}",
            lambda seed=seed: simulate_rl(
                agent=BaselineAgent(fixed_action=FIXED_ACTION),
                agent_type="baseline",
                fixed_action=FIXED_ACTION,
                training=False,
                random_seed=seed,
                **kwargs,
            ),
        )
        reward = value_as_float(row.get("total_reward"))
        if row.get("error") or reward is None:
            objective_value = FAILED_RUN_PENALTY
        else:
            objective_value = -reward
            rewards.append(reward)
            n_valid += 1

        for source, target in [
            ("late_order_fraction", late_fractions),
            ("time_in_system_mean", time_means),
            ("wip_mean", wip_means),
        ]:
            value = value_as_float(row.get(source))
            if value is not None:
                target.append(value)

        objective_values.append(objective_value)
        row.update(
            {
                "trial_index": trial_index,
                "replication": rep,
                "eval_seed": seed,
                "baseline_rule": BASELINE_RULE,
                "objective_value": objective_value,
            }
        )
        append_csv_row(
            row,
            output_dir / "lateness_risk_replications.csv",
            REPLICATION_FIELDS,
        )

    objective_array = np.asarray(objective_values, dtype=float)
    reward_array = np.asarray(rewards, dtype=float)
    return {
        "trial_index": trial_index,
        **parameters,
        "risk_t2": parameters["risk_t1"] + parameters["risk_window"],
        "objective_mean": float(np.mean(objective_array)),
        "objective_std": (
            float(np.std(objective_array, ddof=1)) if len(objective_array) > 1 else 0.0
        ),
        "total_reward_mean": float(np.mean(reward_array)) if len(reward_array) else math.nan,
        "total_reward_std": (
            float(np.std(reward_array, ddof=1)) if len(reward_array) > 1 else 0.0
        ),
        "late_order_fraction_mean": (
            float(np.mean(late_fractions)) if late_fractions else math.nan
        ),
        "time_in_system_mean": float(np.mean(time_means)) if time_means else math.nan,
        "wip_mean": float(np.mean(wip_means)) if wip_means else math.nan,
        "n_valid_replications": n_valid,
    }


def append_trial_row(row: dict[str, Any], output_dir: Path) -> None:
    append_csv_row(row, output_dir / "lateness_risk_trials.csv", TRIAL_FIELDS)


def save_config(args: argparse.Namespace, output_dir: Path, eval_seeds: list[int]) -> None:
    config = {
        "experiment": "lateness_risk_baseline_bo",
        "objective": "Ax minimizes -mean(total_reward); lower objective is better.",
        "baseline_rule": BASELINE_RULE,
        "fixed_action": FIXED_ACTION,
        "n_trials": args.n_trials,
        "n_replications": args.n_replications,
        "run_duration": args.run_duration,
        "rate_multiplier": args.rate_multiplier,
        "base_seed": args.base_seed,
        "seed_step": args.seed_step,
        "bo_random_seed": args.bo_random_seed,
        "eval_seeds": eval_seeds,
        "station_capacities": STATION_CAPACITIES,
        "parameter_bounds": PARAMETER_BOUNDS_LR,
        "default_risk_t1": DEFAULT_RISK_T1,
        "default_risk_window": DEFAULT_RISK_WINDOW,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(output_dir / "lateness_risk_config.json", config)


def save_best(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No trial rows available.")
    best = min(rows, key=lambda row: float(row["objective_mean"]))
    result = {
        "objective": "Ax minimizes -mean(total_reward); lower objective is better.",
        "best_trial_index": best["trial_index"],
        "best_objective": best["objective_mean"],
        "best_total_reward_mean": best["total_reward_mean"],
        "best_late_order_fraction_mean": best["late_order_fraction_mean"],
        "best_parameters": {name: best[name] for name in PARAMETER_NAMES},
        "best_risk_t2": best["risk_t2"],
        "n_valid_replications": best["n_valid_replications"],
    }
    write_json(output_dir / "lateness_risk_best_parameters.json", result)
    return result


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        reset_outputs(output_dir)
    setup_logging(output_dir)

    eval_seeds = make_eval_seeds(args.n_replications, args.base_seed, args.seed_step)
    save_config(args, output_dir, eval_seeds)
    ax_client = create_ax_client(args.bo_random_seed)
    rows: list[dict[str, Any]] = []
    best_objective_so_far = math.inf

    LOGGER.info(
        "Starting lateness-risk baseline BO with %s trials and %s replications",
        args.n_trials,
        args.n_replications,
    )
    for _ in range(args.n_trials):
        raw_parameters, trial_index = ax_client.get_next_trial()
        parameters = sanitize_parameters(raw_parameters)
        trial_row = evaluate_trial(
            parameters=parameters,
            trial_index=trial_index,
            output_dir=output_dir,
            eval_seeds=eval_seeds,
            run_duration=args.run_duration,
            rate_multiplier=args.rate_multiplier,
        )
        best_objective_so_far = min(best_objective_so_far, trial_row["objective_mean"])
        trial_row["best_objective_so_far"] = best_objective_so_far
        rows.append(trial_row)
        append_trial_row(trial_row, output_dir)
        complete_ax_trial(
            ax_client,
            trial_index,
            trial_row["objective_mean"],
            trial_row["objective_std"],
            args.n_replications,
        )
        save_best(output_dir, rows)
        LOGGER.info(
            "Trial %s objective %.3f reward %.3f best %.3f",
            trial_index,
            trial_row["objective_mean"],
            trial_row["total_reward_mean"],
            best_objective_so_far,
        )

    best = save_best(output_dir, rows)
    logging.shutdown()
    return {"output_dir": output_dir, "best": best}


def main() -> None:
    result = run_experiment(parse_args())
    print("Best lateness-risk baseline parameters:")
    print(json.dumps(json_safe(result["best"]), indent=2))
    print(f"Results written to: {result['output_dir']}")


if __name__ == "__main__":
    main()
