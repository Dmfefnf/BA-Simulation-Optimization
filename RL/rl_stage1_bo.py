"""Stage 1: Bayesian optimization over compact laptop RL runs."""

from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
except ImportError as exc:
    raise ImportError(
        "Ax Platform is required for rl_stage1_bo.py. Install it with "
        "`pip install ax-platform` in the project environment."
    ) from exc

from RL_experiment import BASE_SEED, RATE_MULTIPLIER, RUN_DURATION, SEED_STEP, append_csv_row
from rl_tuning_common import (
    ACTIVE_PARAMETER_BOUNDS,
    DEFAULT_OUTPUT_ROOT,
    LOG_PARAMETER_NAMES,
    PARAMETER_NAMES,
    base_run_config,
    json_safe,
    read_json,
    sanitize_parameters,
    write_json,
)

BASE_DIR = Path(__file__).resolve().parent
RUN_SINGLE_SCRIPT = BASE_DIR / "run_single_rl_bo_trial.py"
BO_RANDOM_SEED = 24680
STAGE1_FIELDS = [
    "trial_index",
    *LOG_PARAMETER_NAMES,
    "objective_mean",
    "objective_std",
    "best_objective_so_far",
    "total_reward_mean",
    "total_reward_std",
    "late_order_fraction_mean",
    "n_valid_replications",
    "best_training_reward",
    "best_training_episode",
    "best_training_seed",
    "q_table_size",
    "trial_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--training-episodes", type=int, default=50)
    parser.add_argument("--eval-replications", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    parser.add_argument("--bo-random-seed", type=int, default=BO_RANDOM_SEED)
    parser.add_argument("--run-duration", type=float, default=RUN_DURATION)
    parser.add_argument("--rate-multiplier", type=float, default=RATE_MULTIPLIER)
    return parser.parse_args()


def build_ax_parameters() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "type": "range",
            "bounds": [float(lower), float(upper)],
            "value_type": "float",
        }
        for name, (lower, upper) in ACTIVE_PARAMETER_BOUNDS.items()
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
            name="rl_stage1_laptop_bo",
            parameters=build_ax_parameters(),
            objectives={"objective": ObjectiveProperties(minimize=True)},
        )
    except TypeError:
        ax_client.create_experiment(
            name="rl_stage1_laptop_bo",
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
    eval_replications: int,
) -> None:
    objective_sem = objective_std / math.sqrt(eval_replications) if eval_replications else 0.0
    try:
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={"objective": (objective_mean, objective_sem)},
        )
    except Exception:
        ax_client.complete_trial(trial_index=trial_index, raw_data=objective_mean)


def reset_stage1_outputs(output_dir: Path) -> None:
    for file_name in [
        "stage1_trials.csv",
        "stage1_config.json",
        "stage1_best_parameters.json",
    ]:
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def run_single_trial_process(
    trial_index: int,
    parameters: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    trial_dir = args.output_dir / f"trial_{trial_index:03d}"
    command = [
        sys.executable,
        str(RUN_SINGLE_SCRIPT),
        "--trial-index",
        str(trial_index),
        "--output-dir",
        str(trial_dir),
        "--training-episodes",
        str(args.training_episodes),
        "--eval-replications",
        str(args.eval_replications),
        "--base-seed",
        str(args.base_seed),
        "--seed-step",
        str(args.seed_step),
        "--run-duration",
        str(args.run_duration),
        "--rate-multiplier",
        str(args.rate_multiplier),
    ]
    for name in PARAMETER_NAMES:
        command.extend([f"--{name.replace('_', '-')}", str(parameters[name])])

    subprocess.run(command, check=True, cwd=BASE_DIR)
    summary = read_json(trial_dir / "trial_summary.json")
    summary["trial_dir"] = str(trial_dir)
    return summary


def trial_row(summary: dict[str, Any], best_objective_so_far: float) -> dict[str, Any]:
    parameters = summary["parameters"]
    best_training = summary.get("best_training") or {}
    return {
        "trial_index": summary["trial_index"],
        **parameters,
        "objective_mean": summary["objective_mean"],
        "objective_std": summary["objective_std"],
        "best_objective_so_far": best_objective_so_far,
        "total_reward_mean": summary["total_reward_mean"],
        "total_reward_std": summary["total_reward_std"],
        "late_order_fraction_mean": summary["late_order_fraction_mean"],
        "n_valid_replications": summary["n_valid_replications"],
        "best_training_reward": best_training.get("total_reward", ""),
        "best_training_episode": best_training.get("episode", ""),
        "best_training_seed": best_training.get("training_seed", ""),
        "q_table_size": summary["q_table_size"],
        "trial_dir": summary["trial_dir"],
    }


def save_best(output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("No Stage-1 rows available.")
    best = min(rows, key=lambda row: float(row["objective_mean"]))
    result = {
        "objective": "Ax minimizes -mean(total_reward); lower is better.",
        "late_fraction_note": (
            "late_order_fraction_mean is recorded but not used in the objective. "
            "A later objective can add e.g. lambda * late_fraction."
        ),
        "stage1_best_trial_index": best["trial_index"],
        "stage1_best_objective": best["objective_mean"],
        "stage1_best_total_reward_mean": best["total_reward_mean"],
        "stage1_best_parameters": {name: best[name] for name in LOG_PARAMETER_NAMES},
        "source_trial_dir": best["trial_dir"],
    }
    write_json(output_dir / "stage1_best_parameters.json", result)
    return result


def run_stage1(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reset_stage1_outputs(args.output_dir)
    config = {
        "stage": "stage1_bo",
        "n_trials": args.n_trials,
        "training_episodes": args.training_episodes,
        "eval_replications": args.eval_replications,
        "bo_random_seed": args.bo_random_seed,
        "single_trial_script": str(RUN_SINGLE_SCRIPT),
        **base_run_config(args),
    }
    write_json(args.output_dir / "stage1_config.json", config)

    ax_client = create_ax_client(args.bo_random_seed)
    rows: list[dict[str, Any]] = []
    best_objective_so_far = math.inf

    for _ in range(args.n_trials):
        parameters, trial_index = ax_client.get_next_trial()
        sanitized = sanitize_parameters(parameters, args.training_episodes)
        summary = run_single_trial_process(trial_index, sanitized, args)
        best_objective_so_far = min(best_objective_so_far, summary["objective_mean"])
        row = trial_row(summary, best_objective_so_far)
        rows.append(row)
        append_csv_row(row, args.output_dir / "stage1_trials.csv", STAGE1_FIELDS)
        complete_ax_trial(
            ax_client,
            trial_index,
            summary["objective_mean"],
            summary["objective_std"],
            args.eval_replications,
        )
        save_best(args.output_dir, rows)

    best = save_best(args.output_dir, rows)
    return {"output_dir": args.output_dir, "best": best}


def main() -> None:
    result = run_stage1(parse_args())
    print("Best Stage-1 RL parameters:")
    print(json_safe(result["best"]))
    print(f"Stage-1 results written to: {result['output_dir']}")


if __name__ == "__main__":
    main()
