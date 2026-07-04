"""Stage 2: retrain and evaluate top Stage-1 RL configurations."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

RL_ROOT = Path(__file__).resolve().parents[3]
for path in (
    RL_ROOT / "RL_simulation",
    RL_ROOT / "experiments",
    RL_ROOT / "hyperparameter_tuning" / "RL_agent",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from RL_experiment import BASE_SEED, RATE_MULTIPLIER, RUN_DURATION, SEED_STEP, append_csv_row
from rl_tuning_common import (
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
RUN_SINGLE_SCRIPT = BASE_DIR / "run_single_rl_stage2_candidate.py"
STAGE2_FIELDS = [
    "candidate_rank",
    "source_trial_index",
    *LOG_PARAMETER_NAMES,
    "stage1_objective_mean",
    "stage1_total_reward_mean",
    "objective_mean",
    "objective_std",
    "total_reward_mean",
    "total_reward_std",
    "late_order_fraction_mean",
    "n_valid_replications",
    "best_training_reward",
    "best_training_episode",
    "best_training_seed",
    "best_evaluation_reward",
    "best_evaluation_replication",
    "best_evaluation_seed",
    "q_table_size",
    "candidate_dir",
    "q_table_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage2")
    parser.add_argument("--n-top", type=int, default=1)
    parser.add_argument("--training-episodes", type=int, default=1000)
    parser.add_argument("--eval-replications", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    parser.add_argument("--run-duration", type=float, default=RUN_DURATION)
    parser.add_argument("--rate-multiplier", type=float, default=RATE_MULTIPLIER)
    return parser.parse_args()


def reset_stage2_outputs(output_dir: Path) -> None:
    for file_name in ["stage2_candidates.csv", "stage2_config.json"]:
        path = output_dir / file_name
        if path.exists():
            path.unlink()


def load_top_stage1_rows(stage1_dir: Path, n_top: int) -> list[dict[str, Any]]:
    trials_path = stage1_dir / "stage1_trials.csv"
    if not trials_path.exists():
        raise FileNotFoundError(f"Missing Stage-1 trials file: {trials_path}")
    trials = pd.read_csv(trials_path)
    if trials.empty:
        raise RuntimeError(f"No Stage-1 trials found in {trials_path}")
    required = {"trial_index", "objective_mean", *PARAMETER_NAMES}
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"Stage-1 trials CSV is missing columns: {sorted(missing)}")
    top = trials.sort_values("objective_mean", ascending=True).head(n_top)
    return top.to_dict(orient="records")


def candidate_dir(output_dir: Path, rank: int, trial_index: int) -> Path:
    return output_dir / f"candidate_rank_{rank:02d}_trial_{trial_index:03d}"


def run_single_candidate_process(
    rank: int,
    source_row: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    trial_index = int(source_row["trial_index"])
    parameters = sanitize_parameters(
        {name: source_row[name] for name in PARAMETER_NAMES},
        args.training_episodes,
    )
    out_dir = candidate_dir(args.output_dir, rank, trial_index)
    command = [
        sys.executable,
        str(RUN_SINGLE_SCRIPT),
        "--candidate-rank",
        str(rank),
        "--source-trial-index",
        str(trial_index),
        "--output-dir",
        str(out_dir),
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
    summary = read_json(out_dir / "candidate_summary.json")
    summary["candidate_dir"] = str(out_dir)
    summary["stage1_objective_mean"] = source_row.get("objective_mean", "")
    summary["stage1_total_reward_mean"] = source_row.get("total_reward_mean", "")
    return summary


def candidate_row(summary: dict[str, Any]) -> dict[str, Any]:
    best_training = summary.get("best_training") or {}
    best_eval = summary.get("best_evaluation") or {}
    return {
        "candidate_rank": summary["candidate_rank"],
        "source_trial_index": summary["source_trial_index"],
        **summary["parameters"],
        "stage1_objective_mean": summary.get("stage1_objective_mean", ""),
        "stage1_total_reward_mean": summary.get("stage1_total_reward_mean", ""),
        "objective_mean": summary["objective_mean"],
        "objective_std": summary["objective_std"],
        "total_reward_mean": summary["total_reward_mean"],
        "total_reward_std": summary["total_reward_std"],
        "late_order_fraction_mean": summary["late_order_fraction_mean"],
        "n_valid_replications": summary["n_valid_replications"],
        "best_training_reward": best_training.get("total_reward", ""),
        "best_training_episode": best_training.get("episode", ""),
        "best_training_seed": best_training.get("training_seed", ""),
        "best_evaluation_reward": best_eval.get("total_reward", ""),
        "best_evaluation_replication": best_eval.get("replication", ""),
        "best_evaluation_seed": best_eval.get("eval_seed", ""),
        "q_table_size": summary["q_table_size"],
        "candidate_dir": summary["candidate_dir"],
        "q_table_path": summary["q_table_path"],
    }


def run_stage2(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reset_stage2_outputs(args.output_dir)
    top_rows = load_top_stage1_rows(args.stage1_dir, args.n_top)
    config = {
        "stage": "stage2_evaluate_top",
        "stage1_dir": str(args.stage1_dir),
        "n_top": args.n_top,
        "training_episodes": args.training_episodes,
        "eval_replications": args.eval_replications,
        "single_candidate_script": str(RUN_SINGLE_SCRIPT),
        **base_run_config(args),
    }
    write_json(args.output_dir / "stage2_config.json", config)

    rows: list[dict[str, Any]] = []
    for rank, source_row in enumerate(top_rows, start=1):
        summary = run_single_candidate_process(rank, source_row, args)
        row = candidate_row(summary)
        rows.append(row)
        append_csv_row(row, args.output_dir / "stage2_candidates.csv", STAGE2_FIELDS)

    return {"output_dir": args.output_dir, "rows": rows}


def main() -> None:
    result = run_stage2(parse_args())
    print(f"Stage-2 candidates written to: {result['output_dir']}")
    if result["rows"]:
        print("Best Stage-2 candidate:")
        print(json_safe(min(result["rows"], key=lambda row: float(row["objective_mean"]))))


if __name__ == "__main__":
    main()
