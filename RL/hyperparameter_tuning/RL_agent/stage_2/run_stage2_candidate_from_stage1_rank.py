"""Run one Stage-2 candidate selected by rank from Stage-1 results.

This entry point is intended for SLURM array jobs. Each task selects exactly one
ranked Stage-1 row and writes only to that candidate's output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

RL_ROOT = Path(__file__).resolve().parents[3]
for path in (
    RL_ROOT / "RL_simulation",
    RL_ROOT / "experiments",
    RL_ROOT / "hyperparameter_tuning" / "RL_agent",
    RL_ROOT / "hyperparameter_tuning" / "RL_agent" / "stage_2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from RL_experiment import BASE_SEED, RATE_MULTIPLIER, RUN_DURATION, SEED_STEP
from rl_stage2_evaluate_top import (
    RUN_SINGLE_SCRIPT,
    candidate_dir,
    load_top_stage1_rows,
    run_single_candidate_process,
)
from rl_tuning_common import DEFAULT_OUTPUT_ROOT, read_json, write_json

COMPLETION_FILES = ("candidate_summary.json", "evaluation.csv", "q_table.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage2")
    parser.add_argument(
        "--candidate-rank",
        type=int,
        required=True,
        help="1-based rank after sorting Stage-1 rows by objective_mean ascending.",
    )
    parser.add_argument("--n-top", type=int, default=1)
    parser.add_argument("--training-episodes", type=int, default=1000)
    parser.add_argument("--eval-replications", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    parser.add_argument("--run-duration", type=float, default=RUN_DURATION)
    parser.add_argument("--rate-multiplier", type=float, default=RATE_MULTIPLIER)
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip if candidate_summary.json, evaluation.csv and q_table.pkl exist.",
    )
    return parser.parse_args()


def is_completed(path: Path) -> bool:
    return all((path / file_name).is_file() for file_name in COMPLETION_FILES)


def enrich_candidate_summary(
    candidate_path: Path,
    source_row: dict[str, Any],
) -> dict[str, Any]:
    summary_path = candidate_path / "candidate_summary.json"
    summary = read_json(summary_path)
    summary["candidate_dir"] = str(candidate_path)
    summary["stage1_objective_mean"] = source_row.get("objective_mean", "")
    summary["stage1_total_reward_mean"] = source_row.get("total_reward_mean", "")
    write_json(summary_path, summary)
    return summary


def run_ranked_candidate(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.candidate_rank < 1:
        raise ValueError("--candidate-rank is 1-based and must be >= 1.")
    if args.n_top < 1:
        raise ValueError("--n-top must be >= 1.")
    if args.candidate_rank > args.n_top:
        print(
            f"Skipping candidate rank {args.candidate_rank} because n_top={args.n_top}.",
            flush=True,
        )
        return None

    top_rows = load_top_stage1_rows(args.stage1_dir, args.n_top)
    if args.candidate_rank > len(top_rows):
        print(
            "Skipping candidate rank "
            f"{args.candidate_rank}: only {len(top_rows)} Stage-1 rows available.",
            flush=True,
        )
        return None

    source_row = top_rows[args.candidate_rank - 1]
    trial_index = int(source_row["trial_index"])
    out_dir = candidate_dir(args.output_dir, args.candidate_rank, trial_index)

    if args.skip_completed and is_completed(out_dir):
        print(f"Skipping completed Stage-2 candidate at {out_dir}.", flush=True)
        return read_json(out_dir / "candidate_summary.json")

    print(
        "Running Stage-2 candidate "
        f"rank={args.candidate_rank} source_trial={trial_index} output={out_dir}",
        flush=True,
    )
    summary = run_single_candidate_process(args.candidate_rank, source_row, args)
    return enrich_candidate_summary(out_dir, source_row)


def main() -> None:
    summary = run_ranked_candidate(parse_args())
    if summary is None:
        return
    print(
        "Stage-2 candidate rank {candidate_rank} completed with "
        "objective={objective_mean:.3f} reward_mean={total_reward_mean:.3f}".format(
            **summary
        ),
        flush=True,
    )
    print(f"Single-candidate script: {RUN_SINGLE_SCRIPT}", flush=True)


if __name__ == "__main__":
    main()
