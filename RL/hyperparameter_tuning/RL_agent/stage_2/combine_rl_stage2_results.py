"""Combine completed Stage-2 RL array candidate results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

RL_ROOT = Path(__file__).resolve().parents[3]
for path in (
    RL_ROOT / "hyperparameter_tuning" / "RL_agent",
    RL_ROOT / "hyperparameter_tuning" / "RL_agent" / "stage_2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl_stage2_evaluate_top import STAGE2_FIELDS, candidate_row
from rl_tuning_common import DEFAULT_OUTPUT_ROOT, json_safe, read_json, write_json

COMPLETION_FILES = ("candidate_summary.json", "evaluation.csv", "q_table.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage2-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "stage2")
    parser.add_argument("--expected-n-top", type=int, default=None)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=STAGE2_FIELDS,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(value) for key, value in row.items()})


def load_candidate_rows(stage2_dir: Path) -> list[dict[str, Any]]:
    if not stage2_dir.exists():
        raise FileNotFoundError(f"Missing Stage-2 directory: {stage2_dir}")

    rows: list[dict[str, Any]] = []
    candidate_paths = sorted(stage2_dir.glob("candidate_rank_*_trial_*"))
    if not candidate_paths:
        print(f"No Stage-2 candidate directories found under {stage2_dir}.", flush=True)

    for candidate_path in candidate_paths:
        if not candidate_path.is_dir():
            continue
        missing = [
            file_name
            for file_name in COMPLETION_FILES
            if not (candidate_path / file_name).is_file()
        ]
        if missing:
            print(
                f"Skipping incomplete candidate {candidate_path.name}; "
                f"missing: {', '.join(missing)}",
                flush=True,
            )
            continue
        summary = read_json(candidate_path / "candidate_summary.json")
        summary.setdefault("candidate_dir", str(candidate_path))
        rows.append(candidate_row(summary))

    rows.sort(key=lambda row: float(row["objective_mean"]))
    return rows


def report_missing_expected_ranks(
    stage2_dir: Path,
    rows: list[dict[str, Any]],
    expected_n_top: int | None,
) -> None:
    if expected_n_top is None:
        return
    found = {int(row["candidate_rank"]) for row in rows}
    for rank in range(1, expected_n_top + 1):
        if rank not in found:
            pattern = f"candidate_rank_{rank:02d}_trial_*"
            matches = list(stage2_dir.glob(pattern))
            if matches:
                print(
                    f"Expected rank {rank} exists but is not complete; skipped.",
                    flush=True,
                )
            else:
                print(
                    f"Expected rank {rank} is missing under {stage2_dir}; skipped.",
                    flush=True,
                )


def combine_stage2_results(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_candidate_rows(args.stage2_dir)
    report_missing_expected_ranks(args.stage2_dir, rows, args.expected_n_top)

    candidates_path = args.stage2_dir / "stage2_candidates.csv"
    best_path = args.stage2_dir / "stage2_best_candidate.json"
    write_csv(candidates_path, rows)

    if rows:
        best = rows[0]
        write_json(
            best_path,
            {
                "objective": "minimize -mean(total_reward); lower is better.",
                "best_candidate": best,
            },
        )
    else:
        best_path.write_text(
            json.dumps(
                json_safe(
                    {
                        "objective": "minimize -mean(total_reward); lower is better.",
                        "best_candidate": None,
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    return {
        "n_completed_candidates": len(rows),
        "stage2_candidates_csv": candidates_path,
        "stage2_best_candidate_json": best_path,
    }


def main() -> None:
    result = combine_stage2_results(parse_args())
    print(
        "Combined {n_completed_candidates} completed Stage-2 candidates.".format(
            **result
        ),
        flush=True,
    )
    print(f"Wrote: {result['stage2_candidates_csv']}", flush=True)
    print(f"Wrote: {result['stage2_best_candidate_json']}", flush=True)


if __name__ == "__main__":
    main()
