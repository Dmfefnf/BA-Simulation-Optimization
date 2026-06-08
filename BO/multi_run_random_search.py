from pathlib import Path
from typing import Any

import pandas as pd

import random_search


N_RUNS = 20
N_TRIALS = random_search.N_TRIALS
N_REPLICATIONS = random_search.N_REPLICATIONS
MULTI_RUN_OUTPUT_DIR = "multi_run_results"

BASE_SIMULATION_SEED = 12345
RUN_SEED_STEP = 100_000
BASE_RANDOM_SEARCH_SEED = 54_321
RANDOM_SEARCH_SEED_STEP = 1_000

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / MULTI_RUN_OUTPUT_DIR


def run_output_dir(run_index: int) -> Path:
    return RESULTS_DIR / "random_search" / f"run_{run_index:02d}"


def save_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_path, index=False)


def save_summary() -> None:
    summary_frames = []
    for path in [
        RESULTS_DIR / "bo_all_trials.csv",
        RESULTS_DIR / "random_search_all_trials.csv",
    ]:
        if not path.exists():
            continue
        trials = pd.read_csv(path)
        if trials.empty:
            continue
        final_trials = (
            trials.sort_values(["method", "run_index", "trial_index"])
            .groupby(["method", "run_index"], as_index=False)
            .tail(1)
            .copy()
        )
        final_trials["final_best_objective"] = final_trials[
            "best_objective_so_far"
        ]
        summary_frames.append(final_trials)

    if not summary_frames:
        return

    summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    preferred_columns = [
        "method",
        "run_index",
        "final_best_objective",
        "objective_mean",
        "objective_std",
        "n_valid_replications",
        *random_search.PARAMETER_BOUNDS.keys(),
    ]
    preferred_columns.extend(
        column
        for column in summary.columns
        if column.endswith("_mean") and column not in preferred_columns
    )
    columns = [column for column in preferred_columns if column in summary.columns]
    summary[columns].to_csv(RESULTS_DIR / "summary.csv", index=False)


def run_all(
    n_runs: int = N_RUNS,
    n_trials: int = N_TRIALS,
    n_replications: int = N_REPLICATIONS,
) -> dict[str, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_trials: list[dict[str, Any]] = []
    all_replications: list[dict[str, Any]] = []

    for run_index in range(n_runs):
        result = random_search.run_experiment(
            n_trials=n_trials,
            n_replications=n_replications,
            base_seed=BASE_SIMULATION_SEED + run_index * RUN_SEED_STEP,
            random_search_seed=(
                BASE_RANDOM_SEARCH_SEED + run_index * RANDOM_SEARCH_SEED_STEP
            ),
            output_dir=run_output_dir(run_index),
            run_index=run_index,
        )
        all_trials.extend(result["trials"])
        all_replications.extend(result["replications"])

        save_records(all_trials, RESULTS_DIR / "random_search_all_trials.csv")
        save_records(
            all_replications,
            RESULTS_DIR / "random_search_all_replications.csv",
        )
        save_summary()

    return {
        "trials": pd.DataFrame(all_trials),
        "replications": pd.DataFrame(all_replications),
    }


def main() -> None:
    run_all()
    print(f"Random-search multi-run results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
