from pathlib import Path
from typing import Iterable

import pandas as pd


MULTI_RUN_OUTPUT_DIR = "multi_run_results"
BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / MULTI_RUN_OUTPUT_DIR
SUMMARY_CSV = RESULTS_DIR / "summary.csv"

PARAMETER_COLUMNS = [
    "preparation_capacity",
    "sorting_capacity",
    "analysis1_capacity",
    "analysis2_capacity",
    "evaluation_capacity",
    "dispatching_capacity",
    "worker_capacity",
]

METHOD_CONFIG = {
    "bo": {
        "runs_dir": RESULTS_DIR / "bo",
        "trial_file": "bo_trials.csv",
        "replication_file": "bo_replications.csv",
        "combined_trials": RESULTS_DIR / "bo_all_trials.csv",
        "combined_replications": RESULTS_DIR / "bo_all_replications.csv",
    },
    "random_search": {
        "runs_dir": RESULTS_DIR / "random_search",
        "trial_file": "random_search_trials.csv",
        "replication_file": "random_search_replications.csv",
        "combined_trials": RESULTS_DIR / "random_search_all_trials.csv",
        "combined_replications": RESULTS_DIR
        / "random_search_all_replications.csv",
    },
}


def iter_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(
        path for path in runs_dir.iterdir() if path.is_dir() and path.name.startswith("run_")
    )


def append_csv_file(source_path: Path, target_path: Path) -> int:
    if not source_path.exists():
        return 0

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_exists = target_path.exists()
    rows_written = 0

    with source_path.open("r", encoding="utf-8", newline="") as source:
        header = source.readline()
        if not header:
            return 0

        with target_path.open("a", encoding="utf-8", newline="") as target:
            if not target_exists:
                target.write(header)
            for line in source:
                target.write(line)
                rows_written += 1

    return rows_written


def reset_combined_outputs(methods: Iterable[str]) -> None:
    paths = [SUMMARY_CSV]
    for method in methods:
        config = METHOD_CONFIG[method]
        paths.extend([config["combined_trials"], config["combined_replications"]])

    for path in paths:
        if path.exists():
            path.unlink()


def combine_method(method: str) -> dict[str, int]:
    config = METHOD_CONFIG[method]
    run_dirs = iter_run_dirs(config["runs_dir"])
    trial_rows = 0
    replication_rows = 0

    for run_dir in run_dirs:
        trial_rows += append_csv_file(
            run_dir / config["trial_file"],
            config["combined_trials"],
        )
        replication_rows += append_csv_file(
            run_dir / config["replication_file"],
            config["combined_replications"],
        )

    return {
        "n_run_dirs": len(run_dirs),
        "trial_rows": trial_rows,
        "replication_rows": replication_rows,
    }


def save_summary(methods: Iterable[str]) -> int:
    summary_frames = []
    for method in methods:
        path = METHOD_CONFIG[method]["combined_trials"]
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
        return 0

    summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    preferred_columns = [
        "method",
        "run_index",
        "final_best_objective",
        "objective_mean",
        "objective_std",
        "n_valid_replications",
        *PARAMETER_COLUMNS,
    ]
    preferred_columns.extend(
        column
        for column in summary.columns
        if column.endswith("_mean") and column not in preferred_columns
    )
    columns = [column for column in preferred_columns if column in summary.columns]
    summary[columns].to_csv(SUMMARY_CSV, index=False)
    return len(summary)


def combine_all_results(
    methods: Iterable[str] = ("bo", "random_search"),
) -> dict[str, dict[str, int] | int]:
    methods = tuple(methods)
    unknown_methods = sorted(set(methods) - set(METHOD_CONFIG))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reset_combined_outputs(methods)

    result: dict[str, dict[str, int] | int] = {}
    for method in methods:
        result[method] = combine_method(method)

    result["summary_rows"] = save_summary(methods)
    return result


def main() -> None:
    result = combine_all_results()
    print("Combined multi-run results:")
    for method in ["bo", "random_search"]:
        stats = result.get(method, {})
        print(f"{method}: {stats}")
    print(f"summary_rows: {result['summary_rows']}")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
