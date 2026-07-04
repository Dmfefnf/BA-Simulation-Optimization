from pathlib import Path
import subprocess
import sys


N_RUNS = 20
N_TRIALS = 30
N_REPLICATIONS = 30
MULTI_RUN_OUTPUT_DIR = "multi_run_results"
COMBINE_AFTER_RUNS = False
SKIP_COMPLETED_RUNS = True

BASE_SIMULATION_SEED = 12345
RUN_SEED_STEP = 100_000
BASE_RANDOM_SEARCH_SEED = 54_321
RANDOM_SEARCH_SEED_STEP = 1_000

BASE_DIR = Path(__file__).resolve().parent
BO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = BO_ROOT / "results" / MULTI_RUN_OUTPUT_DIR
RUN_SINGLE_SCRIPT = BO_ROOT / "experiments" / "single_run" / "run_single_random_search.py"


def run_output_dir(run_index: int) -> Path:
    return RESULTS_DIR / "random_search" / f"run_{run_index:02d}"


def run_completed(run_index: int) -> bool:
    output_dir = run_output_dir(run_index)
    return all(
        [
            (output_dir / "random_search_best_parameters.json").exists(),
            (output_dir / "random_search_trials.csv").exists(),
            (output_dir / "random_search_replications.csv").exists(),
        ]
    )


def build_run_command(
    run_index: int,
    n_trials: int,
    n_replications: int,
) -> list[str]:
    return [
        sys.executable,
        str(RUN_SINGLE_SCRIPT),
        "--run-index",
        str(run_index),
        "--n-trials",
        str(n_trials),
        "--n-replications",
        str(n_replications),
        "--base-seed",
        str(BASE_SIMULATION_SEED + run_index * RUN_SEED_STEP),
        "--random-search-seed",
        str(BASE_RANDOM_SEARCH_SEED + run_index * RANDOM_SEARCH_SEED_STEP),
        "--output-dir",
        str(run_output_dir(run_index)),
    ]


def run_all(
    n_runs: int = N_RUNS,
    n_trials: int = N_TRIALS,
    n_replications: int = N_REPLICATIONS,
) -> dict[str, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for run_index in range(n_runs):
        if SKIP_COMPLETED_RUNS and run_completed(run_index):
            print(f"Skipping completed random-search run {run_index}.", flush=True)
            continue

        print(f"Starting random-search run {run_index}.", flush=True)
        subprocess.run(
            build_run_command(run_index, n_trials, n_replications),
            check=True,
            cwd=BO_ROOT,
        )

    if COMBINE_AFTER_RUNS:
        from combine_multi_run_results import combine_all_results

        combine_all_results(methods=("random_search",))

    return {
        "runs_dir": RESULTS_DIR / "random_search",
        "results_dir": RESULTS_DIR,
    }


def main() -> None:
    run_all()
    print(
        f"Random-search run folders written to {RESULTS_DIR / 'random_search'}",
        flush=True,
    )
    if not COMBINE_AFTER_RUNS:
        print(
            "Combine results later with: python experiments/multi_run/combine_multi_run_results.py",
            flush=True,
        )


if __name__ == "__main__":
    main()
