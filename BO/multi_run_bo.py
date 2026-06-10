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
BASE_BO_RANDOM_SEED = 24_680
BO_RANDOM_SEED_STEP = 1_000

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / MULTI_RUN_OUTPUT_DIR
RUN_SINGLE_SCRIPT = BASE_DIR / "run_single_bo.py"


def run_output_dir(run_index: int) -> Path:
    return RESULTS_DIR / "bo" / f"run_{run_index:02d}"


def run_completed(run_index: int) -> bool:
    output_dir = run_output_dir(run_index)
    return all(
        [
            (output_dir / "bo_best_parameters.json").exists(),
            (output_dir / "bo_trials.csv").exists(),
            (output_dir / "bo_replications.csv").exists(),
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
        "--bo-random-seed",
        str(BASE_BO_RANDOM_SEED + run_index * BO_RANDOM_SEED_STEP),
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
            print(f"Skipping completed BO run {run_index}.", flush=True)
            continue

        print(f"Starting BO run {run_index}.", flush=True)
        subprocess.run(
            build_run_command(run_index, n_trials, n_replications),
            check=True,
            cwd=BASE_DIR,
        )

    if COMBINE_AFTER_RUNS:
        from combine_multi_run_results import combine_all_results

        combine_all_results(methods=("bo",))

    return {
        "runs_dir": RESULTS_DIR / "bo",
        "results_dir": RESULTS_DIR,
    }


def main() -> None:
    run_all()
    print(f"BO run folders written to {RESULTS_DIR / 'bo'}", flush=True)
    if not COMBINE_AFTER_RUNS:
        print("Combine results later with: python combine_multi_run_results.py", flush=True)


if __name__ == "__main__":
    main()
