import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from lab_analysis_simulation_BO import DAY, simulate
    from objective_utils import (
        DEFAULT_OBJECTIVE_WEIGHTS,
        DUE_DATE_MEAN,
        VALID_OBJECTIVE_TIME_MODES,
        calculate_objective_details,
    )
except ImportError:
    from lab_analysis_simulation_BO import simulate
    from objective_utils import (
        DEFAULT_OBJECTIVE_WEIGHTS,
        DUE_DATE_MEAN,
        VALID_OBJECTIVE_TIME_MODES,
        calculate_objective_details,
    )

    MINUTE = 1
    HOUR = 60 * MINUTE
    DAY = 24 * HOUR


N_TRIALS = 30
N_REPLICATIONS = 30
RUN_DURATION = 1 * DAY
RATE_MULTIPLIER = 0.5
BASE_SEED = 12345
SEED_STEP = 1000
RANDOM_SEARCH_SEED = 54321
OUTPUT_DIR = "random_search_results"
FAILED_RUN_PENALTY = 1e9
# Time modes: "none" (A), "mean" (B), "mean_std" (C).
OBJECTIVE_TIME_MODE = "mean"
METHOD = "random_search"
RUN_INDEX = 0

PARAMETER_BOUNDS = {
    "preparation_capacity": (1, 5),
    "sorting_capacity": (1, 5),
    "analysis1_capacity": (1, 5),
    "analysis2_capacity": (1, 5),
    "evaluation_capacity": (1, 5),
    "dispatching_capacity": (1, 5),
    "worker_capacity": (1, 10),
}

# Copy the shared defaults so local experiments can override weights here.
OBJECTIVE_WEIGHTS = DEFAULT_OBJECTIVE_WEIGHTS.copy()

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / OUTPUT_DIR
TRIALS_CSV = OUTPUT_PATH / "random_search_trials.csv"
REPLICATIONS_CSV = OUTPUT_PATH / "random_search_replications.csv"
BEST_PARAMETERS_JSON = OUTPUT_PATH / "random_search_best_parameters.json"
CONFIG_JSON = OUTPUT_PATH / "random_search_config.json"

REPLICATION_RECORDS: list[dict[str, Any]] = []
TRIAL_RECORDS: list[dict[str, Any]] = []


def setup_logging() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_PATH / "random_search.log", encoding="utf-8"),
        ],
        force=True,
    )


def configure_output_paths(output_dir: str | Path) -> None:
    global OUTPUT_DIR, OUTPUT_PATH, TRIALS_CSV, REPLICATIONS_CSV
    global BEST_PARAMETERS_JSON, CONFIG_JSON

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    OUTPUT_DIR = str(output_dir)
    OUTPUT_PATH = output_path
    TRIALS_CSV = OUTPUT_PATH / "random_search_trials.csv"
    REPLICATIONS_CSV = OUTPUT_PATH / "random_search_replications.csv"
    BEST_PARAMETERS_JSON = OUTPUT_PATH / "random_search_best_parameters.json"
    CONFIG_JSON = OUTPUT_PATH / "random_search_config.json"


def configure_experiment(
    n_trials: int | None = None,
    n_replications: int | None = None,
    base_seed: int | None = None,
    seed_step: int | None = None,
    output_dir: str | Path | None = None,
    run_index: int | None = None,
    run_duration: float | None = None,
    rate_multiplier: float | None = None,
    random_search_seed: int | None = None,
    objective_time_mode: str | None = None,
) -> None:
    global N_TRIALS, N_REPLICATIONS, BASE_SEED, SEED_STEP, RUN_INDEX
    global RUN_DURATION, RATE_MULTIPLIER, RANDOM_SEARCH_SEED, OBJECTIVE_TIME_MODE

    if n_trials is not None:
        N_TRIALS = int(n_trials)
    if n_replications is not None:
        N_REPLICATIONS = int(n_replications)
    if base_seed is not None:
        BASE_SEED = int(base_seed)
    if seed_step is not None:
        SEED_STEP = int(seed_step)
    if output_dir is not None:
        configure_output_paths(output_dir)
    if run_index is not None:
        RUN_INDEX = int(run_index)
    if run_duration is not None:
        RUN_DURATION = run_duration
    if rate_multiplier is not None:
        RATE_MULTIPLIER = float(rate_multiplier)
    if random_search_seed is not None:
        RANDOM_SEARCH_SEED = int(random_search_seed)
    if objective_time_mode is not None:
        OBJECTIVE_TIME_MODE = objective_time_mode


def reset_records() -> None:
    REPLICATION_RECORDS.clear()
    TRIAL_RECORDS.clear()


def sample_parameters(rng: np.random.Generator) -> dict[str, int]:
    return {
        name: int(rng.integers(lower, upper + 1))
        for name, (lower, upper) in PARAMETER_BOUNDS.items()
    }


def compute_objective(kpis: dict[str, Any], parameters: dict[str, int]) -> float:
    return float(
        calculate_objective_details(
            kpis,
            parameters,
            OBJECTIVE_WEIGHTS,
            OBJECTIVE_TIME_MODE,
        )["objective_value"]
    )


def run_replication(
    parameters: dict[str, int],
    trial_index: int,
    replication_index: int,
) -> dict[str, Any]:
    seed = BASE_SEED + trial_index * SEED_STEP + replication_index
    record: dict[str, Any] = {
        "method": METHOD,
        "run_index": RUN_INDEX,
        "trial_index": trial_index,
        "replication_index": replication_index,
        "seed": seed,
        **parameters,
    }

    try:
        kpis = simulate(
            **parameters,
            random_seed=seed,
            run_duration=RUN_DURATION,
            rate_multiplier=RATE_MULTIPLIER,
            animate=False,
            verbose=False,
        )
        error = ""
        if str(kpis.get("msg", "")).startswith("another exception"):
            error = str(kpis["msg"])
            objective_value = FAILED_RUN_PENALTY
            objective_details = {}
        else:
            objective_details = calculate_objective_details(
                kpis,
                parameters,
                OBJECTIVE_WEIGHTS,
                OBJECTIVE_TIME_MODE,
            )
            objective_value = objective_details["objective_value"]
        record.update(kpis)
        record.update(objective_details)
        record["objective_value"] = objective_value
        record["error"] = error
    except Exception as exc:
        logging.exception(
            "Replication failed for trial=%s replication=%s seed=%s",
            trial_index,
            replication_index,
            seed,
        )
        record["objective_value"] = FAILED_RUN_PENALTY
        record["error"] = repr(exc)

    return record


def evaluate_trial(parameters: dict[str, int], trial_index: int) -> dict[str, Any]:
    logging.info("Trial %s parameters: %s", trial_index, parameters)

    replication_records = [
        run_replication(parameters, trial_index, replication_index)
        for replication_index in range(N_REPLICATIONS)
    ]
    REPLICATION_RECORDS.extend(replication_records)

    objectives = np.asarray(
        [record["objective_value"] for record in replication_records],
        dtype=float,
    )
    valid_records = [record for record in replication_records if not record.get("error")]

    trial_record: dict[str, Any] = {
        "method": METHOD,
        "run_index": RUN_INDEX,
        "trial_index": trial_index,
        **parameters,
        "objective_mean": float(np.mean(objectives)),
        "objective_std": (
            float(np.std(objectives, ddof=1)) if len(objectives) > 1 else 0.0
        ),
        "n_valid_replications": len(valid_records),
    }

    for kpi in [
        "n_orders_completed",
        "n_orders_in_date",
        "n_orders_late",
        "n_orders_created",
        "n_orders_incomplete",
        "late_order_fraction",
        "time_in_system_mean",
        "time_in_system_std",
        "wip_time_in_system_mean",
        "wip_time_in_system_std",
        "wip_time_in_system_min",
        "wip_time_in_system_max",
        "wip_mean",
        "work_in_progress_mean",
        "completed_norm",
        "late_total_norm",
        "on_time_loss_norm",
        "station_capacity_norm",
        "worker_capacity_norm",
        "time_in_system_mean_norm",
        "time_in_system_std_norm",
        "objective_completed_contribution",
        "objective_late_contribution",
        "objective_station_capacity_contribution",
        "objective_worker_capacity_contribution",
        "objective_time_mean_contribution",
        "objective_time_std_contribution",
    ]:
        values = [
            float(record[kpi])
            for record in valid_records
            if kpi in record and record[kpi] is not None and not pd.isna(record[kpi])
        ]
        trial_record[f"{kpi}_mean"] = float(np.mean(values)) if values else np.nan

    previous_best = min(
        (record["objective_mean"] for record in TRIAL_RECORDS),
        default=math.inf,
    )
    trial_record["best_objective_so_far"] = min(
        previous_best,
        trial_record["objective_mean"],
    )

    TRIAL_RECORDS.append(trial_record)
    save_results()
    logging.info(
        "Trial %s objective mean %.3f std %.3f valid replications %s/%s best so far %.3f",
        trial_index,
        trial_record["objective_mean"],
        trial_record["objective_std"],
        trial_record["n_valid_replications"],
        N_REPLICATIONS,
        trial_record["best_objective_so_far"],
    )
    return trial_record


def save_results() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(REPLICATION_RECORDS).to_csv(REPLICATIONS_CSV, index=False)
    pd.DataFrame(TRIAL_RECORDS).to_csv(TRIALS_CSV, index=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def save_config() -> None:
    config = {
        "N_TRIALS": N_TRIALS,
        "N_REPLICATIONS": N_REPLICATIONS,
        "PARAMETER_BOUNDS": PARAMETER_BOUNDS,
        "OBJECTIVE_WEIGHTS": OBJECTIVE_WEIGHTS,
        "OBJECTIVE_TIME_MODE": OBJECTIVE_TIME_MODE,
        "VALID_OBJECTIVE_TIME_MODES": sorted(VALID_OBJECTIVE_TIME_MODES),
        "DUE_DATE_MEAN": DUE_DATE_MEAN,
        "RUN_DURATION": RUN_DURATION,
        "RATE_MULTIPLIER": RATE_MULTIPLIER,
        "BASE_SEED": BASE_SEED,
        "SEED_STEP": SEED_STEP,
        "RANDOM_SEARCH_SEED": RANDOM_SEARCH_SEED,
        "METHOD": METHOD,
        "RUN_INDEX": RUN_INDEX,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modules": {
            "random_search": Path(__file__).name,
            "simulation": "lab_analysis_simulation_BO.py",
        },
    }
    CONFIG_JSON.write_text(json.dumps(json_safe(config), indent=2), encoding="utf-8")


def save_best_result() -> dict[str, Any]:
    if not TRIAL_RECORDS:
        raise RuntimeError("No trial records available; cannot save best result.")

    best = min(TRIAL_RECORDS, key=lambda record: record["objective_mean"])
    best_result = {
        "method": METHOD,
        "run_index": RUN_INDEX,
        "best_parameters": {name: best[name] for name in PARAMETER_BOUNDS},
        "best_objective": best["objective_mean"],
        "aggregated_kpis": {
            key: value
            for key, value in best.items()
            if key.endswith("_mean") and key != "objective_mean"
        },
    }
    BEST_PARAMETERS_JSON.write_text(
        json.dumps(json_safe(best_result), indent=2),
        encoding="utf-8",
    )
    return best_result


def main() -> None:
    result = run_experiment()
    print("Best random-search parameters:")
    print(json.dumps(json_safe(result["best_result"]), indent=2))


def run_experiment(
    n_trials: int | None = None,
    n_replications: int | None = None,
    base_seed: int | None = None,
    output_dir: str | Path | None = None,
    run_index: int | None = None,
    seed_step: int | None = None,
    random_search_seed: int | None = None,
    run_duration: float | None = None,
    rate_multiplier: float | None = None,
    objective_time_mode: str | None = None,
) -> dict[str, Any]:
    configure_experiment(
        n_trials=n_trials,
        n_replications=n_replications,
        base_seed=base_seed,
        seed_step=seed_step,
        output_dir=output_dir,
        run_index=run_index,
        run_duration=run_duration,
        rate_multiplier=rate_multiplier,
        random_search_seed=random_search_seed,
        objective_time_mode=objective_time_mode,
    )
    reset_records()
    setup_logging()
    save_config()
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)

    for trial_index in range(N_TRIALS):
        parameters = sample_parameters(rng)
        evaluate_trial(parameters, trial_index)

    best_result = save_best_result()
    return {
        "method": METHOD,
        "run_index": RUN_INDEX,
        "output_path": OUTPUT_PATH,
        "trials": list(TRIAL_RECORDS),
        "replications": list(REPLICATION_RECORDS),
        "best_result": best_result,
    }


if __name__ == "__main__":
    main()
