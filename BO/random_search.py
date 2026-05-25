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
except ImportError:
    from lab_analysis_simulation_BO import simulate

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

PARAMETER_BOUNDS = {
    "preparation_capacity": (1, 5),
    "sorting_capacity": (1, 5),
    "analysis1_capacity": (1, 5),
    "analysis2_capacity": (1, 5),
    "evaluation_capacity": (1, 5),
    "dispatching_capacity": (1, 5),
    "worker_capacity": (1, 10),
}

# Keep this identical to BO.py when comparing BO and random search.
OBJECTIVE_WEIGHTS = {
    "time_in_system_mean": 1.0,
    "orders_completed": -10.0,
    "late_orders": 50.0,
    "worker_capacity": 5.0,
    "station_capacity": 2.0,
}

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
    )


def sample_parameters(rng: np.random.Generator) -> dict[str, int]:
    return {
        name: int(rng.integers(lower, upper + 1))
        for name, (lower, upper) in PARAMETER_BOUNDS.items()
    }


def compute_objective(kpis: dict[str, Any], parameters: dict[str, int]) -> float:
    time_in_system_mean = kpis.get("time_in_system_mean", np.nan)
    if time_in_system_mean is None or np.isnan(float(time_in_system_mean)):
        return FAILED_RUN_PENALTY

    total_station_capacity = sum(
        parameters[name] for name in PARAMETER_BOUNDS if name != "worker_capacity"
    )

    return float(
        OBJECTIVE_WEIGHTS["time_in_system_mean"] * float(time_in_system_mean)
        + OBJECTIVE_WEIGHTS["orders_completed"]
        * float(kpis.get("n_orders_completed", 0))
        + OBJECTIVE_WEIGHTS["late_orders"] * float(kpis.get("n_orders_late", 0))
        + OBJECTIVE_WEIGHTS["worker_capacity"] * float(parameters["worker_capacity"])
        + OBJECTIVE_WEIGHTS["station_capacity"] * float(total_station_capacity)
    )


def run_replication(
    parameters: dict[str, int],
    trial_index: int,
    replication_index: int,
) -> dict[str, Any]:
    seed = BASE_SEED + trial_index * SEED_STEP + replication_index
    record: dict[str, Any] = {
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
        else:
            objective_value = compute_objective(kpis, parameters)
        record.update(kpis)
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
        "n_orders_late",
        "late_order_fraction",
        "time_in_system_mean",
        "wip_mean",
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
        "RUN_DURATION": RUN_DURATION,
        "RATE_MULTIPLIER": RATE_MULTIPLIER,
        "BASE_SEED": BASE_SEED,
        "SEED_STEP": SEED_STEP,
        "RANDOM_SEARCH_SEED": RANDOM_SEARCH_SEED,
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
    setup_logging()
    save_config()
    rng = np.random.default_rng(RANDOM_SEARCH_SEED)

    for trial_index in range(N_TRIALS):
        parameters = sample_parameters(rng)
        evaluate_trial(parameters, trial_index)

    best_result = save_best_result()
    print("Best random-search parameters:")
    print(json.dumps(json_safe(best_result), indent=2))


if __name__ == "__main__":
    main()
