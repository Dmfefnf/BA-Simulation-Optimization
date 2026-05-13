import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from ax.service.ax_client import AxClient
    from ax.service.utils.instantiation import ObjectiveProperties
except ImportError as exc:
    raise ImportError(
        "Ax Platform is required for BO.py. Install it with `pip install ax-platform` "
        "in the project environment."
    ) from exc

try:
    from lab_analysis_simulation_BO import DAY, simulate
except ImportError:
    from lab_analysis_simulation_BO import simulate

    MINUTE = 1
    HOUR = 60 * MINUTE
    DAY = 24 * HOUR


N_TRIALS = 30
N_REPLICATIONS = 5
RUN_DURATION = 1 * DAY
BASE_SEED = 12345
SEED_STEP = 1000
OUTPUT_DIR = "bo_results"
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

# These weights are experimental and should be calibrated for the business goal.
OBJECTIVE_WEIGHTS = {
    "time_in_system_mean": 1.0,
    "orders_completed": -10.0,
    "late_orders": 50.0,
    "worker_capacity": 5.0,
    "station_capacity": 2.0,
}

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / OUTPUT_DIR
TRIALS_CSV = OUTPUT_PATH / "bo_trials.csv"
REPLICATIONS_CSV = OUTPUT_PATH / "bo_replications.csv"
BEST_PARAMETERS_JSON = OUTPUT_PATH / "bo_best_parameters.json"
CONFIG_JSON = OUTPUT_PATH / "bo_config.json"

REPLICATION_RECORDS: list[dict[str, Any]] = []
TRIAL_RECORDS: list[dict[str, Any]] = []


def setup_logging() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_PATH / "bo.log", encoding="utf-8"),
        ],
    )


def build_ax_parameters() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "type": "range",
            "bounds": [int(bounds[0]), int(bounds[1])],
            "value_type": "int",
        }
        for name, bounds in PARAMETER_BOUNDS.items()
    ]


def sanitize_parameters(parameters: dict[str, Any]) -> dict[str, int]:
    sanitized = {}
    for name, (lower, upper) in PARAMETER_BOUNDS.items():
        value = int(round(float(parameters[name])))
        sanitized[name] = int(np.clip(value, lower, upper))
    return sanitized


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


def evaluate_trial(parameters: dict[str, Any], trial_index: int) -> dict[str, Any]:
    sanitized = sanitize_parameters(parameters)
    logging.info("Trial %s parameters: %s", trial_index, sanitized)

    replication_records = [
        run_replication(sanitized, trial_index, replication_index)
        for replication_index in range(N_REPLICATIONS)
    ]
    REPLICATION_RECORDS.extend(replication_records)

    objectives = np.asarray(
        [record["objective_value"] for record in replication_records],
        dtype=float,
    )
    valid_records = [
        record for record in replication_records if not record.get("error")
    ]

    trial_record: dict[str, Any] = {
        "trial_index": trial_index,
        **sanitized,
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

    TRIAL_RECORDS.append(trial_record)
    save_results()
    logging.info(
        "Trial %s objective mean %.3f std %.3f valid replications %s/%s",
        trial_index,
        trial_record["objective_mean"],
        trial_record["objective_std"],
        trial_record["n_valid_replications"],
        N_REPLICATIONS,
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
        "BASE_SEED": BASE_SEED,
        "SEED_STEP": SEED_STEP,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "modules": {
            "bo": Path(__file__).name,
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


def create_ax_client() -> AxClient:
    ax_client = AxClient()
    try:
        ax_client.create_experiment(
            name="lab_analysis_capacity_optimization",
            parameters=build_ax_parameters(),
            objectives={"objective": ObjectiveProperties(minimize=True)},
        )
    except TypeError:
        ax_client.create_experiment(
            name="lab_analysis_capacity_optimization",
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
) -> None:
    objective_sem = (
        objective_std / math.sqrt(N_REPLICATIONS) if N_REPLICATIONS > 0 else 0.0
    )
    try:
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={"objective": (objective_mean, objective_sem)},
        )
    except Exception:
        ax_client.complete_trial(trial_index=trial_index, raw_data=objective_mean)


def main() -> None:
    setup_logging()
    save_config()
    ax_client = create_ax_client()

    for _ in range(N_TRIALS):
        parameters, trial_index = ax_client.get_next_trial()
        trial_record = evaluate_trial(parameters, trial_index)
        complete_ax_trial(
            ax_client,
            trial_index,
            trial_record["objective_mean"],
            trial_record["objective_std"],
        )

    best_result = save_best_result()
    print("Best parameters:")
    print(json.dumps(json_safe(best_result), indent=2))


if __name__ == "__main__":
    main()
