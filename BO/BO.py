import json
import logging
import math
import random
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
BO_RANDOM_SEED = 24680
OUTPUT_DIR = "bo_results"
FAILED_RUN_PENALTY = 1e9
# Time modes: "none" (A), "mean" (B), "mean_std" (C).
OBJECTIVE_TIME_MODE = "mean"
METHOD = "bo"
RUN_INDEX = 0
SAVE_AFTER_EACH_TRIAL = True

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
TRIALS_CSV = OUTPUT_PATH / "bo_trials.csv"
REPLICATIONS_CSV = OUTPUT_PATH / "bo_replications.csv"
BEST_PARAMETERS_JSON = OUTPUT_PATH / "bo_best_parameters.json"
CONFIG_JSON = OUTPUT_PATH / "bo_config.json"

REPLICATION_RECORDS: list[dict[str, Any]] = []
TRIAL_RECORDS: list[dict[str, Any]] = []

LARGE_REPLICATION_FIELDS = {
    "queue_preparation_length_tx",
    "queue_preparation_length_resampled",
}


def setup_logging() -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_PATH / "bo.log", encoding="utf-8"),
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
    TRIALS_CSV = OUTPUT_PATH / "bo_trials.csv"
    REPLICATIONS_CSV = OUTPUT_PATH / "bo_replications.csv"
    BEST_PARAMETERS_JSON = OUTPUT_PATH / "bo_best_parameters.json"
    CONFIG_JSON = OUTPUT_PATH / "bo_config.json"


def configure_experiment(
    n_trials: int | None = None,
    n_replications: int | None = None,
    base_seed: int | None = None,
    seed_step: int | None = None,
    output_dir: str | Path | None = None,
    run_index: int | None = None,
    run_duration: float | None = None,
    rate_multiplier: float | None = None,
    bo_random_seed: int | None = None,
    objective_time_mode: str | None = None,
    save_after_each_trial: bool | None = None,
) -> None:
    global N_TRIALS, N_REPLICATIONS, BASE_SEED, SEED_STEP, RUN_INDEX
    global RUN_DURATION, RATE_MULTIPLIER, BO_RANDOM_SEED, OBJECTIVE_TIME_MODE
    global SAVE_AFTER_EACH_TRIAL

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
    if bo_random_seed is not None:
        BO_RANDOM_SEED = int(bo_random_seed)
    if objective_time_mode is not None:
        OBJECTIVE_TIME_MODE = objective_time_mode
    if save_after_each_trial is not None:
        SAVE_AFTER_EACH_TRIAL = bool(save_after_each_trial)


def reset_records() -> None:
    REPLICATION_RECORDS.clear()
    TRIAL_RECORDS.clear()


def remove_large_replication_fields(record: dict[str, Any]) -> None:
    for field in LARGE_REPLICATION_FIELDS:
        record.pop(field, None)

# Create Ax parameter definitions based on PARAMETER_BOUNDS
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

# Ensure parameters are integers within bounds
def sanitize_parameters(parameters: dict[str, Any]) -> dict[str, int]:
    sanitized = {}
    for name, (lower, upper) in PARAMETER_BOUNDS.items():
        value = int(round(float(parameters[name])))
        sanitized[name] = int(np.clip(value, lower, upper))
    return sanitized

# Compute the objective value based on KPIs and parameters, applying weights and penalties.
def compute_objective(kpis: dict[str, Any], parameters: dict[str, int]) -> float:
    return float(
        calculate_objective_details(
            kpis,
            parameters,
            OBJECTIVE_WEIGHTS,
            OBJECTIVE_TIME_MODE,
        )["objective_value"]
    )

# Run a single replication of the simulation with the given parameters and return the results.
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
        remove_large_replication_fields(record)
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

# Evaluate a trial by running multiple replications, aggregating results, and saving records.
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
        "method": METHOD,
        "run_index": RUN_INDEX,
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
    if SAVE_AFTER_EACH_TRIAL:
        save_results()
    logging.info(
        "Run %s trial %s objective mean %.3f std %.3f valid replications %s/%s best so far %.3f",
        RUN_INDEX,
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
        "BO_RANDOM_SEED": BO_RANDOM_SEED,
        "METHOD": METHOD,
        "RUN_INDEX": RUN_INDEX,
        "SAVE_AFTER_EACH_TRIAL": SAVE_AFTER_EACH_TRIAL,
        "LARGE_REPLICATION_FIELDS_REMOVED": sorted(LARGE_REPLICATION_FIELDS),
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

# Create and initialize the AxClient with the experiment definition.
def create_ax_client(random_seed: int | None = None) -> AxClient:
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    try:
        ax_client = AxClient(random_seed=random_seed)
    except TypeError:
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

# Complete the Ax trial with the computed objective mean and standard error, handling any exceptions gracefully.
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
    result = run_experiment()
    print("Best parameters:")
    print(json.dumps(json_safe(result["best_result"]), indent=2))


def run_experiment(
    n_trials: int | None = None,
    n_replications: int | None = None,
    base_seed: int | None = None,
    output_dir: str | Path | None = None,
    run_index: int | None = None,
    seed_step: int | None = None,
    bo_random_seed: int | None = None,
    run_duration: float | None = None,
    rate_multiplier: float | None = None,
    objective_time_mode: str | None = None,
    save_after_each_trial: bool | None = None,
    return_records: bool = True,
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
        bo_random_seed=bo_random_seed,
        objective_time_mode=objective_time_mode,
        save_after_each_trial=save_after_each_trial,
    )
    reset_records()
    setup_logging()
    save_config()
    ax_client = create_ax_client(BO_RANDOM_SEED)

    for _ in range(N_TRIALS):
        parameters, trial_index = ax_client.get_next_trial()
        trial_record = evaluate_trial(parameters, trial_index)
        complete_ax_trial(
            ax_client,
            trial_index,
            trial_record["objective_mean"],
            trial_record["objective_std"],
        )

    if not SAVE_AFTER_EACH_TRIAL:
        save_results()

    best_result = save_best_result()
    trials = list(TRIAL_RECORDS) if return_records else []
    replications = list(REPLICATION_RECORDS) if return_records else []
    if not return_records:
        reset_records()
    logging.shutdown()

    return {
        "method": METHOD,
        "run_index": RUN_INDEX,
        "output_path": OUTPUT_PATH,
        "trials": trials,
        "replications": replications,
        "best_result": best_result,
    }


if __name__ == "__main__":
    main()
