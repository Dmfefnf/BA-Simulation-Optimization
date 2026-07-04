import math
from typing import Any


MINUTE = 1
HOUR = 60 * MINUTE
DUE_DATE_MEAN = 6 * HOUR

VALID_OBJECTIVE_TIME_MODES = {"none", "mean", "mean_std"}

# Normalized default objective weights. Late/lost orders are weighted higher than
# completed orders, station capacity is more expensive than worker capacity, and
# time in system is a small-to-medium optional penalty controlled by the caller's
# objective time mode.
DEFAULT_OBJECTIVE_WEIGHTS = {
    "completed": 1.0,
    "late_total": 3.0,
    "station_capacity": 0.8,
    "worker_capacity": 0.35,
    "time_in_system_mean": 0.4,
    "time_in_system_std": 0.2,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _finite_or_nan(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def calculate_objective_details(
    kpis: dict[str, Any],
    parameters: dict[str, int],
    objective_weights: dict[str, float],
    objective_time_mode: str,
) -> dict[str, Any]:
    """Return the normalized objective terms, contributions, and final cost."""
    if objective_time_mode not in VALID_OBJECTIVE_TIME_MODES:
        raise ValueError(
            f"Unknown objective_time_mode={objective_time_mode!r}; "
            f"use one of {sorted(VALID_OBJECTIVE_TIME_MODES)}."
        )

    denom_orders = max(int(_safe_float(kpis.get("n_orders_created"), 0.0)), 1)
    n_orders_completed = _safe_float(kpis.get("n_orders_completed"), 0.0)
    n_orders_in_date = _safe_float(kpis.get("n_orders_in_date"), 0.0)
    n_orders_late = _safe_float(kpis.get("n_orders_late"), 0.0)
    n_orders_incomplete = _safe_float(kpis.get("n_orders_incomplete"), 0.0)

    station_capacity_sum = sum(
        int(parameters[name]) for name in parameters if name != "worker_capacity"
    )
    worker_capacity = int(parameters["worker_capacity"])

    completed_norm = n_orders_completed / denom_orders
    late_total_norm = (n_orders_late + n_orders_incomplete) / denom_orders
    on_time_loss_norm = 1.0 - (n_orders_in_date / denom_orders)
    station_capacity_norm = (station_capacity_sum - 6) / (30 - 6)
    worker_capacity_norm = (worker_capacity - 1) / (10 - 1)

    time_in_system_mean = _finite_or_nan(kpis.get("time_in_system_mean"))
    time_in_system_std = _finite_or_nan(kpis.get("time_in_system_std"))
    time_in_system_mean_norm = time_in_system_mean / DUE_DATE_MEAN
    time_in_system_std_norm = time_in_system_std / DUE_DATE_MEAN

    completed_contribution = -objective_weights["completed"] * completed_norm
    late_contribution = objective_weights["late_total"] * late_total_norm
    station_contribution = (
        objective_weights["station_capacity"] * station_capacity_norm
    )
    worker_contribution = objective_weights["worker_capacity"] * worker_capacity_norm
    time_mean_contribution = 0.0
    time_std_contribution = 0.0

    objective_value = (
        completed_contribution
        + late_contribution
        + station_contribution
        + worker_contribution
    )
    objective_missing_terms = []

    if objective_time_mode in {"mean", "mean_std"}:
        if math.isnan(time_in_system_mean_norm):
            objective_missing_terms.append("time_in_system_mean")
        else:
            time_mean_contribution = (
                objective_weights["time_in_system_mean"]
                * time_in_system_mean_norm
            )
            objective_value += time_mean_contribution

    if objective_time_mode == "mean_std":
        if math.isnan(time_in_system_std_norm):
            objective_missing_terms.append("time_in_system_std")
        else:
            time_std_contribution = (
                objective_weights["time_in_system_std"] * time_in_system_std_norm
            )
            objective_value += time_std_contribution

    return {
        "denom_orders": denom_orders,
        "completed_norm": completed_norm,
        "late_total_norm": late_total_norm,
        "on_time_loss_norm": on_time_loss_norm,
        "station_capacity_sum": station_capacity_sum,
        "station_capacity_norm": station_capacity_norm,
        "worker_capacity_norm": worker_capacity_norm,
        "time_in_system_mean_norm": time_in_system_mean_norm,
        "time_in_system_std_norm": time_in_system_std_norm,
        "objective_completed_contribution": completed_contribution,
        "objective_late_contribution": late_contribution,
        "objective_station_capacity_contribution": station_contribution,
        "objective_worker_capacity_contribution": worker_contribution,
        "objective_time_mean_contribution": time_mean_contribution,
        "objective_time_std_contribution": time_std_contribution,
        "objective_time_mode": objective_time_mode,
        "objective_missing_terms": ",".join(objective_missing_terms),
        "objective_value": float(objective_value),
    }
