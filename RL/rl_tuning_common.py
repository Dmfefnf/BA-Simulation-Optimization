"""Shared helpers for laptop-friendly staged RL hyperparameter tuning."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from RL_agents import QLearningAgent
from RL_experiment import (
    BASE_SEED,
    CSV_FIELDS,
    RATE_MULTIPLIER,
    RUN_DURATION,
    SEED_STEP,
    SIM_RESULT_FIELDS,
    STATION_CAPACITIES,
    append_csv_row,
    json_safe,
    make_eval_seeds,
    make_training_seed,
    run_safely,
)
from lab_analysis_simulation_RL import DEFAULT_RISK_T1, DEFAULT_RISK_WINDOW, simulate_rl

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = BASE_DIR / "rl_tuning_hpc"
AGENT_RANDOM_SEED_OFFSET = 600_000
FAILED_RUN_PENALTY = 1e9

# Set USE_FIXED_RISK to False to tune risk_t1 and risk_window together with the
# Q-learning hyperparameters.
USE_FIXED_RISK = True
FIXED_RISK_T1 = 175.214912
FIXED_RISK_WINDOW = 758.004 # Values: 120.0 or 758.004

# Bounds are intentionally compact for staged laptop/HPC use. Epsilon decay is
# derived from target_final_epsilon and the training episode budget.
# PARAMETER_BOUNDS = {
#     "alpha": (0.02, 0.5),
#     "gamma": (0.70, 0.99),
#     "target_final_epsilon": (0.02, 0.40),
#     "risk_t1": (-1 * 60.0, 2 * 60.0),
#     "risk_window": (0.5 * 60.0, 6 * 60.0),
# }
# PARAMETER_BOUNDS = {
#     "alpha": (0.005, 0.5),
#     "gamma": (0.50, 0.99),
#     "target_final_epsilon": (0.005, 0.30),
#     "risk_t1": (0.0, 360.0),
#     "risk_window": (30.0, 1440.0),
# }
# PARAMETER_BOUNDS = {
#     "alpha": (0.005, 0.7),
#     "gamma": (0.20, 0.95),
#     "target_final_epsilon": (0.02, 0.22),
#     "risk_t1": (40.0, 360.0),
#     "risk_window": (120.0, 1800.0),
# }
PARAMETER_BOUNDS = {
    "alpha": (0.0001, 0.08), # before: (0.001, 0.08)
    "gamma": (0.0, 0.2), # before: (0.20, 0.95)
    "target_final_epsilon": (0.0005, 0.08), # before: (0.005, 0.08)
    "risk_t1": (40.0, 360.0),
    "risk_window": (120.0, 1800.0),
}
RISK_PARAMETER_NAMES = ["risk_t1", "risk_window"]
PARAMETER_NAMES = [
    name
    for name in PARAMETER_BOUNDS
    if not USE_FIXED_RISK or name not in RISK_PARAMETER_NAMES
]
ACTIVE_PARAMETER_BOUNDS = {
    name: PARAMETER_BOUNDS[name]
    for name in PARAMETER_NAMES
}
DERIVED_PARAMETER_NAMES = ["epsilon_decay", "epsilon_min"]
LOG_PARAMETER_NAMES = [*PARAMETER_BOUNDS, *DERIVED_PARAMETER_NAMES]

TRAINING_FIELDS = [
    "episode",
    "training_seed",
    "q_table_size",
    "best_total_reward_so_far",
    *LOG_PARAMETER_NAMES,
    *SIM_RESULT_FIELDS,
    "label",
    "error",
]
EVALUATION_FIELDS = [
    "replication",
    "eval_seed",
    "objective_value",
    "q_table_size",
    *LOG_PARAMETER_NAMES,
    *CSV_FIELDS,
]


def unique_fieldnames(fields: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for field in fields:
        if field not in seen:
            result.append(field)
            seen.add(field)
    return result


TRAINING_FIELDS = unique_fieldnames(TRAINING_FIELDS)
EVALUATION_FIELDS = unique_fieldnames(EVALUATION_FIELDS)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    result = Path(path)
    if result.is_absolute():
        return result
    return BASE_DIR / result


def target_epsilon_decay(target_final_epsilon: float, training_episodes: int) -> float:
    if training_episodes < 1:
        raise ValueError("training_episodes must be >= 1.")
    return float(target_final_epsilon ** (1.0 / training_episodes))


def sanitize_parameters(
    parameters: dict[str, Any],
    training_episodes: int,
) -> dict[str, float]:
    sanitized: dict[str, float] = {}
    for name, (lower, upper) in ACTIVE_PARAMETER_BOUNDS.items():
        if name not in parameters:
            raise ValueError(f"Missing parameter '{name}'.")
        sanitized[name] = float(np.clip(float(parameters[name]), lower, upper))
    if USE_FIXED_RISK:
        sanitized["risk_t1"] = float(FIXED_RISK_T1)
        sanitized["risk_window"] = float(FIXED_RISK_WINDOW)
    sanitized["risk_window"] = max(1.0, sanitized["risk_window"])
    sanitized["epsilon_min"] = sanitized["target_final_epsilon"]
    sanitized["epsilon_decay"] = target_epsilon_decay(
        sanitized["target_final_epsilon"],
        training_episodes,
    )
    return sanitized


def q_learning_config(parameters: dict[str, float]) -> dict[str, float]:
    return {
        "alpha": parameters["alpha"],
        "gamma": parameters["gamma"],
        "epsilon": 1.0,
        "epsilon_decay": parameters["epsilon_decay"],
        "epsilon_min": parameters["epsilon_min"],
    }


def simulation_kwargs(
    parameters: dict[str, float],
    run_duration: float,
    rate_multiplier: float,
) -> dict[str, Any]:
    return {
        "run_duration": run_duration,
        "rate_multiplier": rate_multiplier,
        "risk_t1": parameters["risk_t1"],
        "risk_window": parameters["risk_window"],
        **STATION_CAPACITIES,
    }


def value_as_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def train_q_agent(
    parameters: dict[str, float],
    output_dir: Path,
    training_episodes: int,
    base_seed: int,
    seed_step: int,
    run_duration: float,
    rate_multiplier: float,
    training_csv_name: str | None,
) -> tuple[QLearningAgent, dict[str, Any]]:
    agent = QLearningAgent(
        random_seed=base_seed + AGENT_RANDOM_SEED_OFFSET,
        **q_learning_config(parameters),
    )
    kwargs = simulation_kwargs(parameters, run_duration, rate_multiplier)
    best_training: dict[str, Any] | None = None
    last_training: dict[str, Any] | None = None
    n_failures = 0

    for episode in range(training_episodes):
        seed = make_training_seed(episode, base_seed, seed_step)
        row = run_safely(
            f"rl_training_{episode}",
            lambda seed=seed: simulate_rl(
                agent=agent,
                agent_type="q_learning",
                training=True,
                random_seed=seed,
                **kwargs,
            ),
        )
        if row.get("error"):
            n_failures += 1
        reward = value_as_float(row.get("total_reward"))
        if reward is not None and (
            best_training is None or reward > best_training["total_reward"]
        ):
            best_training = {
                "episode": episode,
                "training_seed": seed,
                "total_reward": reward,
            }

        last_training = {
            "episode": episode,
            "training_seed": seed,
            "total_reward": reward,
            "epsilon": agent.epsilon,
            "q_table_size": len(agent.q_table),
        }
        row.update(
            {
                "episode": episode,
                "training_seed": seed,
                "q_table_size": len(agent.q_table),
                "best_total_reward_so_far": (
                    best_training["total_reward"] if best_training else ""
                ),
                **parameters,
            }
        )
        if training_csv_name is not None:
            append_csv_row(row, output_dir / training_csv_name, TRAINING_FIELDS)
        agent.decay_epsilon()

    summary = {
        "training_episodes": training_episodes,
        "n_training_failures": n_failures,
        "best_training": best_training,
        "last_training": last_training,
        "final_epsilon": agent.epsilon,
        "q_table_size": len(agent.q_table),
    }
    return agent, summary


def evaluate_q_agent(
    agent: QLearningAgent,
    parameters: dict[str, float],
    output_dir: Path,
    eval_replications: int,
    base_seed: int,
    seed_step: int,
    run_duration: float,
    rate_multiplier: float,
) -> dict[str, Any]:
    kwargs = simulation_kwargs(parameters, run_duration, rate_multiplier)
    eval_seeds = make_eval_seeds(eval_replications, base_seed, seed_step)
    objective_values: list[float] = []
    rewards: list[float] = []
    late_fractions: list[float] = []
    best_eval: dict[str, Any] | None = None
    n_valid = 0

    for rep, seed in enumerate(eval_seeds):
        row = run_safely(
            f"rl_evaluation_{rep}",
            lambda seed=seed: simulate_rl(
                agent=agent,
                agent_type="q_learning",
                training=False,
                random_seed=seed,
                **kwargs,
            ),
        )
        reward = value_as_float(row.get("total_reward"))
        if row.get("error") or reward is None:
            objective_value = FAILED_RUN_PENALTY
        else:
            objective_value = -reward
            rewards.append(reward)
            n_valid += 1
            late_fraction = value_as_float(row.get("late_order_fraction"))
            if late_fraction is not None:
                late_fractions.append(late_fraction)
            if best_eval is None or reward > best_eval["total_reward"]:
                best_eval = {
                    "replication": rep,
                    "eval_seed": seed,
                    "total_reward": reward,
                }
        objective_values.append(objective_value)
        row.update(
            {
                "replication": rep,
                "eval_seed": seed,
                "objective_value": objective_value,
                "q_table_size": len(agent.q_table),
                **parameters,
            }
        )
        append_csv_row(row, output_dir / "evaluation.csv", EVALUATION_FIELDS)

    objective_array = np.asarray(objective_values, dtype=float)
    reward_array = np.asarray(rewards, dtype=float)
    return {
        "objective": (
            "minimize -mean(total_reward); late_order_fraction is recorded "
            "for later composite objectives"
        ),
        "objective_mean": float(np.mean(objective_array)),
        "objective_std": (
            float(np.std(objective_array, ddof=1)) if len(objective_array) > 1 else 0.0
        ),
        "total_reward_mean": float(np.mean(reward_array)) if len(reward_array) else math.nan,
        "total_reward_std": (
            float(np.std(reward_array, ddof=1)) if len(reward_array) > 1 else 0.0
        ),
        "late_order_fraction_mean": (
            float(np.mean(late_fractions)) if late_fractions else math.nan
        ),
        "n_valid_replications": n_valid,
        "eval_replications": eval_replications,
        "eval_seeds": eval_seeds,
        "best_evaluation": best_eval,
    }


def base_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_duration": args.run_duration,
        "rate_multiplier": args.rate_multiplier,
        "base_seed": args.base_seed,
        "seed_step": args.seed_step,
        "agent_random_seed_offset": AGENT_RANDOM_SEED_OFFSET,
        "station_capacities": STATION_CAPACITIES,
        "use_fixed_risk": USE_FIXED_RISK,
        "fixed_risk": {
            "risk_t1": FIXED_RISK_T1,
            "risk_window": FIXED_RISK_WINDOW,
            "risk_t2": FIXED_RISK_T1 + FIXED_RISK_WINDOW,
        },
        "parameter_names": PARAMETER_NAMES,
        "log_parameter_names": LOG_PARAMETER_NAMES,
        "parameter_bounds": ACTIVE_PARAMETER_BOUNDS,
        "all_parameter_bounds": PARAMETER_BOUNDS,
        "parameter_bounds_note": (
            "BO tunes target_final_epsilon. epsilon_min is fixed to "
            "target_final_epsilon and epsilon_decay is computed as "
            "target_final_epsilon ** (1 / training_episodes). When "
            "use_fixed_risk is true, risk_t1 and risk_window are logged but "
            "not included in the Ax search space."
        ),
        "default_risk_t1": DEFAULT_RISK_T1,
        "default_risk_window": DEFAULT_RISK_WINDOW,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
