"""Staged Ax tuning for Q-learning and lateness-risk dispatch parameters.

Stage 1 tunes compact training runs. Stage 2 retrains the best configuration(s)
with a larger episode budget and evaluates them on the same evaluation seeds.

Ax minimizes ``objective = -mean(total_reward)`` so larger rewards are better.
"""

from __future__ import annotations

import argparse
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
        "Ax Platform is required for RL_hyperparameter_tuning.py. Install it with "
        "`pip install ax-platform` in the project environment."
    ) from exc

from RL_agents import QLearningAgent
from RL_experiment import (
    BASE_SEED,
    CSV_FIELDS,
    FINAL_TRAINING_EPISODES,
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

N_TRIALS = 30
STAGE1_TRAINING_EPISODES = 1000
N_EVAL_REPLICATIONS = 30
N_FINAL_CONFIGS = 1
BO_RANDOM_SEED = 24680
AGENT_RANDOM_SEED_OFFSET = 600_000
OUTPUT_DIR = BASE_DIR / "rl_bo_results"
FAILED_RUN_PENALTY = 1e9
SAVE_AFTER_EACH_TRIAL = True

PARAMETER_BOUNDS = {
    "alpha": (0.02, 0.5),
    "gamma": (0.70, 0.99),
    "epsilon_decay": (0.980, 0.9999),
    "epsilon_min": (0.01, 0.20),
    "risk_t1": (-1 * 60.0, 2 * 60.0),
    "risk_window": (0.5 * 60.0, 6 * 60.0),
}

PARAMETER_NAMES = list(PARAMETER_BOUNDS)
TRIALS_CSV_NAME = "rl_bo_trials.csv"
EVALUATION_CSV_NAME = "rl_bo_evaluation_records.csv"
FINAL_RESULTS_CSV_NAME = "rl_bo_final_results.csv"
FINAL_TRAINING_CSV_NAME = "rl_bo_final_training.csv"
BEST_PARAMETERS_JSON_NAME = "rl_bo_best_parameters.json"
CONFIG_JSON_NAME = "rl_bo_config.json"

TRIAL_FIELDS = [
    "stage",
    "trial_index",
    *PARAMETER_NAMES,
    "objective_mean",
    "objective_std",
    "best_objective_so_far",
    "total_reward_mean",
    "total_reward_std",
    "late_order_fraction_mean",
    "n_valid_replications",
    "best_training_reward",
    "best_training_episode",
    "best_training_seed",
    "q_table_size",
]
EVALUATION_FIELDS = [
    "stage",
    "trial_index",
    "final_config_rank",
    "replication",
    "eval_seed",
    "objective_value",
    *PARAMETER_NAMES,
    *CSV_FIELDS,
]
FINAL_RESULT_FIELDS = [
    "stage",
    "final_config_rank",
    "source_trial_index",
    *PARAMETER_NAMES,
    "objective_mean",
    "objective_std",
    "total_reward_mean",
    "total_reward_std",
    "late_order_fraction_mean",
    "n_valid_replications",
    "best_training_reward",
    "best_training_episode",
    "best_training_seed",
    "best_evaluation_reward",
    "best_evaluation_replication",
    "best_evaluation_seed",
    "q_table_size",
    "q_table_path",
]
FINAL_TRAINING_FIELDS = [
    "stage",
    "final_config_rank",
    "source_trial_index",
    "episode",
    "best_total_reward_so_far",
    "q_table_size",
    *PARAMETER_NAMES,
    *SIM_RESULT_FIELDS,
    "label",
    "error",
]


def unique_fieldnames(fields: list[str]) -> list[str]:
    result = []
    seen = set()
    for field in fields:
        if field not in seen:
            result.append(field)
            seen.add(field)
    return result


TRIAL_FIELDS = unique_fieldnames(TRIAL_FIELDS)
EVALUATION_FIELDS = unique_fieldnames(EVALUATION_FIELDS)
FINAL_RESULT_FIELDS = unique_fieldnames(FINAL_RESULT_FIELDS)
FINAL_TRAINING_FIELDS = unique_fieldnames(FINAL_TRAINING_FIELDS)

LOGGER = logging.getLogger(__name__)
TRIAL_RECORDS: list[dict[str, Any]] = []
FINAL_RECORDS: list[dict[str, Any]] = []


def setup_logging(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_path / "rl_bo.log", encoding="utf-8"),
        ],
        force=True,
    )


def reset_outputs(output_path: Path) -> None:
    for file_name in [
        TRIALS_CSV_NAME,
        EVALUATION_CSV_NAME,
        FINAL_RESULTS_CSV_NAME,
        FINAL_TRAINING_CSV_NAME,
        BEST_PARAMETERS_JSON_NAME,
        CONFIG_JSON_NAME,
    ]:
        path = output_path / file_name
        if path.exists():
            path.unlink()


def resolve_output_path(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return OUTPUT_DIR
    path = Path(output_dir)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def build_ax_parameters() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "type": "range",
            "bounds": [float(bounds[0]), float(bounds[1])],
            "value_type": "float",
        }
        for name, bounds in PARAMETER_BOUNDS.items()
    ]


def sanitize_parameters(parameters: dict[str, Any]) -> dict[str, float]:
    sanitized: dict[str, float] = {}
    for name, (lower, upper) in PARAMETER_BOUNDS.items():
        value = float(parameters[name])
        sanitized[name] = float(np.clip(value, lower, upper))
    sanitized["risk_window"] = max(1.0, sanitized["risk_window"])
    return sanitized


def build_q_config(parameters: dict[str, float]) -> dict[str, float]:
    return {
        "alpha": parameters["alpha"],
        "gamma": parameters["gamma"],
        "epsilon": 1.0,
        "epsilon_decay": parameters["epsilon_decay"],
        "epsilon_min": parameters["epsilon_min"],
    }


def build_simulation_kwargs(
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


def train_agent(
    parameters: dict[str, float],
    n_training_episodes: int,
    base_seed: int,
    seed_step: int,
    run_duration: float,
    rate_multiplier: float,
    output_path: Path | None = None,
    final_config_rank: int | None = None,
    source_trial_index: int | None = None,
) -> tuple[QLearningAgent, dict[str, Any] | None]:
    q_agent = QLearningAgent(
        random_seed=base_seed + AGENT_RANDOM_SEED_OFFSET,
        **build_q_config(parameters),
    )
    simulation_kwargs = build_simulation_kwargs(parameters, run_duration, rate_multiplier)
    best_training: dict[str, Any] | None = None

    for episode in range(n_training_episodes):
        seed = make_training_seed(episode, base_seed, seed_step)
        row = run_safely(
            f"rl_bo_training_{source_trial_index}_{episode}",
            lambda seed=seed: simulate_rl(
                agent=q_agent,
                agent_type="q_learning",
                training=True,
                random_seed=seed,
                **simulation_kwargs,
            ),
        )
        reward = value_as_float(row.get("total_reward"))
        if reward is not None and (
            best_training is None or reward > best_training["total_reward"]
        ):
            best_training = {
                "episode": episode,
                "random_seed": seed,
                "total_reward": reward,
            }

        row.update(
            {
                "stage": "stage2_final",
                "final_config_rank": final_config_rank,
                "source_trial_index": source_trial_index,
                "episode": episode,
                "best_total_reward_so_far": (
                    best_training["total_reward"] if best_training else ""
                ),
                "q_table_size": len(q_agent.q_table),
                **parameters,
            }
        )
        if output_path is not None:
            append_csv_row(
                row,
                output_path / FINAL_TRAINING_CSV_NAME,
                FINAL_TRAINING_FIELDS,
            )
        q_agent.decay_epsilon()

    return q_agent, best_training


def evaluate_agent(
    q_agent: QLearningAgent,
    parameters: dict[str, float],
    eval_seeds: list[int],
    n_eval_replications: int,
    output_path: Path,
    stage: str,
    trial_index: int | None,
    final_config_rank: int | None,
    run_duration: float,
    rate_multiplier: float,
) -> dict[str, Any]:
    simulation_kwargs = build_simulation_kwargs(parameters, run_duration, rate_multiplier)
    objective_values: list[float] = []
    reward_values: list[float] = []
    late_fractions: list[float] = []
    best_eval: dict[str, Any] | None = None
    n_valid = 0

    for rep in range(n_eval_replications):
        seed = eval_seeds[rep]
        row = run_safely(
            f"rl_bo_eval_{stage}_{trial_index}_{rep}",
            lambda seed=seed: simulate_rl(
                agent=q_agent,
                agent_type="q_learning",
                training=False,
                random_seed=seed,
                **simulation_kwargs,
            ),
        )
        error = row.get("error", "")
        reward = value_as_float(row.get("total_reward"))
        if error or reward is None:
            objective_value = FAILED_RUN_PENALTY
        else:
            objective_value = -reward
            n_valid += 1
            reward_values.append(reward)
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
                "stage": stage,
                "trial_index": trial_index,
                "final_config_rank": final_config_rank,
                "replication": rep,
                "eval_seed": seed,
                "objective_value": objective_value,
                **parameters,
            }
        )
        append_csv_row(row, output_path / EVALUATION_CSV_NAME, EVALUATION_FIELDS)

    objective_array = np.asarray(objective_values, dtype=float)
    reward_array = np.asarray(reward_values, dtype=float)
    return {
        "objective_mean": float(np.mean(objective_array)),
        "objective_std": (
            float(np.std(objective_array, ddof=1))
            if len(objective_array) > 1
            else 0.0
        ),
        "total_reward_mean": float(np.mean(reward_array)) if len(reward_array) else np.nan,
        "total_reward_std": (
            float(np.std(reward_array, ddof=1)) if len(reward_array) > 1 else 0.0
        ),
        "late_order_fraction_mean": (
            float(np.mean(late_fractions)) if late_fractions else np.nan
        ),
        "n_valid_replications": n_valid,
        "best_evaluation": best_eval,
    }


def evaluate_trial(
    parameters: dict[str, Any],
    trial_index: int,
    output_path: Path,
    eval_seeds: list[int],
    n_training_episodes: int,
    n_eval_replications: int,
    base_seed: int,
    seed_step: int,
    run_duration: float,
    rate_multiplier: float,
) -> dict[str, Any]:
    sanitized = sanitize_parameters(parameters)
    logging.info("Stage 1 trial %s parameters: %s", trial_index, sanitized)

    q_agent, best_training = train_agent(
        sanitized,
        n_training_episodes,
        base_seed,
        seed_step,
        run_duration,
        rate_multiplier,
    )
    evaluation = evaluate_agent(
        q_agent,
        sanitized,
        eval_seeds,
        n_eval_replications,
        output_path,
        "stage1_bo",
        trial_index,
        None,
        run_duration,
        rate_multiplier,
    )

    previous_best = min(
        (record["objective_mean"] for record in TRIAL_RECORDS),
        default=math.inf,
    )
    trial_record = {
        "stage": "stage1_bo",
        "trial_index": trial_index,
        **sanitized,
        "objective_mean": evaluation["objective_mean"],
        "objective_std": evaluation["objective_std"],
        "best_objective_so_far": min(previous_best, evaluation["objective_mean"]),
        "total_reward_mean": evaluation["total_reward_mean"],
        "total_reward_std": evaluation["total_reward_std"],
        "late_order_fraction_mean": evaluation["late_order_fraction_mean"],
        "n_valid_replications": evaluation["n_valid_replications"],
        "best_training_reward": (
            best_training["total_reward"] if best_training else np.nan
        ),
        "best_training_episode": best_training["episode"] if best_training else np.nan,
        "best_training_seed": best_training["random_seed"] if best_training else np.nan,
        "q_table_size": len(q_agent.q_table),
    }
    TRIAL_RECORDS.append(trial_record)
    append_csv_row(trial_record, output_path / TRIALS_CSV_NAME, TRIAL_FIELDS)
    logging.info(
        "Stage 1 trial %s objective %.3f reward mean %.3f valid %s/%s",
        trial_index,
        trial_record["objective_mean"],
        trial_record["total_reward_mean"],
        trial_record["n_valid_replications"],
        n_eval_replications,
    )
    return trial_record


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
            name="rl_q_learning_and_lateness_risk_tuning",
            parameters=build_ax_parameters(),
            objectives={"objective": ObjectiveProperties(minimize=True)},
        )
    except TypeError:
        ax_client.create_experiment(
            name="rl_q_learning_and_lateness_risk_tuning",
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
    n_eval_replications: int,
) -> None:
    objective_sem = (
        objective_std / math.sqrt(n_eval_replications)
        if n_eval_replications > 0
        else 0.0
    )
    try:
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={"objective": (objective_mean, objective_sem)},
        )
    except Exception:
        ax_client.complete_trial(trial_index=trial_index, raw_data=objective_mean)


def save_config(
    output_path: Path,
    args: argparse.Namespace,
    eval_seeds: list[int],
    run_final_stage: bool,
) -> None:
    config = {
        "N_TRIALS": args.n_trials,
        "STAGE1_TRAINING_EPISODES": args.training_episodes,
        "FINAL_TRAINING_EPISODES": args.final_training_episodes,
        "N_EVAL_REPLICATIONS": args.eval_replications,
        "N_FINAL_CONFIGS": args.n_final_configs,
        "PARAMETER_BOUNDS": PARAMETER_BOUNDS,
        "OBJECTIVE": "minimize -mean(total_reward) over evaluation seeds",
        "RUN_DURATION": args.run_duration,
        "RATE_MULTIPLIER": args.rate_multiplier,
        "BASE_SEED": args.base_seed,
        "SEED_STEP": args.seed_step,
        "BO_RANDOM_SEED": args.bo_random_seed,
        "AGENT_RANDOM_SEED_OFFSET": AGENT_RANDOM_SEED_OFFSET,
        "EVAL_SEEDS": eval_seeds,
        "DEFAULT_RISK_T1": DEFAULT_RISK_T1,
        "DEFAULT_RISK_WINDOW": DEFAULT_RISK_WINDOW,
        "STATION_CAPACITIES": STATION_CAPACITIES,
        "SAVE_AFTER_EACH_TRIAL": SAVE_AFTER_EACH_TRIAL,
        "run_final_stage": run_final_stage,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (output_path / CONFIG_JSON_NAME).write_text(
        json.dumps(json_safe(config), indent=2),
        encoding="utf-8",
    )


def save_best_result(output_path: Path) -> dict[str, Any]:
    if not TRIAL_RECORDS:
        raise RuntimeError("No trial records available; cannot save best result.")

    best = min(TRIAL_RECORDS, key=lambda record: record["objective_mean"])
    result = {
        "objective": "Ax minimizes -mean(total_reward); lower objective is better.",
        "stage1_best_parameters": {name: best[name] for name in PARAMETER_NAMES},
        "stage1_best_objective": best["objective_mean"],
        "stage1_best_total_reward_mean": best["total_reward_mean"],
        "stage1_best_trial_index": best["trial_index"],
        "stage2_final_results": FINAL_RECORDS,
    }
    (output_path / BEST_PARAMETERS_JSON_NAME).write_text(
        json.dumps(json_safe(result), indent=2),
        encoding="utf-8",
    )
    return result


def run_final_stage(
    output_path: Path,
    eval_seeds: list[int],
    n_final_configs: int,
    final_training_episodes: int,
    n_eval_replications: int,
    base_seed: int,
    seed_step: int,
    run_duration: float,
    rate_multiplier: float,
) -> None:
    best_records = sorted(TRIAL_RECORDS, key=lambda record: record["objective_mean"])[
        :n_final_configs
    ]

    for rank, source_record in enumerate(best_records, start=1):
        parameters = {name: float(source_record[name]) for name in PARAMETER_NAMES}
        logging.info(
            "Stage 2 final rank %s from trial %s parameters: %s",
            rank,
            source_record["trial_index"],
            parameters,
        )
        q_agent, best_training = train_agent(
            parameters,
            final_training_episodes,
            base_seed,
            seed_step,
            run_duration,
            rate_multiplier,
            output_path=output_path,
            final_config_rank=rank,
            source_trial_index=int(source_record["trial_index"]),
        )
        q_table_path = output_path / f"q_table_stage2_rank_{rank}.pkl"
        q_agent.save(q_table_path)

        evaluation = evaluate_agent(
            q_agent,
            parameters,
            eval_seeds,
            n_eval_replications,
            output_path,
            "stage2_final",
            int(source_record["trial_index"]),
            rank,
            run_duration,
            rate_multiplier,
        )
        best_eval = evaluation["best_evaluation"] or {}
        final_record = {
            "stage": "stage2_final",
            "final_config_rank": rank,
            "source_trial_index": int(source_record["trial_index"]),
            **parameters,
            "objective_mean": evaluation["objective_mean"],
            "objective_std": evaluation["objective_std"],
            "total_reward_mean": evaluation["total_reward_mean"],
            "total_reward_std": evaluation["total_reward_std"],
            "late_order_fraction_mean": evaluation["late_order_fraction_mean"],
            "n_valid_replications": evaluation["n_valid_replications"],
            "best_training_reward": (
                best_training["total_reward"] if best_training else np.nan
            ),
            "best_training_episode": (
                best_training["episode"] if best_training else np.nan
            ),
            "best_training_seed": (
                best_training["random_seed"] if best_training else np.nan
            ),
            "best_evaluation_reward": best_eval.get("total_reward", np.nan),
            "best_evaluation_replication": best_eval.get("replication", np.nan),
            "best_evaluation_seed": best_eval.get("eval_seed", np.nan),
            "q_table_size": len(q_agent.q_table),
            "q_table_path": q_table_path,
        }
        FINAL_RECORDS.append(final_record)
        append_csv_row(final_record, output_path / FINAL_RESULTS_CSV_NAME, FINAL_RESULT_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--training-episodes", type=int, default=STAGE1_TRAINING_EPISODES)
    parser.add_argument("--final-training-episodes", type=int, default=FINAL_TRAINING_EPISODES)
    parser.add_argument("--eval-replications", type=int, default=N_EVAL_REPLICATIONS)
    parser.add_argument("--n-final-configs", type=int, default=N_FINAL_CONFIGS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    parser.add_argument("--bo-random-seed", type=int, default=BO_RANDOM_SEED)
    parser.add_argument("--run-duration", type=float, default=RUN_DURATION)
    parser.add_argument("--rate-multiplier", type=float, default=RATE_MULTIPLIER)
    parser.add_argument("--skip-final-stage", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def default_args() -> argparse.Namespace:
    return argparse.Namespace(
        n_trials=N_TRIALS,
        training_episodes=STAGE1_TRAINING_EPISODES,
        final_training_episodes=FINAL_TRAINING_EPISODES,
        eval_replications=N_EVAL_REPLICATIONS,
        n_final_configs=N_FINAL_CONFIGS,
        output_dir=None,
        base_seed=BASE_SEED,
        seed_step=SEED_STEP,
        bo_random_seed=BO_RANDOM_SEED,
        run_duration=RUN_DURATION,
        rate_multiplier=RATE_MULTIPLIER,
        skip_final_stage=False,
        smoke_test=False,
    )


def run_experiment(
    n_trials: int | None = None,
    training_episodes: int | None = None,
    final_training_episodes: int | None = None,
    eval_replications: int | None = None,
    output_dir: str | Path | None = None,
    skip_final_stage: bool | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    args = default_args()
    if n_trials is not None:
        args.n_trials = n_trials
    if training_episodes is not None:
        args.training_episodes = training_episodes
    if final_training_episodes is not None:
        args.final_training_episodes = final_training_episodes
    if eval_replications is not None:
        args.eval_replications = eval_replications
    if output_dir is not None:
        args.output_dir = Path(output_dir)
    if skip_final_stage is not None:
        args.skip_final_stage = skip_final_stage
    if smoke_test:
        args.smoke_test = True

    return run_with_args(args)


def run_with_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke_test:
        args.n_trials = min(args.n_trials, 2)
        args.training_episodes = min(args.training_episodes, 3)
        args.final_training_episodes = min(args.final_training_episodes, 5)
        args.eval_replications = min(args.eval_replications, 2)
        args.n_final_configs = min(args.n_final_configs, 1)
        if args.output_dir is None:
            args.output_dir = Path("rl_bo_results_smoke")

    output_path = resolve_output_path(args.output_dir)
    setup_logging(output_path)
    reset_outputs(output_path)
    TRIAL_RECORDS.clear()
    FINAL_RECORDS.clear()

    eval_seeds = make_eval_seeds(args.eval_replications, args.base_seed, args.seed_step)
    run_final = not args.skip_final_stage
    save_config(output_path, args, eval_seeds, run_final)
    ax_client = create_ax_client(args.bo_random_seed)

    for _ in range(args.n_trials):
        parameters, trial_index = ax_client.get_next_trial()
        trial_record = evaluate_trial(
            parameters,
            trial_index,
            output_path,
            eval_seeds,
            args.training_episodes,
            args.eval_replications,
            args.base_seed,
            args.seed_step,
            args.run_duration,
            args.rate_multiplier,
        )
        complete_ax_trial(
            ax_client,
            trial_index,
            trial_record["objective_mean"],
            trial_record["objective_std"],
            args.eval_replications,
        )
        if SAVE_AFTER_EACH_TRIAL:
            save_best_result(output_path)

    if run_final:
        run_final_stage(
            output_path,
            eval_seeds,
            args.n_final_configs,
            args.final_training_episodes,
            args.eval_replications,
            args.base_seed,
            args.seed_step,
            args.run_duration,
            args.rate_multiplier,
        )

    best_result = save_best_result(output_path)
    logging.shutdown()
    return {
        "output_path": output_path,
        "trials": list(TRIAL_RECORDS),
        "final_results": list(FINAL_RECORDS),
        "best_result": best_result,
    }


def main() -> None:
    result = run_with_args(parse_args())
    print("Best RL tuning result:")
    print(json.dumps(json_safe(result["best_result"]), indent=2))
    print(f"Results written to: {result['output_path']}")


if __name__ == "__main__":
    main()
