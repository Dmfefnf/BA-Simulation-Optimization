"""Run baseline, random-agent and Q-learning experiments for the RL simulation."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from RL_agents import BaselineAgent, QLearningAgent, RandomAgent
from lab_analysis_simulation_RL import (
    DAY,
    DEFAULT_RISK_T1,
    DEFAULT_RISK_WINDOW,
    DUE_DATE_LOWER_BOUND,
    DUE_DATE_UPPER_BOUND,
    RATE_MULTIPLIER,
    simulate_rl,
)

BASE_DIR = Path(__file__).resolve().parent

N_BASELINE_REPLICATIONS = 30
N_RANDOM_REPLICATIONS = 30
N_TRAINING_EPISODES = 1000
FINAL_TRAINING_EPISODES = 10_000
N_EVAL_REPLICATIONS = 30

RUN_DURATION = 1 * DAY
BASE_SEED = 12345
SEED_STEP = 1000
TRAINING_SEED_OFFSET = 200_000
EVAL_SEED_OFFSET = 300_000
RANDOM_AGENT_ACTION_SEED_OFFSET = 500_000

OUTPUT_DIR = BASE_DIR / "rl_results_test"
FINAL_OUTPUT_DIR = BASE_DIR / "rl_results_final_10000"

STATION_CAPACITIES = {
    "preparation_capacity": 1,
    "sorting_capacity": 1,
    "analysis1_capacity": 1,
    "analysis2_capacity": 1,
    "evaluation_capacity": 1,
    "dispatching_capacity": 1,
    "worker_capacity": 1,
}

Q_LEARNING_CONFIG = {
    "alpha": 0.1,
    "gamma": 0.95,
    "epsilon": 1.0,
    "epsilon_decay": 0.999, # Default 0.995
    "epsilon_min": 0.05,
}

RISK_CONFIG = {
    "risk_t1": DEFAULT_RISK_T1,
    "risk_window": DEFAULT_RISK_WINDOW,
}

BASELINE_ACTIONS = {
    0: "FIFO",
    1: "EDD",
    2: "Longest Waiting Time",
    3: "Highest Lateness Risk",
}

SIM_RESULT_FIELDS = [
    "random_seed",
    "run_duration",
    "due_date_lower_bound",
    "due_date_upper_bound",
    "rate_multiplier",
    "risk_t1",
    "risk_window",
    "risk_t2",
    "agent_type",
    "training",
    "fixed_action",
    "msg",
    "n_orders_created",
    "n_orders_completed",
    "n_orders_in_date",
    "n_orders_late",
    "late_order_fraction",
    "time_in_system_mean",
    "time_in_system_std",
    "time_in_system_min",
    "time_in_system_max",
    "wip_mean",
    "wip_max",
    "total_reward",
    "n_eval_failures",
    "n_decision_points",
    "action_0_count",
    "action_1_count",
    "action_2_count",
    "action_3_count",
    "epsilon",
    "preparation_capacity",
    "sorting_capacity",
    "analysis1_capacity",
    "analysis2_capacity",
    "evaluation_capacity",
    "dispatching_capacity",
    "worker_capacity",
]
CSV_FIELDS = [
    "label",
    "error",
    "replication",
    "eval_seed",
    "episode",
    "baseline_rule",
    "action_random_seed",
    "q_table_size",
    "best_total_reward_so_far",
    *SIM_RESULT_FIELDS,
]

EVAL_SEEDS = [
    BASE_SEED + EVAL_SEED_OFFSET + rep * SEED_STEP
    for rep in range(N_EVAL_REPLICATIONS)
]

LOGGER = logging.getLogger(__name__)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(val) for val in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def make_eval_seeds(
    n_replications: int,
    base_seed: int = BASE_SEED,
    seed_step: int = SEED_STEP,
) -> list[int]:
    """Shared simulation seeds for all evaluation-style comparisons."""

    return [
        base_seed + EVAL_SEED_OFFSET + rep * seed_step
        for rep in range(n_replications)
    ]


def make_training_seed(
    episode: int,
    base_seed: int = BASE_SEED,
    seed_step: int = SEED_STEP,
) -> int:
    return base_seed + TRAINING_SEED_OFFSET + episode * seed_step


def make_random_action_seed(
    replication: int,
    base_seed: int = BASE_SEED,
    seed_step: int = SEED_STEP,
) -> int:
    return base_seed + RANDOM_AGENT_ACTION_SEED_OFFSET + replication * seed_step


def run_safely(label: str, fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Run one episode and return a CSV-friendly error row on failure."""

    try:
        result = fn()
        result["error"] = ""
        return result
    except Exception as exc:
        LOGGER.exception("%s failed", label)
        return {"label": label, "msg": "error", "error": repr(exc)}


def append_csv_row(row: dict[str, Any], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    safe_row = {key: json_safe(value) for key, value in row.items()}
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
            restval="",
        )
        if write_header:
            writer.writeheader()
        writer.writerow(safe_row)


def reset_csv_outputs(output_path: Path) -> None:
    for file_name in [
        "baselines.csv",
        "random_agent.csv",
        "q_learning_training.csv",
        "q_learning_evaluation.csv",
    ]:
        path = output_path / file_name
        if path.exists():
            path.unlink()


def valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "error" not in df.columns:
        return df
    return df[df["error"].fillna("") == ""]


def best_reward_from_frame(
    df: pd.DataFrame,
    include_fields: list[str],
) -> dict[str, Any] | None:
    if df.empty or "total_reward" not in df.columns:
        return None
    rewards = pd.to_numeric(df["total_reward"], errors="coerce")
    if rewards.dropna().empty:
        return None
    idx = rewards.idxmax()
    row = df.loc[idx]
    result: dict[str, Any] = {"total_reward": float(rewards.loc[idx])}
    for field in include_fields:
        if field in row and not pd.isna(row[field]):
            value = row[field]
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            result[field] = value
    return result


def resolve_output_path(output_dir: str | Path | None, final_run: bool) -> Path:
    if output_dir is None:
        if final_run:
            return FINAL_OUTPUT_DIR
        return OUTPUT_DIR

    path = Path(output_dir)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def setup_logging(output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_path / "rl_experiment.log", encoding="utf-8"),
        ],
        force=True,
    )


def run_baselines(
    output_path: Path,
    eval_seeds: list[int],
    n_replications: int,
    simulation_kwargs: dict[str, Any],
) -> None:
    for action, rule_name in BASELINE_ACTIONS.items():
        for rep in range(n_replications):
            seed = eval_seeds[rep]
            row = run_safely(
                f"baseline_{rule_name}_{rep}",
                lambda action=action, seed=seed: simulate_rl(
                    agent=BaselineAgent(fixed_action=action),
                    agent_type="baseline",
                    fixed_action=action,
                    training=False,
                    random_seed=seed,
                    **simulation_kwargs,
                ),
            )
            row.update(
                {
                    "baseline_rule": rule_name,
                    "replication": rep,
                    "eval_seed": seed,
                }
            )
            append_csv_row(row, output_path / "baselines.csv", CSV_FIELDS)


def run_random_agent(
    output_path: Path,
    eval_seeds: list[int],
    n_replications: int,
    simulation_kwargs: dict[str, Any],
    base_seed: int,
    seed_step: int,
) -> None:
    for rep in range(n_replications):
        seed = eval_seeds[rep]
        action_seed = make_random_action_seed(rep, base_seed, seed_step)
        row = run_safely(
            f"random_agent_{rep}",
            lambda seed=seed, action_seed=action_seed: simulate_rl(
                agent=RandomAgent(random_seed=action_seed),
                agent_type="random",
                training=False,
                random_seed=seed,
                **simulation_kwargs,
            ),
        )
        row.update(
            {
                "replication": rep,
                "eval_seed": seed,
                "action_random_seed": action_seed,
            }
        )
        append_csv_row(row, output_path / "random_agent.csv", CSV_FIELDS)


def train_q_learning(
    q_agent: QLearningAgent,
    output_path: Path,
    n_training_episodes: int,
    simulation_kwargs: dict[str, Any],
    base_seed: int,
    seed_step: int,
) -> dict[str, Any] | None:
    best_training: dict[str, Any] | None = None

    for episode in range(n_training_episodes):
        seed = make_training_seed(episode, base_seed, seed_step)
        row = run_safely(
            f"q_learning_training_{episode}",
            lambda seed=seed: simulate_rl(
                agent=q_agent,
                agent_type="q_learning",
                training=True,
                random_seed=seed,
                **simulation_kwargs,
            ),
        )
        reward = pd.to_numeric(pd.Series([row.get("total_reward")]), errors="coerce").iloc[0]
        if not pd.isna(reward) and (
            best_training is None or float(reward) > best_training["total_reward"]
        ):
            best_training = {
                "episode": episode,
                "random_seed": seed,
                "total_reward": float(reward),
            }

        row.update(
            {
                "episode": episode,
                "q_table_size": len(q_agent.q_table),
                "best_total_reward_so_far": (
                    best_training["total_reward"] if best_training else ""
                ),
            }
        )
        append_csv_row(row, output_path / "q_learning_training.csv", CSV_FIELDS)
        q_agent.decay_epsilon()

    return best_training


def evaluate_q_learning(
    q_agent: QLearningAgent,
    output_path: Path,
    eval_seeds: list[int],
    n_replications: int,
    simulation_kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    best_eval: dict[str, Any] | None = None

    for rep in range(n_replications):
        seed = eval_seeds[rep]
        row = run_safely(
            f"q_learning_evaluation_{rep}",
            lambda seed=seed: simulate_rl(
                agent=q_agent,
                agent_type="q_learning",
                training=False,
                random_seed=seed,
                **simulation_kwargs,
            ),
        )
        reward = pd.to_numeric(pd.Series([row.get("total_reward")]), errors="coerce").iloc[0]
        if not pd.isna(reward) and (
            best_eval is None or float(reward) > best_eval["total_reward"]
        ):
            best_eval = {
                "replication": rep,
                "eval_seed": seed,
                "total_reward": float(reward),
            }

        row.update(
            {
                "replication": rep,
                "eval_seed": seed,
                "q_table_size": len(q_agent.q_table),
            }
        )
        append_csv_row(row, output_path / "q_learning_evaluation.csv", CSV_FIELDS)

    return best_eval


def save_config(
    output_path: Path,
    run_mode: str,
    n_training_episodes: int,
    eval_seeds: list[int],
    args: argparse.Namespace,
    simulation_kwargs: dict[str, Any],
) -> None:
    config = {
        "run_mode": run_mode,
        "N_BASELINE_REPLICATIONS": args.baseline_replications,
        "N_RANDOM_REPLICATIONS": args.random_replications,
        "N_TRAINING_EPISODES": n_training_episodes,
        "FINAL_TRAINING_EPISODES": FINAL_TRAINING_EPISODES,
        "N_EVAL_REPLICATIONS": args.eval_replications,
        "RUN_DURATION": args.run_duration,
        "RATE_MULTIPLIER": args.rate_multiplier,
        "DUE_DATE_LOWER_BOUND": DUE_DATE_LOWER_BOUND,
        "DUE_DATE_UPPER_BOUND": DUE_DATE_UPPER_BOUND,
        "BASE_SEED": args.base_seed,
        "SEED_STEP": args.seed_step,
        "TRAINING_SEED_OFFSET": TRAINING_SEED_OFFSET,
        "EVAL_SEED_OFFSET": EVAL_SEED_OFFSET,
        "EVAL_SEEDS": eval_seeds,
        "RANDOM_AGENT_SEEDING_NOTE": (
            "RandomAgent uses EVAL_SEEDS for simulation randomness like the "
            "baselines and Q-learning evaluation; action randomness uses a "
            "separate deterministic seed per replication."
        ),
        "STATION_CAPACITIES": STATION_CAPACITIES,
        "Q_LEARNING_CONFIG": Q_LEARNING_CONFIG,
        "RISK_CONFIG": {
            "risk_t1": simulation_kwargs["risk_t1"],
            "risk_window": simulation_kwargs["risk_window"],
            "risk_t2": simulation_kwargs["risk_t1"] + simulation_kwargs["risk_window"],
        },
        "BASELINE_ACTIONS": BASELINE_ACTIONS,
        "skip_baselines": args.skip_baselines,
        "skip_random": args.skip_random,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (output_path / "rl_config.json").write_text(
        json.dumps(json_safe(config), indent=2),
        encoding="utf-8",
    )


def summarize_results(
    output_path: Path,
    best_training: dict[str, Any] | None,
    best_eval: dict[str, Any] | None,
) -> dict[str, Any]:
    baselines_path = output_path / "baselines.csv"
    q_eval_path = output_path / "q_learning_evaluation.csv"
    random_path = output_path / "random_agent.csv"

    summary: dict[str, Any] = {
        "best_training_reward": best_training,
        "best_q_evaluation_reward": best_eval,
    }

    if baselines_path.exists():
        baselines = valid_rows(pd.read_csv(baselines_path))
        if not baselines.empty:
            by_rule = baselines.groupby("baseline_rule")["total_reward"].mean()
            summary["baseline_mean_total_reward"] = by_rule.to_dict()
            best_rule = by_rule.idxmax()
            summary["best_baseline_rule"] = {
                "baseline_rule": best_rule,
                "mean_total_reward": float(by_rule[best_rule]),
            }
            print(
                "Best baseline by mean total_reward: "
                f"{best_rule} ({by_rule[best_rule]:.2f})"
            )

    if random_path.exists():
        random_eval = valid_rows(pd.read_csv(random_path))
        if not random_eval.empty:
            summary["random_agent_mean_total_reward"] = float(
                random_eval["total_reward"].mean()
            )

    if q_eval_path.exists():
        q_eval = valid_rows(pd.read_csv(q_eval_path))
        if not q_eval.empty:
            summary["q_learning_mean_total_reward"] = float(
                q_eval["total_reward"].mean()
            )
            summary["q_learning_mean_late_order_fraction"] = float(
                q_eval["late_order_fraction"].mean()
            )
            summary["best_q_evaluation_reward"] = best_reward_from_frame(
                q_eval,
                ["replication", "eval_seed", "random_seed"],
            )
            print(
                "Q-learning evaluation mean total_reward: "
                f"{summary['q_learning_mean_total_reward']:.2f}"
            )
            print(
                "Q-learning evaluation mean late fraction: "
                f"{summary['q_learning_mean_late_order_fraction']:.4f}"
            )

    (output_path / "rl_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(f"Results written to: {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-episodes", type=int, default=None)
    parser.add_argument(
        "--final-run",
        action="store_true",
        help="Use 10000 training episodes and the final-run output directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-replications", type=int, default=N_BASELINE_REPLICATIONS)
    parser.add_argument("--random-replications", type=int, default=N_RANDOM_REPLICATIONS)
    parser.add_argument("--eval-replications", type=int, default=N_EVAL_REPLICATIONS)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--run-duration", type=float, default=RUN_DURATION)
    parser.add_argument("--rate-multiplier", type=float, default=RATE_MULTIPLIER)
    parser.add_argument("--risk-t1", type=float, default=RISK_CONFIG["risk_t1"])
    parser.add_argument("--risk-window", type=float, default=RISK_CONFIG["risk_window"])
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.risk_window <= 0:
        raise ValueError("--risk-window must be positive.")

    n_training_episodes = (
        FINAL_TRAINING_EPISODES if args.final_run else N_TRAINING_EPISODES
    )
    if args.training_episodes is not None:
        n_training_episodes = args.training_episodes

    run_mode = "final" if args.final_run else "standard"
    output_path = resolve_output_path(args.output_dir, args.final_run)
    setup_logging(output_path)
    reset_csv_outputs(output_path)

    max_eval_seeds = max(
        args.eval_replications,
        0 if args.skip_baselines else args.baseline_replications,
        0 if args.skip_random else args.random_replications,
    )
    eval_seeds = make_eval_seeds(max_eval_seeds, args.base_seed, args.seed_step)

    simulation_kwargs = {
        "run_duration": args.run_duration,
        "rate_multiplier": args.rate_multiplier,
        "risk_t1": args.risk_t1,
        "risk_window": args.risk_window,
        **STATION_CAPACITIES,
    }
    save_config(output_path, run_mode, n_training_episodes, eval_seeds, args, simulation_kwargs)

    LOGGER.info(
        "Starting %s RL run with %s training episodes, %s evaluation replications",
        run_mode,
        n_training_episodes,
        args.eval_replications,
    )

    if not args.skip_baselines:
        run_baselines(
            output_path,
            eval_seeds,
            args.baseline_replications,
            simulation_kwargs,
        )

    if not args.skip_random:
        run_random_agent(
            output_path,
            eval_seeds,
            args.random_replications,
            simulation_kwargs,
            args.base_seed,
            args.seed_step,
        )

    q_agent = QLearningAgent(random_seed=args.base_seed, **Q_LEARNING_CONFIG)
    best_training = train_q_learning(
        q_agent,
        output_path,
        n_training_episodes,
        simulation_kwargs,
        args.base_seed,
        args.seed_step,
    )
    q_agent.save(output_path / "q_table.pkl")

    best_eval = evaluate_q_learning(
        q_agent,
        output_path,
        eval_seeds,
        args.eval_replications,
        simulation_kwargs,
    )
    q_agent.save(output_path / "q_table_final.pkl")

    summary = summarize_results(output_path, best_training, best_eval)
    if summary.get("best_training_reward"):
        best = summary["best_training_reward"]
        print(
            "Best training reward: "
            f"{best['total_reward']:.2f} at episode {best['episode']} "
            f"(seed {best['random_seed']})"
        )
    if summary.get("best_q_evaluation_reward"):
        best = summary["best_q_evaluation_reward"]
        print(
            "Best Q-evaluation reward: "
            f"{best['total_reward']:.2f} at replication {best['replication']} "
            f"(seed {best['eval_seed']})"
        )


if __name__ == "__main__":
    main()
