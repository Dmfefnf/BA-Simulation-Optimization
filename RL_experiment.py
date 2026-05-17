"""Run baseline, random-agent and Q-learning experiments for the RL simulation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from RL_agents import BaselineAgent, QLearningAgent, RandomAgent
from lab_analysis_simulation_RL import DAY, simulate_rl

BASE_DIR = Path(__file__).resolve().parent

N_BASELINE_REPLICATIONS = 5
N_RANDOM_REPLICATIONS = 3
N_TRAINING_EPISODES = 100
N_EVAL_REPLICATIONS = 10

RUN_DURATION = 5 * DAY
BASE_SEED = 12345
SEED_STEP = 1000

OUTPUT_DIR = BASE_DIR / "rl_results"

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
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
}

BASELINE_ACTIONS = {
    0: "FIFO",
    1: "EDD",
    2: "Longest Waiting Time",
    3: "Highest Lateness Risk",
}

LOGGER = logging.getLogger(__name__)


def run_safely(label: str, fn: Callable[[], dict]) -> dict:
    """Run one episode and return a CSV-friendly error row on failure."""

    try:
        result = fn()
        result["error"] = ""
        return result
    except Exception as exc:
        LOGGER.exception("%s failed", label)
        return {"label": label, "msg": "error", "error": str(exc)}


def save_csv(rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def summarize_results(baselines: pd.DataFrame, q_eval: pd.DataFrame) -> None:
    valid_baselines = baselines[baselines.get("error", "") == ""]
    if not valid_baselines.empty:
        by_rule = valid_baselines.groupby("baseline_rule")["total_reward"].mean()
        best_rule = by_rule.idxmax()
        print(f"Best baseline by mean total_reward: {best_rule} ({by_rule[best_rule]:.2f})")

        for rule in ["FIFO", "EDD"]:
            if rule in by_rule.index:
                print(f"{rule} mean total_reward: {by_rule[rule]:.2f}")

    valid_q_eval = q_eval[q_eval.get("error", "") == ""]
    if not valid_q_eval.empty:
        print(
            "Q-learning evaluation mean total_reward: "
            f"{valid_q_eval['total_reward'].mean():.2f}"
        )
        print(
            "Q-learning evaluation mean late fraction: "
            f"{valid_q_eval['late_order_fraction'].mean():.4f}"
        )

    print(f"Results written to: {OUTPUT_DIR}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "N_BASELINE_REPLICATIONS": N_BASELINE_REPLICATIONS,
        "N_RANDOM_REPLICATIONS": N_RANDOM_REPLICATIONS,
        "N_TRAINING_EPISODES": N_TRAINING_EPISODES,
        "N_EVAL_REPLICATIONS": N_EVAL_REPLICATIONS,
        "RUN_DURATION": RUN_DURATION,
        "BASE_SEED": BASE_SEED,
        "SEED_STEP": SEED_STEP,
        "STATION_CAPACITIES": STATION_CAPACITIES,
        "Q_LEARNING_CONFIG": Q_LEARNING_CONFIG,
    }
    with (OUTPUT_DIR / "rl_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    baseline_rows = []
    for action, rule_name in BASELINE_ACTIONS.items():
        for rep in range(N_BASELINE_REPLICATIONS):
            seed = BASE_SEED + action * SEED_STEP * 10 + rep * SEED_STEP
            row = run_safely(
                f"baseline_{rule_name}_{rep}",
                lambda action=action, seed=seed: simulate_rl(
                    agent=BaselineAgent(fixed_action=action),
                    agent_type="baseline",
                    fixed_action=action,
                    training=False,
                    run_duration=RUN_DURATION,
                    random_seed=seed,
                    **STATION_CAPACITIES,
                ),
            )
            row["baseline_rule"] = rule_name
            row["replication"] = rep
            baseline_rows.append(row)
    baselines = save_csv(baseline_rows, OUTPUT_DIR / "baselines.csv")

    random_rows = []
    for rep in range(N_RANDOM_REPLICATIONS):
        seed = BASE_SEED + 100_000 + rep * SEED_STEP
        row = run_safely(
            f"random_agent_{rep}",
            lambda seed=seed: simulate_rl(
                agent=RandomAgent(random_seed=seed),
                agent_type="random",
                training=False,
                run_duration=RUN_DURATION,
                random_seed=seed,
                **STATION_CAPACITIES,
            ),
        )
        row["replication"] = rep
        random_rows.append(row)
    save_csv(random_rows, OUTPUT_DIR / "random_agent.csv")

    q_agent = QLearningAgent(random_seed=BASE_SEED, **Q_LEARNING_CONFIG)
    training_rows = []
    for episode in range(N_TRAINING_EPISODES):
        seed = BASE_SEED + 200_000 + episode * SEED_STEP
        row = run_safely(
            f"q_learning_training_{episode}",
            lambda seed=seed: simulate_rl(
                agent=q_agent,
                agent_type="q_learning",
                training=True,
                run_duration=RUN_DURATION,
                random_seed=seed,
                **STATION_CAPACITIES,
            ),
        )
        row["episode"] = episode
        row["q_table_size"] = len(q_agent.q_table)
        training_rows.append(row)
        q_agent.decay_epsilon()
    save_csv(training_rows, OUTPUT_DIR / "q_learning_training.csv")
    q_agent.save(OUTPUT_DIR / "q_table.pkl")

    q_eval_rows = []
    for rep in range(N_EVAL_REPLICATIONS):
        seed = BASE_SEED + 300_000 + rep * SEED_STEP
        row = run_safely(
            f"q_learning_evaluation_{rep}",
            lambda seed=seed: simulate_rl(
                agent=q_agent,
                agent_type="q_learning",
                training=False,
                run_duration=RUN_DURATION,
                random_seed=seed,
                **STATION_CAPACITIES,
            ),
        )
        row["replication"] = rep
        row["q_table_size"] = len(q_agent.q_table)
        q_eval_rows.append(row)
    q_eval = save_csv(q_eval_rows, OUTPUT_DIR / "q_learning_evaluation.csv")

    summarize_results(baselines, q_eval)


if __name__ == "__main__":
    main()
