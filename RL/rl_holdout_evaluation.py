"""Independent holdout evaluation for trained RL dispatching policies.

The holdout uses new simulation seeds and never retrains agents. It is intended
to re-evaluate the selected policies after Stage 2 without reusing the original
30 evaluation seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from RL_agents import BaselineAgent, QLearningAgent, RandomAgent
from RL_experiment import SIM_RESULT_FIELDS, json_safe, run_safely
from lab_analysis_simulation_RL import simulate_rl


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
STANDARD_RESULTS_DIR = RESULTS_DIR / "rl_standard_results"
TUNING_RESULTS_DIR = RESULTS_DIR / "rl_tuning_results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "rl_holdout_results" / "holdout_100_seeds"

DEFAULT_N_REPLICATIONS = 100
DEFAULT_EVAL_SEED_BASE = 900_000
DEFAULT_SEED_STEP = 1_000
DEFAULT_ACTION_SEED_BASE = 1_900_000
POLICY_ACTION_SEED_STRIDE = 100_000

BASELINE_ACTIONS = {
    0: ("baseline_fifo", "FIFO"),
    1: ("baseline_earliest_due_date", "Earliest Due Date"),
    2: ("baseline_longest_waiting_time", "Longest Waiting Time"),
    3: ("baseline_highest_lateness_risk", "Highest Lateness Risk"),
}

REQUIRED_OUTPUT_FIELDS = [
    "policy_id",
    "policy_group",
    "agent_type",
    "policy_label",
    "replication",
    "eval_seed",
    "random_seed",
    "action_random_seed",
    "total_reward",
    "late_order_fraction",
    "n_orders_created",
    "n_orders_completed",
    "n_orders_in_date",
    "n_orders_late",
    "time_in_system_mean",
    "wip_mean",
    "n_eval_failures",
    "n_decision_points",
    "action_0_count",
    "action_1_count",
    "action_2_count",
    "action_3_count",
    "risk_t1",
    "risk_window",
    "risk_t2",
    "q_table_size",
    "error",
]

EXTRA_OUTPUT_FIELDS = [
    "fixed_action",
    "baseline_rule",
    "q_table_path",
    "source_config_path",
    "candidate_dir",
    "candidate_rank",
    "source_trial_index",
    "alpha",
    "gamma",
    "epsilon",
    "epsilon_decay",
    "epsilon_min",
    "target_final_epsilon",
    "msg",
    "run_duration",
    "rate_multiplier",
    "due_date_lower_bound",
    "due_date_upper_bound",
    "time_in_system_std",
    "time_in_system_min",
    "time_in_system_max",
    "wip_max",
    "training",
    "preparation_capacity",
    "sorting_capacity",
    "analysis1_capacity",
    "analysis2_capacity",
    "evaluation_capacity",
    "dispatching_capacity",
    "worker_capacity",
]

OUTPUT_FIELDS = list(dict.fromkeys(REQUIRED_OUTPUT_FIELDS + EXTRA_OUTPUT_FIELDS + SIM_RESULT_FIELDS))


@dataclass(frozen=True)
class PolicySpec:
    policy_id: str
    policy_group: str
    policy_label: str
    agent_type: str
    simulation_kwargs: dict[str, Any]
    fixed_action: int | None = None
    q_table_path: Path | None = None
    source_config_path: Path | None = None
    candidate_dir: Path | None = None
    candidate_rank: int | None = None
    source_trial_index: int | None = None
    q_parameters: dict[str, Any] | None = None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any], overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def value_as_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    result = Path(path)
    if result.is_absolute():
        return result
    return BASE_DIR / result


def load_standard_simulation_kwargs(run_dir: Path) -> dict[str, Any]:
    config_path = require_file(run_dir / "rl_config.json")
    config = read_json(config_path)
    risk = config.get("RISK_CONFIG", {})
    station_capacities = config.get("STATION_CAPACITIES", {})
    return {
        "run_duration": config["RUN_DURATION"],
        "rate_multiplier": config["RATE_MULTIPLIER"],
        "risk_t1": risk.get("risk_t1", config.get("FIXED_RISK_T1")),
        "risk_window": risk.get("risk_window", config.get("FIXED_RISK_WINDOW")),
        **station_capacities,
    }


def load_candidate_simulation_kwargs(candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config_path = require_file(candidate_dir / "candidate_config.json")
    config = read_json(config_path)
    parameters = config.get("parameters", {})
    station_capacities = config.get("station_capacities", {})
    return (
        {
            "run_duration": config["run_duration"],
            "rate_multiplier": config["rate_multiplier"],
            "risk_t1": parameters["risk_t1"],
            "risk_window": parameters["risk_window"],
            **station_capacities,
        },
        parameters,
        config_path,
    )


def resolve_best_candidate_dir(tuning_dir: Path) -> tuple[Path, dict[str, Any]]:
    stage2_dir = tuning_dir / "stage2"
    best_path = require_file(stage2_dir / "stage2_best_candidate.json")
    best_payload = read_json(best_path)
    best = best_payload["best_candidate"]
    rank = int(best["candidate_rank"])
    trial = int(best["source_trial_index"])
    candidate_dir = stage2_dir / f"candidate_rank_{rank:02d}_trial_{trial:03d}"
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Could not resolve local best candidate directory: {candidate_dir}")
    return candidate_dir, best


def make_policy_registry() -> list[PolicySpec]:
    fixed120_dir = STANDARD_RESULTS_DIR / "rl_results_final_10000_hpc_120"
    fixed120_kwargs = load_standard_simulation_kwargs(fixed120_dir)
    fixed120_config_path = fixed120_dir / "rl_config.json"
    fixed758_dir = STANDARD_RESULTS_DIR / "rl_results_final_10000_hpc_758"
    fixed758_kwargs = load_standard_simulation_kwargs(fixed758_dir)
    fixed758_config_path = fixed758_dir / "rl_config.json"

    policies: list[PolicySpec] = []
    for action, (policy_id, label) in BASELINE_ACTIONS.items():
        policies.append(
            PolicySpec(
                policy_id=policy_id,
                policy_group="baseline_fixed758",
                policy_label=label,
                agent_type="baseline",
                fixed_action=action,
                simulation_kwargs=dict(fixed758_kwargs),
                source_config_path=fixed758_config_path,
            )
        )

    policies.append(
        PolicySpec(
            policy_id="random_agent",
            policy_group="random_fixed758",
            policy_label="Random agent",
            agent_type="random",
            simulation_kwargs=dict(fixed758_kwargs),
            source_config_path=fixed758_config_path,
        )
    )

    manual_runs = [
        (
            "manual_standard",
            "manual_q_learning",
            "Manual Q-learning standard",
            STANDARD_RESULTS_DIR / "rl_results_final_10000_hpc_standard",
        ),
        (
            "manual_fixed120",
            "manual_q_learning",
            "Manual Q-learning fixed120",
            fixed120_dir,
        ),
        (
            "manual_fixed758",
            "manual_q_learning",
            "Manual Q-learning fixed758",
            fixed758_dir,
        ),
    ]
    for policy_id, group, label, run_dir in manual_runs:
        policies.append(
            PolicySpec(
                policy_id=policy_id,
                policy_group=group,
                policy_label=label,
                agent_type="q_learning",
                simulation_kwargs=load_standard_simulation_kwargs(run_dir),
                q_table_path=require_file(run_dir / "q_table_final.pkl"),
                source_config_path=require_file(run_dir / "rl_config.json"),
            )
        )

    tuned_runs = [
        ("bo_tuned_joint100", "BO-tuned Q-learning joint100", TUNING_RESULTS_DIR / "rl_tuning_hpc_100"),
        ("bo_tuned_fixed120", "BO-tuned Q-learning fixed120", TUNING_RESULTS_DIR / "rl_tuning_hpc_fixed120"),
        ("bo_tuned_fixed758", "BO-tuned Q-learning fixed758", TUNING_RESULTS_DIR / "rl_tuning_hpc_fixed758"),
        ("bo_tuned_gamma_high", "BO-tuned Q-learning gamma_high", TUNING_RESULTS_DIR / "rl_tuning_hpc_gamma_high"),
        ("bo_tuned_gamma_low", "BO-tuned Q-learning gamma_low", TUNING_RESULTS_DIR / "rl_tuning_hpc_gamma_low"),
    ]
    for policy_id, label, tuning_dir in tuned_runs:
        candidate_dir, best = resolve_best_candidate_dir(tuning_dir)
        kwargs, parameters, config_path = load_candidate_simulation_kwargs(candidate_dir)
        policies.append(
            PolicySpec(
                policy_id=policy_id,
                policy_group="bo_tuned_q_learning",
                policy_label=label,
                agent_type="q_learning",
                simulation_kwargs=kwargs,
                q_table_path=require_file(candidate_dir / "q_table.pkl"),
                source_config_path=config_path,
                candidate_dir=candidate_dir,
                candidate_rank=int(best["candidate_rank"]),
                source_trial_index=int(best["source_trial_index"]),
                q_parameters=parameters,
            )
        )

    policies.append(
        PolicySpec(
            policy_id="baseline_highest_lateness_risk_fixed120",
            policy_group="baseline_fixed120",
            policy_label="Highest Lateness Risk fixed120",
            agent_type="baseline",
            fixed_action=3,
            simulation_kwargs=dict(fixed120_kwargs),
            source_config_path=fixed120_config_path,
        )
    )

    return policies


def make_eval_seed_rows(
    n_replications: int,
    eval_seed_base: int,
    seed_step: int,
    action_seed_base: int,
) -> list[dict[str, int]]:
    return [
        {
            "replication": rep,
            "eval_seed": eval_seed_base + rep * seed_step,
            "random_agent_action_seed": action_seed_base + rep * seed_step,
        }
        for rep in range(n_replications)
    ]


def policy_action_seed(policy_index: int, replication: int, action_seed_base: int, seed_step: int) -> int:
    return action_seed_base + policy_index * POLICY_ACTION_SEED_STRIDE + replication * seed_step


def write_seed_csv(path: Path, seed_rows: list[dict[str, int]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["replication", "eval_seed", "random_agent_action_seed"])
        writer.writeheader()
        writer.writerows(seed_rows)


def registry_row(policy: PolicySpec, policy_index: int) -> dict[str, Any]:
    q_table_size: int | None = None
    if policy.q_table_path is not None and policy.q_table_path.exists():
        q_table_size = len(QLearningAgent.load(policy.q_table_path).q_table)

    parameters = policy.q_parameters or {}
    return {
        "policy_index": policy_index,
        "policy_id": policy.policy_id,
        "policy_group": policy.policy_group,
        "agent_type": policy.agent_type,
        "policy_label": policy.policy_label,
        "fixed_action": policy.fixed_action,
        "q_table_path": policy.q_table_path,
        "source_config_path": policy.source_config_path,
        "candidate_dir": policy.candidate_dir,
        "candidate_rank": policy.candidate_rank,
        "source_trial_index": policy.source_trial_index,
        "q_table_size": q_table_size,
        "risk_t1": policy.simulation_kwargs["risk_t1"],
        "risk_window": policy.simulation_kwargs["risk_window"],
        "risk_t2": policy.simulation_kwargs["risk_t1"] + policy.simulation_kwargs["risk_window"],
        "run_duration": policy.simulation_kwargs["run_duration"],
        "rate_multiplier": policy.simulation_kwargs["rate_multiplier"],
        "alpha": parameters.get("alpha"),
        "gamma": parameters.get("gamma"),
        "epsilon_decay": parameters.get("epsilon_decay"),
        "epsilon_min": parameters.get("epsilon_min"),
        "target_final_epsilon": parameters.get("target_final_epsilon"),
    }


def write_policy_registry(output_dir: Path, policies: list[PolicySpec], overwrite: bool) -> None:
    rows = [registry_row(policy, index) for index, policy in enumerate(policies)]
    registry_csv = output_dir / "policy_registry.csv"
    registry_json = output_dir / "policy_registry.json"
    if overwrite or not registry_csv.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(registry_csv, index=False)
    write_json(registry_json, {"policies": rows}, overwrite=overwrite or not registry_json.exists())


def write_holdout_config(
    output_dir: Path,
    args: argparse.Namespace,
    policies: list[PolicySpec],
    selected_policy_ids: list[str],
    seed_rows: list[dict[str, int]],
    overwrite: bool,
) -> None:
    payload = {
        "experiment": "rl_holdout_100_seeds",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_replications": args.n_replications,
        "eval_seed_base": args.eval_seed_base,
        "seed_step": args.seed_step,
        "action_seed_base": args.action_seed_base,
        "policy_action_seed_stride": POLICY_ACTION_SEED_STRIDE,
        "eval_seeds": [row["eval_seed"] for row in seed_rows],
        "random_agent_action_seeds": [row["random_agent_action_seed"] for row in seed_rows],
        "selected_policy_ids": selected_policy_ids,
        "all_policy_ids": [policy.policy_id for policy in policies],
        "seed_note": (
            "Evaluation seeds are eval_seed_base + replication * seed_step. "
            "Random-agent action seeds are stored in holdout_eval_seeds.csv. "
            "Q-learning agents are re-seeded deterministically per policy and replication "
            "before evaluation to make unseen states and action ties reproducible."
        ),
    }
    write_json(output_dir / "holdout_config.json", payload, overwrite=overwrite or not (output_dir / "holdout_config.json").exists())


def append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_safe(value) for key, value in row.items()})


def load_q_agent(policy: PolicySpec, action_seed: int) -> QLearningAgent:
    if policy.q_table_path is None:
        raise ValueError(f"Policy {policy.policy_id} has no q_table_path.")
    agent = QLearningAgent.load(policy.q_table_path)
    agent.rng = random.Random(action_seed)
    return agent


def run_one_replication(
    policy: PolicySpec,
    policy_index: int,
    replication: int,
    eval_seed: int,
    action_seed_base: int,
    seed_step: int,
) -> dict[str, Any]:
    action_seed: int | None = None
    q_table_size: int | None = None

    if policy.agent_type == "baseline":
        agent = BaselineAgent(fixed_action=int(policy.fixed_action))
        agent_type = "baseline"
    elif policy.agent_type == "random":
        action_seed = action_seed_base + replication * seed_step
        agent = RandomAgent(random_seed=action_seed)
        agent_type = "random"
    elif policy.agent_type == "q_learning":
        action_seed = policy_action_seed(policy_index, replication, action_seed_base, seed_step)
        agent = load_q_agent(policy, action_seed)
        q_table_size = len(agent.q_table)
        agent_type = "q_learning"
    else:
        raise ValueError(f"Unsupported agent_type for {policy.policy_id}: {policy.agent_type}")

    row = run_safely(
        f"{policy.policy_id}_{replication}",
        lambda: simulate_rl(
            agent=agent,
            agent_type=agent_type,
            fixed_action=policy.fixed_action or 0,
            training=False,
            random_seed=eval_seed,
            **policy.simulation_kwargs,
        ),
    )
    reward = value_as_float(row.get("total_reward"))
    if reward is not None and (math.isnan(reward) or math.isinf(reward)):
        row["total_reward"] = ""

    parameters = policy.q_parameters or {}
    row.update(
        {
            "policy_id": policy.policy_id,
            "policy_group": policy.policy_group,
            "policy_label": policy.policy_label,
            "agent_type": policy.agent_type,
            "replication": replication,
            "eval_seed": eval_seed,
            "random_seed": eval_seed,
            "action_random_seed": action_seed,
            "fixed_action": policy.fixed_action,
            "baseline_rule": policy.policy_label if policy.agent_type == "baseline" else "",
            "q_table_path": policy.q_table_path,
            "source_config_path": policy.source_config_path,
            "candidate_dir": policy.candidate_dir,
            "candidate_rank": policy.candidate_rank,
            "source_trial_index": policy.source_trial_index,
            "q_table_size": q_table_size,
            "alpha": parameters.get("alpha"),
            "gamma": parameters.get("gamma"),
            "epsilon_decay": parameters.get("epsilon_decay"),
            "epsilon_min": parameters.get("epsilon_min"),
            "target_final_epsilon": parameters.get("target_final_epsilon"),
        }
    )
    row.setdefault("error", "")
    return row


def evaluate_policy(
    policy: PolicySpec,
    policy_index: int,
    seed_rows: list[dict[str, int]],
    output_dir: Path,
    overwrite: bool,
) -> Path:
    output_path = output_dir / "evaluations" / f"{policy.policy_id}.csv"
    if output_path.exists() and not overwrite:
        print(f"Skipping existing policy output without --overwrite: {output_path}")
        return output_path

    rows = [
        run_one_replication(
            policy=policy,
            policy_index=policy_index,
            replication=int(seed_row["replication"]),
            eval_seed=int(seed_row["eval_seed"]),
            action_seed_base=int(seed_rows[0]["random_agent_action_seed"]),
            seed_step=int(seed_rows[1]["eval_seed"] - seed_rows[0]["eval_seed"]) if len(seed_rows) > 1 else DEFAULT_SEED_STEP,
        )
        for seed_row in seed_rows
    ]
    append_rows(output_path, rows, OUTPUT_FIELDS, overwrite=True)
    print(f"Wrote {len(rows)} rows: {output_path}")
    return output_path


def combine_evaluations(
    output_dir: Path,
    expected_policy_ids: list[str] | None,
    overwrite: bool,
) -> Path | None:
    evaluations_dir = output_dir / "evaluations"
    if not evaluations_dir.exists():
        print(f"No evaluations directory found: {evaluations_dir}")
        return None

    if expected_policy_ids is None:
        paths = sorted(evaluations_dir.glob("*.csv"))
    else:
        paths = [evaluations_dir / f"{policy_id}.csv" for policy_id in expected_policy_ids]
        missing = [path for path in paths if not path.exists()]
        if missing:
            print("Not combining yet; missing policy CSVs:")
            for path in missing:
                print(f"  {path}")
            return None

    if not paths:
        print(f"No policy CSVs found in {evaluations_dir}")
        return None

    output_path = output_dir / "holdout_all_evaluations.csv"
    if output_path.exists() and not overwrite:
        print(f"Combined CSV already exists; use --overwrite to replace it: {output_path}")
        return output_path

    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote combined evaluations: {output_path} ({len(combined)} rows)")
    return output_path


def select_policies(
    policies: list[PolicySpec],
    policy_ids: list[str] | None,
    policy_index: int | None,
) -> list[tuple[int, PolicySpec]]:
    indexed = list(enumerate(policies))
    if policy_index is not None:
        if policy_index < 0 or policy_index >= len(policies):
            raise IndexError(f"--policy-index must be in 0..{len(policies) - 1}")
        indexed = [indexed[policy_index]]

    if policy_ids:
        requested = {policy_id for item in policy_ids for policy_id in item.split(",") if policy_id}
        known = {policy.policy_id for policy in policies}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown policy id(s): {', '.join(unknown)}")
        indexed = [(index, policy) for index, policy in indexed if policy.policy_id in requested]

    return indexed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-replications", type=int, default=DEFAULT_N_REPLICATIONS)
    parser.add_argument("--eval-seed-base", type=int, default=DEFAULT_EVAL_SEED_BASE)
    parser.add_argument("--seed-step", type=int, default=DEFAULT_SEED_STEP)
    parser.add_argument("--action-seed-base", type=int, default=DEFAULT_ACTION_SEED_BASE)
    parser.add_argument("--policy", action="append", help="Policy id to evaluate. Can be repeated or comma-separated.")
    parser.add_argument("--policy-index", type=int, default=None, help="0-based policy index for SLURM arrays.")
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--list-policies", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_replications < 1:
        raise ValueError("--n-replications must be positive.")

    output_dir = resolve_path(args.output_dir, DEFAULT_OUTPUT_DIR)
    policies = make_policy_registry()
    selected = select_policies(policies, args.policy, args.policy_index)
    selected_policy_ids = [policy.policy_id for _, policy in selected]

    if args.list_policies:
        for index, policy in enumerate(policies):
            print(f"{index:02d} {policy.policy_id}: {policy.policy_label}")
        return

    seed_rows = make_eval_seed_rows(
        n_replications=args.n_replications,
        eval_seed_base=args.eval_seed_base,
        seed_step=args.seed_step,
        action_seed_base=args.action_seed_base,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_seed_csv(output_dir / "holdout_eval_seeds.csv", seed_rows, overwrite=args.overwrite)
    write_policy_registry(output_dir, policies, overwrite=args.overwrite)
    write_holdout_config(output_dir, args, policies, selected_policy_ids, seed_rows, overwrite=args.overwrite)

    if args.combine_only:
        expected_ids = selected_policy_ids if args.policy or args.policy_index is not None else [policy.policy_id for policy in policies]
        combine_evaluations(output_dir, expected_policy_ids=expected_ids, overwrite=args.overwrite)
        return

    for policy_index, policy in selected:
        evaluate_policy(policy, policy_index, seed_rows, output_dir, overwrite=args.overwrite)

    expected_ids = selected_policy_ids if args.policy or args.policy_index is not None else [policy.policy_id for policy in policies]
    combine_evaluations(output_dir, expected_policy_ids=expected_ids, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
