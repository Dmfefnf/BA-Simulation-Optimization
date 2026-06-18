"""Run one Stage-1 RL Bayesian-optimization trial in its own process."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from RL_experiment import BASE_SEED, RATE_MULTIPLIER, RUN_DURATION, SEED_STEP, json_safe
from rl_tuning_common import (
    PARAMETER_NAMES,
    base_run_config,
    evaluate_q_agent,
    sanitize_parameters,
    train_q_agent,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-episodes", type=int, default=50)
    parser.add_argument("--eval-replications", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=SEED_STEP)
    parser.add_argument("--run-duration", type=float, default=RUN_DURATION)
    parser.add_argument("--rate-multiplier", type=float, default=RATE_MULTIPLIER)
    for name in PARAMETER_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, required=True)
    return parser.parse_args()


def parameters_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in PARAMETER_NAMES}


def run_trial(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = sanitize_parameters(parameters_from_args(args), args.training_episodes)

    config = {
        "stage": "stage1_bo_trial",
        "trial_index": args.trial_index,
        "training_episodes": args.training_episodes,
        "eval_replications": args.eval_replications,
        "parameters": parameters,
        **base_run_config(args),
    }
    write_json(output_dir / "trial_config.json", config)

    agent, training_summary = train_q_agent(
        parameters=parameters,
        output_dir=output_dir,
        training_episodes=args.training_episodes,
        base_seed=args.base_seed,
        seed_step=args.seed_step,
        run_duration=args.run_duration,
        rate_multiplier=args.rate_multiplier,
        training_csv_name=None,
    )
    evaluation_summary = evaluate_q_agent(
        agent=agent,
        parameters=parameters,
        output_dir=output_dir,
        eval_replications=args.eval_replications,
        base_seed=args.base_seed,
        seed_step=args.seed_step,
        run_duration=args.run_duration,
        rate_multiplier=args.rate_multiplier,
    )

    summary = {
        "stage": "stage1_bo_trial",
        "trial_index": args.trial_index,
        "parameters": parameters,
        **training_summary,
        **evaluation_summary,
    }
    write_json(output_dir / "training_summary.json", training_summary)
    write_json(output_dir / "trial_summary.json", summary)
    return json_safe(summary)


def main() -> None:
    summary = run_trial(parse_args())
    print(
        "Trial {trial_index} objective={objective_mean:.3f} "
        "reward_mean={total_reward_mean:.3f}".format(**summary),
        flush=True,
    )


if __name__ == "__main__":
    main()
