import argparse
import json
from pathlib import Path

import BO as bo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one independent Bayesian-optimization run."
    )
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--n-trials", type=int, default=bo.N_TRIALS)
    parser.add_argument("--n-replications", type=int, default=bo.N_REPLICATIONS)
    parser.add_argument("--base-seed", type=int, default=bo.BASE_SEED)
    parser.add_argument("--seed-step", type=int, default=bo.SEED_STEP)
    parser.add_argument("--bo-random-seed", type=int, default=bo.BO_RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-duration", type=float, default=None)
    parser.add_argument("--rate-multiplier", type=float, default=None)
    parser.add_argument("--objective-time-mode", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = bo.run_experiment(
        n_trials=args.n_trials,
        n_replications=args.n_replications,
        base_seed=args.base_seed,
        seed_step=args.seed_step,
        bo_random_seed=args.bo_random_seed,
        output_dir=args.output_dir,
        run_index=args.run_index,
        run_duration=args.run_duration,
        rate_multiplier=args.rate_multiplier,
        objective_time_mode=args.objective_time_mode,
        save_after_each_trial=False,
        return_records=False,
    )
    print(json.dumps(bo.json_safe(result["best_result"]), indent=2))


if __name__ == "__main__":
    main()
