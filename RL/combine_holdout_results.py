"""Combine per-policy RL holdout evaluation CSVs into one analysis file."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HOLDOUT_DIR = BASE_DIR / "results" / "rl_holdout_results" / "holdout_100_seeds"
DEFAULT_OUTPUT_NAME = "holdout_all_evaluations.csv"


def resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    result = Path(path)
    if result.is_absolute():
        return result
    return BASE_DIR / result


def policy_ids_from_registry(holdout_dir: Path) -> list[str] | None:
    registry_path = holdout_dir / "policy_registry.csv"
    if not registry_path.exists():
        return None

    registry = pd.read_csv(registry_path)
    if "policy_id" not in registry.columns:
        raise ValueError(f"policy_registry.csv has no policy_id column: {registry_path}")

    if "policy_index" in registry.columns:
        registry = registry.sort_values("policy_index")

    return registry["policy_id"].dropna().astype(str).tolist()


def discover_policy_csvs(holdout_dir: Path, use_registry_order: bool = True) -> list[Path]:
    evaluations_dir = holdout_dir / "evaluations"
    if not evaluations_dir.exists():
        raise FileNotFoundError(f"Missing evaluations directory: {evaluations_dir}")

    if use_registry_order:
        policy_ids = policy_ids_from_registry(holdout_dir)
        if policy_ids:
            paths = [evaluations_dir / f"{policy_id}.csv" for policy_id in policy_ids]
            missing = [path for path in paths if not path.exists()]
            if missing:
                missing_text = "\n".join(f"  {path}" for path in missing)
                raise FileNotFoundError(f"Missing expected policy CSVs:\n{missing_text}")
            registry_path_names = {path.name for path in paths}
            extra_paths = sorted(
                path for path in evaluations_dir.glob("*.csv")
                if path.name not in registry_path_names
            )
            return [*paths, *extra_paths]

    paths = sorted(evaluations_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No policy CSVs found in: {evaluations_dir}")
    return paths


def combine_policy_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "policy_id" not in frame.columns:
            frame.insert(0, "policy_id", path.stem)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    sort_columns = [column for column in ["policy_id", "replication", "eval_seed"] if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    return combined


def backup_existing_output(output_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_path.with_name(f"{output_path.stem}_backup_{timestamp}{output_path.suffix}")
    output_path.replace(backup_path)
    return backup_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backup-existing", action="store_true")
    parser.add_argument("--ignore-registry", action="store_true", help="Combine all CSVs alphabetically instead of policy_registry.csv order.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    holdout_dir = resolve_path(args.holdout_dir, DEFAULT_HOLDOUT_DIR)
    output_path = resolve_path(args.output, holdout_dir / DEFAULT_OUTPUT_NAME)

    paths = discover_policy_csvs(holdout_dir, use_registry_order=not args.ignore_registry)
    combined = combine_policy_csvs(paths)

    policy_counts = combined["policy_id"].value_counts().sort_index() if "policy_id" in combined.columns else pd.Series(dtype=int)
    print(f"Input directory: {holdout_dir}")
    print(f"Policy CSV files: {len(paths)}")
    print(f"Combined rows: {len(combined)}")
    if not policy_counts.empty:
        print("Rows per policy:")
        print(policy_counts.to_string())

    if args.dry_run:
        print("Dry run only; no file written.")
        return

    if output_path.exists():
        if args.backup_existing:
            backup_path = backup_existing_output(output_path)
            print(f"Backed up existing output to: {backup_path}")
        elif not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Use --overwrite to replace it or --backup-existing to keep a timestamped backup."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote combined holdout results: {output_path}")


if __name__ == "__main__":
    main()
