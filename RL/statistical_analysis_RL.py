"""Statistical analysis for RL baseline and tuning experiments.

Select the experiment preset below before running the script. The preset keeps
standard RL results and matching tuning results together, for example fixed120
with fixed120.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import pandas as pd
from scipy import stats


ALPHA = 0.05
METRIC = "total_reward"
TEST_ALTERNATIVE = "two-sided"

BASE_DIR = Path(__file__).resolve().parent

EXPERIMENT_PRESETS = {
    "fixed120": {
        "standard_dir": BASE_DIR
        / "results"
        / "rl_standard_results"
        / "rl_results_final_10000_hpc_120",
        "tuning_dir": BASE_DIR
        / "results"
        / "rl_tuning_results"
        / "rl_tuning_hpc_fixed120",
    },
    "fixed758": {
        "standard_dir": BASE_DIR
        / "results"
        / "rl_standard_results"
        / "rl_results_final_10000_hpc_758",
        "tuning_dir": BASE_DIR
        / "results"
        / "rl_tuning_results"
        / "rl_tuning_hpc_fixed758",
    },
    "joint100": {
        "standard_dir": BASE_DIR
        / "results"
        / "rl_standard_results"
        / "rl_results_final_10000_hpc_standard",
        "tuning_dir": BASE_DIR / "results" / "rl_tuning_results" / "rl_tuning_hpc_100",
    },
    "joint60": {
        "standard_dir": BASE_DIR
        / "results"
        / "rl_standard_results"
        / "rl_results_final_10000_hpc_standard",
        "tuning_dir": BASE_DIR / "results" / "rl_tuning_results" / "rl_tuning_hpc_60",
    },
    "gamma_high": {
        "standard_dir": BASE_DIR
        / "results"
        / "rl_standard_results"
        / "rl_results_final_10000_hpc_120",
        "tuning_dir": BASE_DIR
        / "results"
        / "rl_tuning_results"
        / "rl_tuning_hpc_gamma_high",
    },
    "gamma_low": {
        "standard_dir": BASE_DIR
        / "results"
        / "rl_standard_results"
        / "rl_results_final_10000_hpc_120",
        "tuning_dir": BASE_DIR
        / "results"
        / "rl_tuning_results"
        / "rl_tuning_hpc_gamma_low",
    },
}

# Change this value to analyse another matching result set.
SELECTED_EXPERIMENT = "gamma_low"

# Set to None to use stage2_best_candidate.json, or set a directory name such as
# "candidate_rank_01_trial_055" to analyse a specific tuned candidate.
TUNED_CANDIDATE_DIRNAME: str | None = None

OUTPUT_DIR = BASE_DIR / "statistical_analysis_results"


def standard_error(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / sqrt(len(values)))


def describe(values: pd.Series) -> dict[str, float | int]:
    values = values.dropna()
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "standard_error": standard_error(values),
        "median": float(values.median()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values in the original order."""

    m = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [float("nan")] * m
    running_max = 0.0

    for rank, (original_index, p_value) in enumerate(ordered, start=1):
        corrected = min((m - rank + 1) * p_value, 1.0)
        running_max = max(running_max, corrected)
        adjusted[original_index] = running_max

    return adjusted


def paired_wilcoxon(
    first: pd.Series,
    second: pd.Series,
    first_name: str,
    second_name: str,
    comparison: str,
    metric: str = METRIC,
) -> dict[str, float | int | str | bool]:
    first = first.astype(float)
    second = second.astype(float)
    differences = first - second

    try:
        test = stats.wilcoxon(first, second, alternative=TEST_ALTERNATIVE)
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    except ValueError:
        statistic = 0.0
        p_value = 1.0

    row = {
        "comparison": comparison,
        "metric": metric,
        "first_group": first_name,
        "second_group": second_name,
        "n_pairs": int(len(differences)),
        "mean_first": float(first.mean()),
        "mean_second": float(second.mean()),
        "standard_error_first": standard_error(first),
        "standard_error_second": standard_error(second),
        "median_first": float(first.median()),
        "median_second": float(second.median()),
        "q1_first": float(first.quantile(0.25)),
        "q1_second": float(second.quantile(0.25)),
        "q3_first": float(first.quantile(0.75)),
        "q3_second": float(second.quantile(0.75)),
        "iqr_first": float(first.quantile(0.75) - first.quantile(0.25)),
        "iqr_second": float(second.quantile(0.75) - second.quantile(0.25)),
        "mean_difference_first_minus_second": float(differences.mean()),
        "standard_error_difference": standard_error(differences),
        "median_difference_first_minus_second": float(differences.median()),
        "test": "Wilcoxon signed-rank",
        "alternative": TEST_ALTERNATIVE,
        "test_statistic": statistic,
        "p_value": p_value,
        "alpha": ALPHA,
        "significant": bool(p_value < ALPHA),
    }
    return row


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def read_standard_results(standard_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q_learning = pd.read_csv(require_file(standard_dir / "q_learning_evaluation.csv"))
    baselines = pd.read_csv(require_file(standard_dir / "baselines.csv"))
    random_agent = pd.read_csv(require_file(standard_dir / "random_agent.csv"))
    return q_learning, baselines, random_agent


def find_best_candidate_dir(tuning_dir: Path) -> Path:
    stage2_dir = tuning_dir / "stage2"
    if TUNED_CANDIDATE_DIRNAME is not None:
        candidate_dir = stage2_dir / TUNED_CANDIDATE_DIRNAME
        if not candidate_dir.exists():
            raise FileNotFoundError(f"Configured candidate does not exist: {candidate_dir}")
        return candidate_dir

    best_path = require_file(stage2_dir / "stage2_best_candidate.json")
    with best_path.open("r", encoding="utf-8") as handle:
        best = json.load(handle)["best_candidate"]

    rank = int(best["candidate_rank"])
    trial = int(best["source_trial_index"])
    candidate_dir = stage2_dir / f"candidate_rank_{rank:02d}_trial_{trial:03d}"
    if candidate_dir.exists():
        return candidate_dir

    candidates = pd.read_csv(require_file(stage2_dir / "stage2_candidates.csv"))
    best_row = candidates.sort_values("objective_mean", ascending=True).iloc[0]
    rank = int(best_row["candidate_rank"])
    trial = int(best_row["source_trial_index"])
    candidate_dir = stage2_dir / f"candidate_rank_{rank:02d}_trial_{trial:03d}"
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Could not resolve best candidate directory: {candidate_dir}")
    return candidate_dir


def add_descriptive_row(
    rows: list[dict[str, float | int | str]],
    experiment: str,
    group: str,
    values: pd.Series,
) -> None:
    row = {
        "experiment": experiment,
        "group": group,
        "metric": METRIC,
    }
    row.update(describe(values))
    rows.append(row)


def analyse_h3(
    experiment: str,
    q_learning: pd.DataFrame,
    baselines: pd.DataFrame,
    random_agent: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_rows = []
    descriptive_rows = []

    add_descriptive_row(descriptive_rows, experiment, "Q-learning", q_learning[METRIC])
    add_descriptive_row(descriptive_rows, experiment, "Random agent", random_agent[METRIC])
    for baseline_name, baseline_frame in baselines.groupby("baseline_rule"):
        add_descriptive_row(
            descriptive_rows,
            experiment,
            f"Baseline: {baseline_name}",
            baseline_frame[METRIC],
        )

    comparison_frames = [("Random agent", random_agent)]
    comparison_frames.extend(
        (f"Baseline: {name}", frame)
        for name, frame in baselines.groupby("baseline_rule")
    )

    for comparison_name, comparison_frame in comparison_frames:
        paired = q_learning[["eval_seed", METRIC]].merge(
            comparison_frame[["eval_seed", METRIC]],
            on="eval_seed",
            suffixes=("_q_learning", "_comparison"),
        )
        if paired.empty:
            raise ValueError(f"No shared eval_seed values for {comparison_name}.")

        test_rows.append(
            paired_wilcoxon(
                paired[f"{METRIC}_q_learning"],
                paired[f"{METRIC}_comparison"],
                "Q-learning",
                comparison_name,
                f"H3: Q-learning vs {comparison_name}",
            )
        )

    adjusted = holm_bonferroni([float(row["p_value"]) for row in test_rows])
    for row, adjusted_p in zip(test_rows, adjusted):
        row["p_value_holm"] = adjusted_p
        row["significant_holm"] = bool(adjusted_p < ALPHA)

    return pd.DataFrame(test_rows), pd.DataFrame(descriptive_rows)


def analyse_h4(
    experiment: str,
    q_learning: pd.DataFrame,
    tuned_evaluation: pd.DataFrame,
    candidate_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = tuned_evaluation[["eval_seed", METRIC]].merge(
        q_learning[["eval_seed", METRIC]],
        on="eval_seed",
        suffixes=("_tuned", "_manual"),
    )
    if paired.empty:
        raise ValueError("No shared eval_seed values for tuned and manual Q-learning.")

    test_row = paired_wilcoxon(
        paired[f"{METRIC}_tuned"],
        paired[f"{METRIC}_manual"],
        "BO-tuned Q-learning",
        "Manual Q-learning",
        "H4: BO-tuned Q-learning vs manual Q-learning",
    )
    test_row["candidate_dir"] = str(candidate_dir)

    descriptive_rows = []
    add_descriptive_row(
        descriptive_rows,
        experiment,
        "BO-tuned Q-learning",
        tuned_evaluation[METRIC],
    )
    add_descriptive_row(
        descriptive_rows,
        experiment,
        "Manual Q-learning",
        q_learning[METRIC],
    )

    return pd.DataFrame([test_row]), pd.DataFrame(descriptive_rows)


def main() -> None:
    if SELECTED_EXPERIMENT not in EXPERIMENT_PRESETS:
        available = ", ".join(EXPERIMENT_PRESETS)
        raise ValueError(
            f"Unknown SELECTED_EXPERIMENT={SELECTED_EXPERIMENT!r}. "
            f"Available presets: {available}"
        )

    preset = EXPERIMENT_PRESETS[SELECTED_EXPERIMENT]
    standard_dir = preset["standard_dir"]
    tuning_dir = preset["tuning_dir"]

    q_learning, baselines, random_agent = read_standard_results(standard_dir)
    candidate_dir = find_best_candidate_dir(tuning_dir)
    tuned_evaluation = pd.read_csv(require_file(candidate_dir / "evaluation.csv"))

    h3_tests, h3_descriptives = analyse_h3(
        SELECTED_EXPERIMENT,
        q_learning,
        baselines,
        random_agent,
    )
    h4_test, h4_descriptives = analyse_h4(
        SELECTED_EXPERIMENT,
        q_learning,
        tuned_evaluation,
        candidate_dir,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = SELECTED_EXPERIMENT
    h3_tests.to_csv(OUTPUT_DIR / f"{prefix}_h3_policy_comparison_tests.csv", index=False)
    h3_descriptives.to_csv(
        OUTPUT_DIR / f"{prefix}_h3_policy_descriptives.csv",
        index=False,
    )
    h4_test.to_csv(OUTPUT_DIR / f"{prefix}_h4_tuned_vs_manual_test.csv", index=False)
    h4_descriptives.to_csv(
        OUTPUT_DIR / f"{prefix}_h4_tuned_vs_manual_descriptives.csv",
        index=False,
    )

    print(f"\nSelected experiment: {SELECTED_EXPERIMENT}")
    print(f"Standard results: {standard_dir}")
    print(f"Tuned candidate: {candidate_dir}")
    print("\nH3: Q-learning vs baselines/random agent")
    print(h3_tests.to_string(index=False))
    print("\nH4: BO-tuned Q-learning vs manual Q-learning")
    print(h4_test.to_string(index=False))
    print(f"\nResults written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
