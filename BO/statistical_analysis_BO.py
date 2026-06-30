"""Statistical analysis for BO and Random Search multi-run experiments.

The script uses the already combined result files created by
combine_multi_run_results.py. Configure the input directories below if another
experiment set should be analysed.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import pandas as pd
from scipy import stats


ALPHA = 0.05

BASE_DIR = Path(__file__).resolve().parent

# Main BO/Random Search result sets.
DEFAULT_RESULTS_DIR = BASE_DIR / "multi_run_results"
CHANGED_WEIGHTS_RESULTS_DIR = BASE_DIR / "multi_run_results_changed_weights"

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


def mann_whitney_result(
    group_a: pd.Series,
    group_b: pd.Series,
    group_a_name: str,
    group_b_name: str,
    metric: str,
    comparison: str,
) -> dict[str, float | int | str | bool]:
    group_a = group_a.dropna()
    group_b = group_b.dropna()
    test = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

    result = {
        "comparison": comparison,
        "metric": metric,
        "group_a": group_a_name,
        "group_b": group_b_name,
        "n_a": int(len(group_a)),
        "n_b": int(len(group_b)),
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "standard_error_a": standard_error(group_a),
        "standard_error_b": standard_error(group_b),
        "median_a": float(group_a.median()),
        "median_b": float(group_b.median()),
        "q1_a": float(group_a.quantile(0.25)),
        "q1_b": float(group_b.quantile(0.25)),
        "q3_a": float(group_a.quantile(0.75)),
        "q3_b": float(group_b.quantile(0.75)),
        "iqr_a": float(group_a.quantile(0.75) - group_a.quantile(0.25)),
        "iqr_b": float(group_b.quantile(0.75) - group_b.quantile(0.25)),
        "mean_difference_a_minus_b": float(group_a.mean() - group_b.mean()),
        "median_difference_a_minus_b": float(group_a.median() - group_b.median()),
        "test": "Mann-Whitney U",
        "alternative": "two-sided",
        "test_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "alpha": ALPHA,
        "significant": bool(test.pvalue < ALPHA),
    }
    return result


def load_summary(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    df = pd.read_csv(path)
    df["method"] = df["method"].str.lower()
    return df


def load_best_parameters(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "best_parameters_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing best-parameters file: {path}")
    df = pd.read_csv(path)
    df["method"] = df["method"].str.lower()
    return df


def analyse_h1(results_dir: Path, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = load_summary(results_dir)
    bo = summary.loc[summary["method"] == "bo", "final_best_objective"]
    random_search = summary.loc[
        summary["method"].isin(["random_search", "random search"]),
        "final_best_objective",
    ]

    test_row = mann_whitney_result(
        bo,
        random_search,
        "BO",
        "Random Search",
        "final_best_objective",
        f"H1: BO vs Random Search ({label})",
    )

    descriptives = []
    for method, group in summary.groupby("method"):
        row = {
            "experiment": label,
            "method": method,
            "metric": "final_best_objective",
        }
        row.update(describe(group["final_best_objective"]))
        descriptives.append(row)

    return pd.DataFrame([test_row]), pd.DataFrame(descriptives)


def analyse_h2() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    default_best = load_best_parameters(DEFAULT_RESULTS_DIR)
    changed_best = load_best_parameters(CHANGED_WEIGHTS_RESULTS_DIR)

    default_bo = default_best[default_best["method"] == "bo"].copy()
    changed_bo = changed_best[changed_best["method"] == "bo"].copy()

    primary_test = mann_whitney_result(
        default_bo["total_capacity"],
        changed_bo["total_capacity"],
        "Default weights",
        "Changed weights",
        "total_capacity",
        "H2: best BO total capacity under different objective weights",
    )

    capacity_columns = [
        "total_capacity",
        "preparation_capacity",
        "sorting_capacity",
        "analysis1_capacity",
        "analysis2_capacity",
        "evaluation_capacity",
        "dispatching_capacity",
        "worker_capacity",
    ]

    descriptive_rows = []
    component_test_rows = []
    for column in capacity_columns:
        for label, frame in [
            ("Default weights", default_bo),
            ("Changed weights", changed_bo),
        ]:
            row = {
                "experiment": label,
                "metric": column,
            }
            row.update(describe(frame[column]))
            descriptive_rows.append(row)

        component_test_rows.append(
            mann_whitney_result(
                default_bo[column],
                changed_bo[column],
                "Default weights",
                "Changed weights",
                column,
                "Exploratory capacity component comparison",
            )
        )

    p_adjusted = holm_bonferroni(
        [float(row["p_value"]) for row in component_test_rows]
    )
    for row, adjusted in zip(component_test_rows, p_adjusted):
        row["p_value_holm"] = adjusted
        row["significant_holm"] = bool(adjusted < ALPHA)

    return (
        pd.DataFrame([primary_test]),
        pd.DataFrame(descriptive_rows),
        pd.DataFrame(component_test_rows),
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    h1_default, h1_default_desc = analyse_h1(DEFAULT_RESULTS_DIR, "default weights")
    h1_changed, h1_changed_desc = analyse_h1(
        CHANGED_WEIGHTS_RESULTS_DIR,
        "changed weights",
    )
    h1_tests = pd.concat([h1_default, h1_changed], ignore_index=True)
    h1_descriptives = pd.concat(
        [h1_default_desc, h1_changed_desc],
        ignore_index=True,
    )

    h2_primary, h2_descriptives, h2_components = analyse_h2()

    h1_tests.to_csv(OUTPUT_DIR / "h1_bo_vs_random_search_tests.csv", index=False)
    h1_descriptives.to_csv(
        OUTPUT_DIR / "h1_bo_vs_random_search_descriptives.csv",
        index=False,
    )
    h2_primary.to_csv(OUTPUT_DIR / "h2_total_capacity_primary_test.csv", index=False)
    h2_descriptives.to_csv(
        OUTPUT_DIR / "h2_capacity_descriptives.csv",
        index=False,
    )
    h2_components.to_csv(
        OUTPUT_DIR / "h2_capacity_component_tests_exploratory.csv",
        index=False,
    )

    print("\nH1: BO vs Random Search")
    print(h1_tests.to_string(index=False))
    print("\nH2: total capacity under different BO objective weights")
    print(h2_primary.to_string(index=False))
    print(f"\nResults written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
