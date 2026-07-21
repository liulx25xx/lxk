"""
Statistical tests and confidence intervals for experiment results.

Provides:
  - Bootstrap confidence intervals
  - McNemar's test for paired accuracy comparison
  - Permutation test for method comparison
  - Effect size (Cohen's d)
"""
from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence interval for the mean.

    Returns: {"mean": float, "ci_lower": float, "ci_upper": float, "std": float}
    """
    rng = np.random.RandomState(seed)
    values = np.array(values)
    n = len(values)

    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    # Bootstrap resampling
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "std": float(np.std(values)),
    }


def mcnemar_test(
    results_a: list[dict],
    results_b: list[dict],
) -> dict:
    """
    McNemar's test for comparing two classifiers on the same test set.

    Tests whether the two methods have significantly different error rates.

    Returns: {"statistic": float, "p_value": float, "significant_05": bool}
    """
    assert len(results_a) == len(results_b), "Results must be aligned (same test set)"

    # Count the 2x2 contingency table
    # b_correct=T, b_correct=F
    # a_correct=T:   n11         n12
    # a_correct=F:   n21         n22
    n12 = 0  # a correct, b incorrect
    n21 = 0  # a incorrect, b correct

    for ra, rb in zip(results_a, results_b):
        a_correct = ra.get("correct", False)
        b_correct = rb.get("correct", False)
        if a_correct and not b_correct:
            n12 += 1
        elif not a_correct and b_correct:
            n21 += 1

    # McNemar's statistic with continuity correction
    if n12 + n21 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant_05": False}

    statistic = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "n_a_only": n12,
        "n_b_only": n21,
    }


def permutation_test(
    scores_a: list[float],
    scores_b: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Two-sided permutation test for difference in means.

    Tests H0: mean(A) = mean(B) vs H1: mean(A) != mean(B).
    """
    rng = np.random.RandomState(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)

    observed_diff = np.mean(a) - np.mean(b)
    combined = np.concatenate([a, b])
    n_a = len(a)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n_a]) - np.mean(combined[n_a:])
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "significant_05": p_value < 0.05,
        "n_permutations": n_permutations,
    }


def cohen_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Compute Cohen's d effect size."""
    a = np.array(scores_a)
    b = np.array(scores_b)

    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0

    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    # Pooled standard deviation
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std < 1e-10:
        return 0.0

    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


# ---------------------------------------------------------------------------
# Full comparison pipeline
# ---------------------------------------------------------------------------

def compare_methods(
    results_dir: str,
    output_path: str,
    methods: tuple[str, str] = ("grpo", "sft"),
) -> dict:
    """
    Run full statistical comparison between two methods across all domains and sizes.
    """
    results_path = Path(results_dir)
    method_a, method_b = methods

    all_comparisons = {}

    domains = ["math", "science", "law", "medicine", "code", "commonsense"]
    sizes = [500, 2000, 5000, 20000]

    for domain in domains:
        all_comparisons[domain] = {}
        for size in sizes:
            file_a = results_path / method_a / domain / str(size) / f"{domain}_results.json"
            file_b = results_path / method_b / domain / str(size) / f"{domain}_results.json"

            if not file_a.exists() or not file_b.exists():
                continue

            with open(file_a) as f:
                data_a = json.load(f)
            with open(file_b) as f:
                data_b = json.load(f)

            results_a = data_a.get("results", [])
            results_b = data_b.get("results", [])

            if not results_a or not results_b:
                continue

            # Compute metrics
            acc_a = sum(1 for r in results_a if r.get("correct")) / len(results_a)
            acc_b = sum(1 for r in results_b if r.get("correct")) / len(results_b)

            scores_a = [float(r.get("correct", False)) for r in results_a]
            scores_b = [float(r.get("correct", False)) for r in results_b]

            # Statistical tests
            mcnemar = mcnemar_test(results_a, results_b)
            perm = permutation_test(scores_a, scores_b)
            d = cohen_d(scores_a, scores_b)

            comparison = {
                f"{method_a}_accuracy": acc_a,
                f"{method_b}_accuracy": acc_b,
                "difference": acc_a - acc_b,
                "mcnemar": mcnemar,
                "permutation_test": perm,
                "cohen_d": d,
                "effect_size": _interpret_effect_size(d),
            }

            all_comparisons[domain][size] = comparison

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_comparisons, f, indent=2)

    logger.info(f"Statistical comparisons saved to {output_path}")

    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info(f"Statistical Comparison: {method_a.upper()} vs {method_b.upper()}")
    logger.info(f"{'='*80}")
    for domain in domains:
        for size in sizes:
            comp = all_comparisons.get(domain, {}).get(size)
            if comp:
                sig = "*" if comp["mcnemar"]["significant_05"] else ""
                logger.info(
                    f"  {domain:>12} ({size:>6}): "
                    f"{method_a}={comp[f'{method_a}_accuracy']:.3f} vs "
                    f"{method_b}={comp[f'{method_b}_accuracy']:.3f} "
                    f"(diff={comp['difference']:+.3f}{sig}, d={comp['cohen_d']:.2f} [{comp['effect_size']}])"
                )

    return all_comparisons


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/statistics.json")
    parser.add_argument("--method_a", type=str, default="grpo")
    parser.add_argument("--method_b", type=str, default="sft")
    args = parser.parse_args()

    compare_methods(args.results_dir, args.output, (args.method_a, args.method_b))


if __name__ == "__main__":
    main()
