"""
Evaluation metrics: accuracy, pass@k, confidence intervals, OOD metrics.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def compute_accuracy(results: list[dict]) -> float:
    """Simple accuracy: fraction of correct results."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("correct", False))
    return correct / len(results)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Compute pass@k metric.

    n: total number of samples
    c: number of correct samples
    k: k for pass@k

    Uses the unbiased estimator from Chen et al. (2021) HumanEval paper:
      pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(range(n - c, n - c - k, -1)) / math.prod(range(n, n - k, -1))


def _wilson_ci(n_success: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n_total == 0:
        return (0.0, 0.0)

    p_hat = n_success / n_total
    denominator = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denominator
    margin = z * math.sqrt(
        (p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total
    ) / denominator

    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_confidence_interval(
    results: list[dict],
    confidence: float = 0.95,
    method: str = "wilson",
) -> dict:
    """
    Compute confidence interval for accuracy.

    Returns: {"accuracy": float, "ci_lower": float, "ci_upper": float, "n": int}
    """
    n = len(results)
    if n == 0:
        return {"accuracy": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    n_correct = sum(1 for r in results if r.get("correct", False))
    accuracy = n_correct / n

    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(confidence, 1.96)

    if method == "wilson":
        ci_lower, ci_upper = _wilson_ci(n_correct, n, z)
    else:
        # Normal approximation
        se = math.sqrt(accuracy * (1 - accuracy) / n)
        ci_lower = max(0.0, accuracy - z * se)
        ci_upper = min(1.0, accuracy + z * se)

    return {
        "accuracy": accuracy,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
    }


def compute_domain_metrics(
    results: list[dict],
    mode: str = "greedy",
    n_samples: int = 1,
) -> dict:
    """
    Compute comprehensive metrics for a set of evaluation results.

    Returns metrics organized by:
      - overall: aggregate metrics
      - by_subdomain: metrics per subdomain
      - by_difficulty: metrics per difficulty level
    """
    if not results:
        return {"overall": {"accuracy": 0.0, "count": 0}}

    # Overall metrics
    overall = compute_confidence_interval(results)

    if mode == "sample":
        # Compute pass@k for various k
        pass_at = {}
        for k in [1, 5, 10]:
            if k <= n_samples:
                scores = [
                    compute_pass_at_k(r["n_total"], r["n_correct"], k)
                    for r in results
                    if "n_total" in r
                ]
                pass_at[f"pass@{k}"] = sum(scores) / len(scores) if scores else 0.0
        overall.update(pass_at)

    if mode == "majority":
        # Average number of correct samples in the group
        avg_n_correct = sum(r.get("n_correct", 0) for r in results) / len(results)
        overall["avg_n_correct"] = avg_n_correct
        overall["avg_sample_acc"] = avg_n_correct / max(n_samples, 1)

    overall["count"] = len(results)

    # By subdomain
    by_subdomain = defaultdict(list)
    for r in results:
        by_subdomain[r.get("subdomain", "unknown")].append(r)

    subdomain_metrics = {}
    for sub, sub_results in by_subdomain.items():
        subdomain_metrics[sub] = compute_confidence_interval(sub_results)
        subdomain_metrics[sub]["count"] = len(sub_results)

    # By difficulty
    by_difficulty = defaultdict(list)
    for r in results:
        diff = r.get("difficulty")
        if diff:
            by_difficulty[diff].append(r)

    difficulty_metrics = {}
    for diff, diff_results in by_difficulty.items():
        difficulty_metrics[diff] = compute_confidence_interval(diff_results)
        difficulty_metrics[diff]["count"] = len(diff_results)

    return {
        "overall": overall,
        "by_subdomain": subdomain_metrics,
        "by_difficulty": difficulty_metrics,
    }


def aggregate_results(all_metrics: dict[str, dict]) -> dict:
    """Aggregate per-domain metrics into a summary."""
    summary = {
        "per_domain": {},
        "overall": {},
    }

    accuracies = []
    total_correct = 0
    total_count = 0

    for domain, metrics in all_metrics.items():
        overall = metrics.get("overall", {})
        acc = overall.get("accuracy", 0.0)
        count = overall.get("count", 0)

        summary["per_domain"][domain] = {
            "accuracy": acc,
            "ci_lower": overall.get("ci_lower", 0.0),
            "ci_upper": overall.get("ci_upper", 0.0),
            "count": count,
        }

        accuracies.append(acc)
        total_correct += int(acc * count)
        total_count += count

    if accuracies:
        summary["overall"]["macro_accuracy"] = sum(accuracies) / len(accuracies)
        summary["overall"]["micro_accuracy"] = total_correct / max(total_count, 1)
        summary["overall"]["total_count"] = total_count
        summary["overall"]["n_domains"] = len(accuracies)

    return summary


def compute_ood_metrics(
    in_domain_results: list[dict],
    out_domain_results: list[dict],
) -> dict:
    """Compute out-of-distribution generalization metrics."""
    id_acc = compute_accuracy(in_domain_results)
    ood_acc = compute_accuracy(out_domain_results)

    return {
        "in_domain_accuracy": id_acc,
        "ood_accuracy": ood_acc,
        "ood_gap": id_acc - ood_acc,
        "ood_retention": ood_acc / max(id_acc, 1e-8),
        "in_domain_count": len(in_domain_results),
        "ood_count": len(out_domain_results),
    }


def compute_rlvr_benefit(
    sft_accuracy: float,
    rlvr_accuracy: float,
    dpo_accuracy: float | None = None,
) -> dict:
    """
    Compute the RLVR benefit: how much does RL training add over SFT?

    This is the core metric for the "RLVR benefit frontier" paper.
    """
    rlvr_benefit = rlvr_accuracy - sft_accuracy
    rlvr_relative = rlvr_benefit / max(sft_accuracy, 1e-8)

    result = {
        "sft_accuracy": sft_accuracy,
        "rlvr_accuracy": rlvr_accuracy,
        "rlvr_benefit_absolute": rlvr_benefit,
        "rlvr_benefit_relative": rlvr_relative,
        "rlvr_wins": rlvr_accuracy > sft_accuracy,
    }

    if dpo_accuracy is not None:
        result["dpo_accuracy"] = dpo_accuracy
        result["dpo_benefit_absolute"] = dpo_accuracy - sft_accuracy
        result["rlvr_vs_dpo"] = rlvr_accuracy - dpo_accuracy

    return result
