"""
Generate LaTeX tables for the paper.

Tables:
  1. Main results table (6 domains x 3 methods + base)
  2. Data scaling table
  3. Difficulty breakdown table
  4. OOD generalization table
  5. Statistical significance table
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ["math", "science", "law", "medicine", "code", "commonsense"]
DOMAIN_SHORT = {
    "math": "Math",
    "science": "Sci.",
    "law": "Law",
    "medicine": "Med.",
    "code": "Code",
    "commonsense": "C.Sense",
}
METHODS = ["base", "sft", "grpo", "dpo"]
METHOD_DISPLAY = {"base": "Base", "sft": "SFT", "grpo": "GRPO", "dpo": "DPO"}
SIZES = [500, 2000, 5000, 20000]


def _load_results(results_dir: str) -> dict:
    """Load all results into nested dict."""
    results = {}
    path = Path(results_dir)
    for method in METHODS:
        results[method] = {}
        for domain in DOMAINS:
            results[method][domain] = {}
            for size in SIZES:
                summary = path / method / domain / str(size) / "summary.json"
                if summary.exists():
                    with open(summary) as f:
                        results[method][domain][size] = json.load(f)
    return results


def _fmt(val: float | None, best: float | None = None) -> str:
    """Format accuracy value, bold if best."""
    if val is None:
        return "—"
    s = f"{val*100:.1f}"
    if best is not None and abs(val - best) < 1e-6:
        return f"\\textbf{{{s}}}"
    return s


def _get_acc(results: dict, method: str, domain: str, size: int = 5000) -> float | None:
    data = results.get(method, {}).get(domain, {}).get(size)
    if data is None:
        return None
    return data.get("metrics", {}).get("overall", {}).get("accuracy")


# ---------------------------------------------------------------------------
# Table 1: Main results (6 domains x 4 methods)
# ---------------------------------------------------------------------------

def generate_main_table(results: dict, output_path: str, size: int = 5000):
    """
    Main results table.
    Columns: Method | Math | Science | Law | Medicine | Code | Commonsense | Avg
    """
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results across 6 domains (accuracy \%, " +
                 f"{size//1000}K training instances). " +
                 r"Bold indicates best method per domain.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\begin{tabular}{l" + "c" * (len(DOMAINS) + 1) + "}")
    lines.append(r"\toprule")

    # Header
    header = "\\textbf{Method}"
    for d in DOMAINS:
        header += f" & \\textbf{{{DOMAIN_SHORT[d]}}}"
    header += r" & \textbf{Avg.} \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Find best per domain
    best_per_domain = {}
    for domain in DOMAINS:
        accs = {m: _get_acc(results, m, domain, size) for m in METHODS}
        valid = {m: a for m, a in accs.items() if a is not None}
        if valid:
            best_per_domain[domain] = max(valid.values())
        else:
            best_per_domain[domain] = None

    # Rows
    for method in METHODS:
        row = METHOD_DISPLAY[method]
        domain_accs = []
        for domain in DOMAINS:
            acc = _get_acc(results, method, domain, size)
            row += f" & {_fmt(acc, best_per_domain.get(domain))}"
            if acc is not None:
                domain_accs.append(acc)

        # Average
        if domain_accs:
            avg = sum(domain_accs) / len(domain_accs)
            row += f" & {avg*100:.1f}"
        else:
            row += " & —"
        row += r" \\"

        if method == "base":
            row += "\n" + r"\midrule"

        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Main table saved to {output_path}")
    return latex


# ---------------------------------------------------------------------------
# Table 2: Data scaling
# ---------------------------------------------------------------------------

def generate_scaling_table(results: dict, output_path: str, domain: str = "math"):
    """Data scaling table for a single domain."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{Data scaling results for {DOMAIN_SHORT[domain]} domain (accuracy \\%).}}")
    lines.append(r"\label{tab:scaling_" + domain + "}")
    lines.append(r"\begin{tabular}{l" + "c" * len(SIZES) + "}")
    lines.append(r"\toprule")

    # Header
    header = "\\textbf{Method}"
    for s in SIZES:
        header += f" & \\textbf{{{s//1000}K}}" if s >= 1000 else f" & \\textbf{{{s}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in ["sft", "grpo", "dpo"]:
        row = METHOD_DISPLAY[method]
        for size in SIZES:
            acc = _get_acc(results, method, domain, size)
            row += f" & {_fmt(acc)}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Scaling table saved to {output_path}")
    return latex


# ---------------------------------------------------------------------------
# Table 3: Full scaling across all domains
# ---------------------------------------------------------------------------

def generate_full_scaling_table(results: dict, output_path: str):
    """Full data scaling table: all domains x all sizes x all methods."""
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Complete data scaling results (accuracy \%). "
                 r"$\Delta$ shows GRPO improvement over SFT.}")
    lines.append(r"\label{tab:full_scaling}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    # Columns: Domain | Method | 500 | 2K | 5K | 20K
    lines.append(r"\begin{tabular}{ll" + "c" * len(SIZES) + "}")
    lines.append(r"\toprule")

    header = r"\textbf{Domain} & \textbf{Method}"
    for s in SIZES:
        label = f"{s//1000}K" if s >= 1000 else str(s)
        header += f" & \\textbf{{{label}}}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for i, domain in enumerate(DOMAINS):
        for j, method in enumerate(["sft", "grpo", "dpo"]):
            d_label = DOMAIN_SHORT[domain] if j == 0 else ""
            row = f"{d_label} & {METHOD_DISPLAY[method]}"
            for size in SIZES:
                acc = _get_acc(results, method, domain, size)
                row += f" & {_fmt(acc)}"
            row += r" \\"
            lines.append(row)

        # Add delta row
        row = f" & $\\Delta$ (GRPO$-$SFT)"
        for size in SIZES:
            g = _get_acc(results, "grpo", domain, size)
            s = _get_acc(results, "sft", domain, size)
            if g is not None and s is not None:
                diff = (g - s) * 100
                sign = "+" if diff >= 0 else ""
                row += f" & {sign}{diff:.1f}"
            else:
                row += " & —"
        row += r" \\"
        lines.append(row)

        if i < len(DOMAINS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")

    latex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Full scaling table saved to {output_path}")
    return latex


# ---------------------------------------------------------------------------
# Table 4: RLVR Benefit Summary
# ---------------------------------------------------------------------------

def generate_benefit_summary_table(results: dict, output_path: str, size: int = 5000):
    """
    Summary table of RLVR benefit per domain.
    Columns: Domain | SFT | GRPO | DPO | GRPO-SFT | DPO-SFT | GRPO-DPO | Sig?
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{RLVR benefit analysis (" + f"{size//1000}K" +
                 r" training instances). $\Delta$ in percentage points.}")
    lines.append(r"\label{tab:rlvr_benefit}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Domain} & \textbf{SFT} & \textbf{GRPO} & \textbf{DPO} "
                 r"& $\Delta_{\text{G-S}}$ & $\Delta_{\text{D-S}}$ & $\Delta_{\text{G-D}}$ \\")
    lines.append(r"\midrule")

    for domain in DOMAINS:
        sft = _get_acc(results, "sft", domain, size)
        grpo = _get_acc(results, "grpo", domain, size)
        dpo = _get_acc(results, "dpo", domain, size)

        row = DOMAIN_SHORT[domain]
        row += f" & {_fmt(sft)}"
        row += f" & {_fmt(grpo)}"
        row += f" & {_fmt(dpo)}"

        if grpo is not None and sft is not None:
            d_gs = (grpo - sft) * 100
            row += f" & {d_gs:+.1f}"
        else:
            row += " & —"

        if dpo is not None and sft is not None:
            d_ds = (dpo - sft) * 100
            row += f" & {d_ds:+.1f}"
        else:
            row += " & —"

        if grpo is not None and dpo is not None:
            d_gd = (grpo - dpo) * 100
            row += f" & {d_gd:+.1f}"
        else:
            row += " & —"

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex)
    logger.info(f"Benefit summary table saved to {output_path}")
    return latex


# ---------------------------------------------------------------------------
# Master
# ---------------------------------------------------------------------------

def generate_all_tables(results_dir: str, output_dir: str):
    """Generate all paper tables."""
    os.makedirs(output_dir, exist_ok=True)
    results = _load_results(results_dir)

    generate_main_table(results, os.path.join(output_dir, "tab_main_results.tex"))
    generate_full_scaling_table(results, os.path.join(output_dir, "tab_full_scaling.tex"))
    generate_benefit_summary_table(results, os.path.join(output_dir, "tab_rlvr_benefit.tex"))

    # Per-domain scaling tables
    for domain in DOMAINS:
        generate_scaling_table(
            results,
            os.path.join(output_dir, f"tab_scaling_{domain}.tex"),
            domain=domain,
        )

    logger.info(f"\nAll tables saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="tables")
    args = parser.parse_args()
    generate_all_tables(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
