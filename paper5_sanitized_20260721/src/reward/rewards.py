"""
Domain-specific verifiable reward functions for RLVR training.

All rewards are rule-based (no neural reward model):
  - math:        Extract final number, exact match
  - mcq:         Extract answer letter (A/B/C/D/E), exact match
  - code:        Execute code in sandbox, pass/fail on test cases
  - format:      Penalize excessively long or malformed outputs

Each reward function returns a float in [0, 1].
"""
from __future__ import annotations

import logging
import math
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from fractions import Fraction
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Math reward
# ---------------------------------------------------------------------------

_MATH_ANSWER_PATTERNS = [
    r"####\s*(.+?)(?:\n|$)",                           # GSM8K format
    r"\\boxed\{(.+?)\}",                                # LaTeX boxed
    r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.|$)",
    r"[Aa]nswer\s*[:=]\s*(.+?)(?:\n|$)",
    r"=\s*(\-?[\d,\.]+)\s*$",                            # trailing equation
]


def _normalize_number(s: str) -> str | None:
    """Normalize a numerical string for comparison."""
    s = s.strip().rstrip(".")
    # Remove commas, dollar signs, percent signs
    s = s.replace(",", "").replace("$", "").replace("%", "")
    # Handle fractions
    if "/" in s:
        try:
            return str(float(Fraction(s)))
        except (ValueError, ZeroDivisionError):
            pass
    # Handle LaTeX fractions
    frac_match = re.search(r"\\frac\{(.+?)\}\{(.+?)\}", s)
    if frac_match:
        try:
            return str(float(Fraction(f"{frac_match.group(1)}/{frac_match.group(2)}")))
        except (ValueError, ZeroDivisionError):
            pass
    # Try direct float conversion
    try:
        return str(float(s))
    except ValueError:
        return s.strip().lower()


def extract_math_answer(text: str) -> str | None:
    """Extract the final numerical answer from model output."""
    for pattern in _MATH_ANSWER_PATTERNS:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return matches[-1].strip()
    # Fallback: last number in the text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
    return None


def math_reward(output: str, gold: str, **kwargs) -> float:
    """
    Math domain reward: exact match on final numerical answer.

    Returns 1.0 for correct, 0.0 for incorrect.
    """
    predicted = extract_math_answer(output)
    if predicted is None:
        return 0.0

    pred_norm = _normalize_number(predicted)
    gold_norm = _normalize_number(gold)

    if pred_norm is None or gold_norm is None:
        return 0.0

    # Exact string match after normalization
    if pred_norm == gold_norm:
        return 1.0

    # Approximate numerical match (tolerance 1e-4)
    try:
        pred_val = float(pred_norm)
        gold_val = float(gold_norm)
        if abs(pred_val - gold_val) < 1e-4:
            return 1.0
        # Relative tolerance for large numbers
        if gold_val != 0 and abs(pred_val - gold_val) / abs(gold_val) < 1e-4:
            return 1.0
    except (ValueError, TypeError):
        pass

    return 0.0


# ---------------------------------------------------------------------------
# MCQ reward
# ---------------------------------------------------------------------------

_MCQ_PATTERNS = [
    r"[Tt]he\s+(?:correct\s+)?answer\s+is\s*[:=]?\s*\(?([A-Ea-e])\)?",
    r"[Aa]nswer\s*[:=]\s*\(?([A-Ea-e])\)?",
    r"\b([A-Ea-e])\s*[\.\)]\s*$",            # letter at end of line
    r"^\s*\(?([A-Ea-e])\)?\s*$",              # standalone letter
    r"\\boxed\{([A-Ea-e])\}",                 # LaTeX boxed
]


def extract_mcq_answer(text: str) -> str | None:
    """Extract an MCQ answer letter from model output."""
    # Try patterns in priority order
    for pattern in _MCQ_PATTERNS:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            return matches[-1].upper()

    # Fallback: find the last standalone capital letter A-E
    letters = re.findall(r"\b([A-Ea-e])\b", text)
    if letters:
        return letters[-1].upper()

    return None


def mcq_reward(output: str, gold: str, **kwargs) -> float:
    """
    MCQ domain reward: exact match of answer letter.

    Returns 1.0 for correct, 0.0 for incorrect.
    """
    predicted = extract_mcq_answer(output)
    if predicted is None:
        return 0.0

    gold_clean = gold.strip().upper()

    # Handle both letter-only and "A. text" formats
    if len(gold_clean) > 1:
        gold_match = re.match(r"^([A-E])", gold_clean)
        if gold_match:
            gold_clean = gold_match.group(1)

    return 1.0 if predicted == gold_clean else 0.0


# ---------------------------------------------------------------------------
# Code reward
# ---------------------------------------------------------------------------

class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Code execution timed out")


def code_reward(
    output: str,
    gold: str,
    metadata: dict | None = None,
    timeout: int = 10,
    **kwargs,
) -> float:
    """
    Code domain reward: execute code and check test cases.

    Returns 1.0 if all test cases pass, 0.0 otherwise.
    Partial credit: fraction of passing tests.
    """
    metadata = metadata or {}

    # Extract code from output (handle markdown code blocks)
    code = _extract_code(output)
    if not code:
        return 0.0

    # Build test program
    test_code = _build_test_code(code, metadata)
    if not test_code:
        return 0.0

    # Execute in sandbox
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": "/usr/bin:/bin"},
        )
        if result.returncode == 0:
            return 1.0

        # Try to extract partial credit from output
        stdout = result.stdout.strip()
        if stdout.startswith("PARTIAL:"):
            try:
                return float(stdout.split("PARTIAL:")[1].strip())
            except ValueError:
                pass
        return 0.0

    except subprocess.TimeoutExpired:
        return 0.0
    except Exception as e:
        logger.debug(f"Code execution error: {e}")
        return 0.0


def _extract_code(text: str) -> str:
    """Extract Python code from model output."""
    # Try to extract from code blocks
    code_blocks = re.findall(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # Try to find a function definition
    func_match = re.search(r"(def\s+\w+.*?)(?=\n\S|\Z)", text, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()

    # Fallback: use entire text as code (for HumanEval prompts)
    return text.strip()


def _build_test_code(code: str, metadata: dict) -> str:
    """Build a test harness for the generated code."""
    parts = [code, "\n"]

    # HumanEval format
    if "test" in metadata and metadata["test"]:
        entry_point = metadata.get("entry_point", "")
        parts.append(metadata["test"])
        if entry_point:
            parts.append(f"\ncheck({entry_point})")

    # MBPP format
    elif "test_list" in metadata and metadata["test_list"]:
        test_list = metadata["test_list"]
        setup = metadata.get("test_setup_code", "")
        if setup:
            parts.append(setup + "\n")

        # Run each test, count passes
        parts.append(f"\n_total = {len(test_list)}")
        parts.append("\n_passed = 0")
        for test in test_list:
            parts.append(f"\ntry:\n    {test}\n    _passed += 1\nexcept:\n    pass")
        parts.append(f"\nif _passed == _total:\n    pass")
        parts.append(f"\nelse:\n    print(f'PARTIAL:{{_passed / _total}}')")
        parts.append(f"\n    raise SystemExit(1)")

    else:
        return ""

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Binary / yes-no reward (for LegalBench)
# ---------------------------------------------------------------------------

def binary_reward(output: str, gold: str, **kwargs) -> float:
    """Reward for binary (yes/no) answers."""
    output_lower = output.strip().lower()
    gold_lower = gold.strip().lower()

    # Extract yes/no from output
    if "yes" in output_lower.split()[-5:]:
        predicted = "yes"
    elif "no" in output_lower.split()[-5:]:
        predicted = "no"
    else:
        # Fallback: look for the word anywhere
        yes_count = output_lower.count("yes")
        no_count = output_lower.count("no")
        if yes_count > no_count:
            predicted = "yes"
        elif no_count > yes_count:
            predicted = "no"
        else:
            return 0.0

    return 1.0 if predicted == gold_lower else 0.0


# ---------------------------------------------------------------------------
# Format reward (bonus/penalty for output format quality)
# ---------------------------------------------------------------------------

MAX_OUTPUT_LENGTH = 4096  # tokens (approximated by words * 1.3)


def format_reward(output: str, **kwargs) -> float:
    """
    Penalize excessively long or malformed outputs (DAPO-style).

    Returns a penalty in [-0.5, 0].
    """
    word_count = len(output.split())
    approx_tokens = int(word_count * 1.3)

    if approx_tokens > MAX_OUTPUT_LENGTH:
        # Overlong penalty scales with excess length
        excess_ratio = (approx_tokens - MAX_OUTPUT_LENGTH) / MAX_OUTPUT_LENGTH
        return -min(0.5, 0.1 * excess_ratio)

    # Check for degenerate outputs
    if word_count < 3:
        return -0.2

    # Check for repetition (same sentence repeated)
    sentences = output.split(".")
    if len(sentences) > 5:
        unique = len(set(s.strip().lower() for s in sentences if s.strip()))
        if unique / len(sentences) < 0.3:
            return -0.3

    return 0.0


# ---------------------------------------------------------------------------
# Composite reward (domain reward + format penalty)
# ---------------------------------------------------------------------------

def composite_reward(
    output: str,
    gold: str,
    domain: str,
    metadata: dict | None = None,
    format_weight: float = 0.1,
    **kwargs,
) -> float:
    """
    Combined reward: domain_reward + format_weight * format_reward.

    Returns a float in [0, 1] (clamped).
    """
    reward_fn = get_reward_fn(domain)
    domain_r = reward_fn(output, gold, metadata=metadata or {})
    format_r = format_reward(output)
    total = domain_r + format_weight * format_r
    return max(0.0, min(1.0, total))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REWARD_FNS: dict[str, Callable] = {
    "math": math_reward,
    "science": mcq_reward,
    "law": binary_reward,      # many LegalBench tasks are binary
    "medicine": mcq_reward,
    "code": code_reward,
    "commonsense": mcq_reward,
}


def get_reward_fn(domain: str) -> Callable:
    """Get the reward function for a domain."""
    if domain not in REWARD_FNS:
        logger.warning(f"Unknown domain '{domain}', defaulting to mcq_reward")
        return mcq_reward
    return REWARD_FNS[domain]


def compute_reward(
    output: str,
    gold: str,
    domain: str,
    metadata: dict | None = None,
    use_format_reward: bool = True,
    format_weight: float = 0.1,
) -> float:
    """Unified reward computation entry point."""
    if use_format_reward:
        return composite_reward(
            output, gold, domain,
            metadata=metadata,
            format_weight=format_weight,
        )
    else:
        reward_fn = get_reward_fn(domain)
        return reward_fn(output, gold, metadata=metadata or {})


# ---------------------------------------------------------------------------
# Batch reward computation (for GRPO training integration)
# ---------------------------------------------------------------------------

def compute_rewards_batch(
    outputs: list[str],
    golds: list[str],
    domains: list[str],
    metadatas: list[dict] | None = None,
    use_format_reward: bool = True,
    format_weight: float = 0.1,
) -> list[float]:
    """Compute rewards for a batch of outputs."""
    if metadatas is None:
        metadatas = [{}] * len(outputs)

    rewards = []
    for output, gold, domain, meta in zip(outputs, golds, domains, metadatas):
        r = compute_reward(
            output, gold, domain,
            metadata=meta,
            use_format_reward=use_format_reward,
            format_weight=format_weight,
        )
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _test():
    """Quick self-test of reward functions."""
    # Math
    assert math_reward("The answer is 42.", "42") == 1.0
    assert math_reward("#### 3.14", "3.14") == 1.0
    assert math_reward("\\boxed{7}", "7") == 1.0
    assert math_reward("I think 5", "10") == 0.0
    assert math_reward("The answer is 1,234", "1234") == 1.0

    # MCQ
    assert mcq_reward("The answer is B.", "B") == 1.0
    assert mcq_reward("After analysis, I choose (C)", "C") == 1.0
    assert mcq_reward("A", "A") == 1.0
    assert mcq_reward("The answer is A.", "B") == 0.0

    # Binary
    assert binary_reward("Yes, this is correct.", "Yes") == 1.0
    assert binary_reward("No, I disagree.", "No") == 1.0

    # Format
    assert format_reward("Good answer here.") == 0.0
    assert format_reward("x") < 0  # too short
    assert format_reward(" ".join(["word"] * 5000)) < 0  # too long

    print("All reward tests passed!")


if __name__ == "__main__":
    _test()
