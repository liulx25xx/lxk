"""
Checkpoint-and-Backtrack (C&B) Engine.

FIXED: Uses Venus API + OpenAI SDK (no litellm), includes 6.5s rate limiting,
MAX_CALLS budget cap, and incremental result saving.

Implements the C&B algorithm for error recovery in code agents:
1. Detect errors in agent trajectory (Oracle / Heuristic / LLM)
2. Backtrack to a checkpoint before the error
3. Inject type-specific scaffolding feedback
4. Resume agent execution

Usage:
    python cb_engine.py --model gpt-4o-mini --mode oracle --trajectories ../data/trajectories/gpt-4o-mini/
    python cb_engine.py --model gpt-4.1 --mode heuristic
    python cb_engine.py --model gpt-4.1 --mode llm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from run_agent import SWEBenchAgent, load_instance, AgentTrajectory, MODEL_CONFIG

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results"

# Venus API configuration
VENUS_PROXY_URL = os.environ.get("VENUS_PROXY_URL", "<REDACTED_URL>")
# Rate limiting: 6.5s per API call
RATE_LIMIT_DELAY_SECONDS = 6.5
# Budget cap: max LLM calls for error detection
MAX_CALLS_DEFAULT = 200

# Best strategy per failure type (will be updated from EXP-005 results)
# Default: first strategy for each type
BEST_STRATEGY = {
    "LOC": "LOC_C_test_guided",
    "EDIT": "EDIT_A_reread_file",
    "LOGIC": "LOGIC_A_test_analysis",
    "TEST": "TEST_B_test_first",
    "PLAN": "PLAN_A_step_back",
}


def is_checkpoint_worthy(step: dict) -> bool:
    """Determine if a step is a good checkpoint location."""
    action = step.get("action", "")
    args = step.get("action_args", "")
    obs = step.get("observation", "")

    return any([
        "find" in action or "search" in action,  # After localization
        "str_replace" in args or "edit" in action,  # Before/after edit
        "test" in args.lower() or "pytest" in args.lower(),  # After test
        "Error" in obs or "FAILED" in obs,  # After error signal
        "Traceback" in obs,  # After exception
    ])


def detect_error_oracle(trajectory: dict, instance: dict) -> Tuple[Optional[str], Optional[int]]:
    """
    Oracle error detection: uses gold patch to determine first error.

    Returns: (error_type, first_error_step) or (None, None)
    """
    patch = instance.get("patch", "")
    gold_files = set()
    for line in patch.split('\n'):
        if line.startswith('diff --git'):
            parts = line.split()
            if len(parts) >= 4:
                gold_files.add(parts[3].lstrip('b/'))

    steps = trajectory.get("steps", [])

    for i, step in enumerate(steps):
        obs = step.get("observation", "")
        args = step.get("action_args", "")
        action = step.get("action", "")

        # If agent edits a file NOT in gold_files → LOC error
        if action in ("str_replace", "edit", "bash"):
            if any(f in args for f in gold_files):
                continue  # Correct file
            if "str_replace" in args or "sed" in args:
                # Editing wrong file
                return "LOC", i

        # If str_replace fails → EDIT error
        if "not found" in obs.lower() or "no match" in obs.lower():
            if "str_replace" in args or "edit" in action:
                return "EDIT", i

        # If test fails AFTER an edit on correct file → LOGIC error
        if "FAILED" in obs or "AssertionError" in obs:
            if i > 0 and steps[i-1].get("action") in ("str_replace", "edit"):
                return "LOGIC", i - 1

    # If no specific error found but still failed → PLAN
    return "PLAN", 0


def detect_error_heuristic(trajectory: dict) -> Tuple[Optional[str], Optional[int]]:
    """
    Heuristic error detection: rule-based signals without gold patch.

    Returns: (error_type, first_error_step) or (None, None)
    """
    steps = trajectory.get("steps", [])
    signals = []

    for i, step in enumerate(steps):
        obs = step.get("observation", "")
        action = step.get("action", "")
        args = step.get("action_args", "")

        # Signal 1: Tool error (str_replace not found)
        if ("not found" in obs.lower() or "no match" in obs.lower()):
            if "str_replace" in args or "edit" in action:
                signals.append((i, "EDIT", 0.8))

        # Signal 2: Test failure after edit
        if ("FAILED" in obs or "AssertionError" in obs):
            if i > 0 and steps[i-1].get("action") in ("str_replace", "edit", "bash"):
                signals.append((i-1, "LOGIC", 0.7))

        # Signal 3: Repeated identical actions (stuck in loop)
        if i >= 2:
            if (step.get("action") == steps[i-1].get("action") ==
                steps[i-2].get("action")):
                actions_same = (step.get("action_args", "")[:50] ==
                               steps[i-1].get("action_args", "")[:50])
                if actions_same:
                    signals.append((i-2, "PLAN", 0.6))

        # Signal 4: Long search with no results
        if "search" in action and ("No results" in obs or "0 matches" in obs):
            signals.append((i, "LOC", 0.7))

        # Signal 5: SyntaxError after edit
        if "SyntaxError" in obs:
            signals.append((i, "EDIT", 0.9))

    if signals:
        # Return highest-confidence earliest signal
        signals.sort(key=lambda x: (-x[2], x[0]))
        return signals[0][1], signals[0][0]

    return None, None


def detect_error_llm(trajectory: dict, instance: dict, client: OpenAI,
                     judge_model: str = "gpt-4o-mini") -> Tuple[Optional[str], Optional[int]]:
    """
    LLM-based error detection via Venus proxy: ask an LLM to identify the first error.

    Returns: (error_type, first_error_step) or (None, None)
    """
    # Format trajectory for LLM
    steps = trajectory.get("steps", [])
    traj_text = ""
    for step in steps[-15:]:  # Last 15 steps
        traj_text += (
            f"Step {step.get('step', '?')}: "
            f"Action={step.get('action', '')} Args={step.get('action_args', '')[:100]} "
            f"Obs={step.get('observation', '')[:200]}\n"
        )

    prompt = f"""Analyze this code agent trajectory that FAILED to solve a bug.
Identify the FIRST step where the agent made a critical error.

Issue summary: {instance.get('problem_statement', '')[:500]}

Trajectory:
{traj_text}

What is the primary failure type?
1. LOC - Agent looked at / edited wrong file or function
2. EDIT - Agent found right location but edit command failed (text mismatch)
3. LOGIC - Edit applied but logic is wrong
4. TEST - Agent misunderstood the requirements
5. PLAN - Agent chose wrong overall approach

Return JSON: {{"failure_type": "...", "first_error_step": N, "confidence": 0.0-1.0}}"""

    try:
        # Apply rate limiting BEFORE the call
        time.sleep(RATE_LIMIT_DELAY_SECONDS)
        
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        if '```' in content:
            content = content.split('```')[1].split('```')[0]
            if content.startswith('json'):
                content = content[4:]
        result = json.loads(content.strip())
        return result.get("failure_type"), result.get("first_error_step")
    except Exception as e:
        print(f"    LLM detection failed: {e}")
        return None, None


def find_checkpoint_before(steps: list, error_step: int) -> Optional[int]:
    """Find the most recent checkpoint before the error step."""
    best_cp = None
    for i, step in enumerate(steps):
        if i >= error_step:
            break
        if is_checkpoint_worthy(step):
            best_cp = i
    return best_cp


def run_cb(instance_id: str, model: str, mode: str,
           trajectory: dict, instance: dict, client: Optional[OpenAI] = None,
           max_backtracks: int = 3) -> dict:
    """
    Run the Checkpoint-and-Backtrack algorithm on a failed trajectory.

    Args:
        instance_id: SWE-bench instance ID
        model: Model to use for the agent
        mode: Error detection mode ("oracle", "heuristic", "llm")
        trajectory: The original failed trajectory
        instance: The SWE-bench instance data
        client: OpenAI client (required for llm mode)
        max_backtracks: Maximum number of backtrack attempts

    Returns:
        Result dict with resolve status, steps, cost, etc.
    """
    steps = trajectory.get("steps", [])

    # Detect error
    if mode == "oracle":
        error_type, error_step = detect_error_oracle(trajectory, instance)
    elif mode == "heuristic":
        error_type, error_step = detect_error_heuristic(trajectory)
    elif mode == "llm":
        if not client:
            raise ValueError("client required for llm mode")
        error_type, error_step = detect_error_llm(trajectory, instance, client)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if error_type is None:
        return {
            "instance_id": instance_id,
            "mode": mode,
            "resolved": False,
            "reason": "no_error_detected",
            "backtracks_used": 0,
        }

    # Find checkpoint
    checkpoint_step = find_checkpoint_before(steps, error_step)
    if checkpoint_step is None:
        checkpoint_step = max(0, error_step - 3)  # Fallback: go back 3 steps

    # Get scaffolding strategy for the detected type
    strategy = BEST_STRATEGY.get(error_type, "CONTROL_no_scaffold")

    # Run agent with scaffold from checkpoint
    agent = SWEBenchAgent(model=model, max_steps=30, scaffold=strategy)
    prev_traj = AgentTrajectory(instance_id, model)
    for step in steps:
        prev_traj.add_step(step)

    try:
        new_traj = agent.run_with_scaffold(instance, prev_traj, checkpoint_step)
        return {
            "instance_id": instance_id,
            "mode": mode,
            "error_type": error_type,
            "error_step": error_step,
            "checkpoint_step": checkpoint_step,
            "strategy": strategy,
            "resolved": new_traj.resolved,
            "new_steps": len(new_traj.steps),
            "cost": new_traj.total_cost,
            "backtracks_used": 1,  # Single backtrack for now
        }
    except Exception as e:
        return {
            "instance_id": instance_id,
            "mode": mode,
            "resolved": False,
            "error": str(e),
            "backtracks_used": 0,
        }


def run_cb_batch(model: str, mode: str, traj_dir: Path, output_dir: Path,
                 max_instances: Optional[int] = None, max_calls: int = MAX_CALLS_DEFAULT):
    """
    Run C&B on all failed trajectories.
    
    FIXED: Includes 6.5s rate limiting, MAX_CALLS budget cap, incremental saves.
    """
    # Load instances
    os.environ.setdefault('HF_HOME', '/home/xiankunlin/.cache/huggingface')
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test',
                      cache_dir='/home/xiankunlin/.cache/huggingface/datasets')
    instances = {item["instance_id"]: dict(item) for item in ds}

    # Initialize OpenAI client for LLM mode
    client = None
    if mode == "llm":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for llm mode")
        client = OpenAI(api_key=api_key, base_url=VENUS_PROXY_URL)

    # Find failed trajectories
    traj_files = list(traj_dir.glob("*.json"))
    failed_trajs = []
    for f in traj_files:
        with open(f) as fp:
            traj = json.load(fp)
        if traj.get("resolved") is False:
            failed_trajs.append(traj)

    if max_instances:
        failed_trajs = failed_trajs[:max_instances]

    print(f"Running C&B ({mode}) on {len(failed_trajs)} failed trajectories")
    print(f"Max calls budget: {max_calls}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    calls_made = 0
    
    for i, traj in enumerate(failed_trajs):
        # Budget cap check
        if calls_made >= max_calls:
            print(f"\n[BUDGET CAP] Reached max_calls limit ({max_calls}). Stopping.")
            results.append({
                "instance_id": traj["instance_id"],
                "status": "skipped",
                "reason": f"Budget cap reached after {calls_made} calls"
            })
            continue
        
        instance_id = traj["instance_id"]
        instance = instances.get(instance_id)
        if not instance:
            continue

        print(f"  [{i+1}/{len(failed_trajs)}] (calls: {calls_made}/{max_calls}) {instance_id}...", end="")
        result = run_cb(instance_id, model, mode, traj, instance, client)
        results.append(result)
        
        if mode == "llm":
            calls_made += 1  # LLM mode makes 1 call per instance

        status = "RESOLVED" if result.get("resolved") else "failed"
        print(f" {status} (type={result.get('error_type', '?')})")

        # Save individual result incrementally
        result_path = output_dir / f"{instance_id}.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Incremental summary save every 10 instances
        if (i + 1) % 10 == 0:
            resolved = sum(1 for r in results if r.get("resolved"))
            total = len(results)
            print(f"    [Progress: {resolved}/{total} resolved so far]")

    # Summary
    resolved = sum(1 for r in results if r.get("resolved"))
    total = len(results)
    print(f"\nC&B-{mode} Results:")
    print(f"  Resolved: {resolved}/{total} = {resolved/total*100:.1f}%" if total else "  No results")
    print(f"  Total calls made: {calls_made}/{max_calls}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "model": model,
            "mode": mode,
            "total": total,
            "resolved": resolved,
            "resolve_rate": resolved / total if total else 0,
            "calls_made": calls_made,
            "max_calls_limit": max_calls,
            "results": results,
        }, f, indent=2)
    print(f"  Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Checkpoint-and-Backtrack")
    parser.add_argument("--model", "-m", type=str, required=True,
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--mode", type=str, required=True,
                        choices=["oracle", "heuristic", "llm"])
    parser.add_argument("--trajectories", "-t", type=str, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--max_calls", type=int, default=MAX_CALLS_DEFAULT,
                        help=f"Max LLM calls for error detection (default: {MAX_CALLS_DEFAULT})")

    args = parser.parse_args()

    traj_dir = Path(args.trajectories) if args.trajectories else DATA_DIR / "trajectories" / args.model
    output_dir = Path(args.output_dir) if args.output_dir else DATA_DIR / f"cb_{args.mode}" / args.model

    run_cb_batch(args.model, args.mode, traj_dir, output_dir, args.max_instances, args.max_calls)


if __name__ == "__main__":
    main()
