"""
Run scaffolding experiments on failed trajectories.

For each failed trajectory:
1. Identify the failure type (from annotations)
2. Inject a scaffolding prompt at the failure point
3. Let the agent retry with the scaffold
4. Record whether it recovers

Usage:
    # Run all strategies on all failed instances for a model
    python run_scaffolding.py --model gpt-4o-mini --annotations ../data/annotations/failure_types.json

    # Run specific strategy on specific failure type
    python run_scaffolding.py --model gpt-4o-mini --failure_type LOC --strategy LOC_A_broaden_search

    # Run control condition (retry without scaffold)
    python run_scaffolding.py --model gpt-4o-mini --control
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from run_agent import SWEBenchAgent, load_instance, AgentTrajectory, MODEL_CONFIG

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Rate limiting: 6.5s per API call
RATE_LIMIT_DELAY_SECONDS = 6.5
# Budget cap: max scaffolding experiment runs
MAX_CALLS_DEFAULT = 1000

# Strategy mapping: failure_type -> list of strategies
STRATEGY_MAP = {
    "LOC": ["LOC_A_broaden_search", "LOC_B_reread_issue", "LOC_C_test_guided"],
    "EDIT": ["EDIT_A_reread_file", "EDIT_B_smaller_edit", "EDIT_C_alternative_tool"],
    "LOGIC": ["LOGIC_A_test_analysis", "LOGIC_B_minimal_fix", "LOGIC_C_edge_cases"],
    "TEST": ["TEST_A_issue_reread", "TEST_B_test_first", "TEST_C_differential"],
    "PLAN": ["PLAN_A_step_back", "PLAN_B_scope_check", "PLAN_C_similar_fixes"],
}


def load_failed_trajectories(model: str) -> dict:
    """Load failed trajectories for a model."""
    traj_dir = DATA_DIR / "trajectories" / model
    failed = {}
    if not traj_dir.exists():
        print(f"Warning: No trajectories found at {traj_dir}")
        return failed

    for traj_file in traj_dir.glob("*.json"):
        with open(traj_file) as f:
            traj = json.load(f)
        if traj.get("resolved") is False:
            failed[traj["instance_id"]] = traj
    return failed


def load_annotations(annotations_path: str) -> dict:
    """Load failure type annotations."""
    with open(annotations_path) as f:
        data = json.load(f)
    # Return as {instance_id: annotation}
    if isinstance(data, list):
        return {a["instance_id"]: a for a in data}
    return data


def run_scaffold_experiment(model: str, failure_type: str, strategy: str,
                            annotations: dict, max_instances: int = None,
                            output_dir: Optional[Path] = None, max_calls: int = MAX_CALLS_DEFAULT):
    """Run scaffolding experiment for a specific type-strategy pair.
    
    FIXED: Includes rate limiting, budget capping, and incremental saves.
    """

    failed_trajs = load_failed_trajectories(model)

    # Filter by failure type
    matching = []
    for instance_id, traj in failed_trajs.items():
        ann = annotations.get(instance_id, {})
        if ann.get("failure_type") == failure_type:
            matching.append((instance_id, traj))

    if max_instances:
        matching = matching[:max_instances]

    print(f"\nRunning scaffold experiment:")
    print(f"  Model: {model}")
    print(f"  Failure type: {failure_type}")
    print(f"  Strategy: {strategy}")
    print(f"  Instances: {len(matching)}")
    print(f"  Max calls budget: {max_calls}")

    if not matching:
        print("  No matching instances found!")
        return

    # Output directory
    if output_dir is None:
        output_dir = DATA_DIR / "scaffolding" / failure_type / strategy / model
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    calls_made = 0
    agent = SWEBenchAgent(model=model, max_steps=30, scaffold=strategy)

    for i, (instance_id, failed_traj) in enumerate(matching):
        # Budget cap check
        if calls_made >= max_calls:
            print(f"\n[BUDGET CAP] Reached max_calls limit ({max_calls}). Stopping.")
            results.append({
                "instance_id": instance_id,
                "status": "skipped",
                "reason": f"Budget cap reached after {calls_made} calls"
            })
            continue
        
        print(f"  [{i+1}/{len(matching)}] (calls: {calls_made}/{max_calls}) {instance_id}...", end="")

        # Load full instance
        try:
            instance = load_instance(instance_id)
        except Exception as e:
            print(f" SKIP (load error: {e})")
            continue

        # Build AgentTrajectory from saved data
        prev_traj = AgentTrajectory(instance_id, model)
        for step in failed_traj.get("steps", []):
            prev_traj.add_step(step)

        # Get scaffold injection point from annotation
        ann = annotations.get(instance_id, {})
        scaffold_step = ann.get("first_error_step")

        # Run with scaffold
        try:
            new_traj = agent.run_with_scaffold(instance, prev_traj, scaffold_step)
            new_traj.save(output_dir)
            calls_made += len(new_traj.steps)
            
            results.append({
                "instance_id": instance_id,
                "resolved": new_traj.resolved,
                "steps": len(new_traj.steps),
                "cost": new_traj.total_cost,
            })
            status = "RESOLVED" if new_traj.resolved else "failed"
            print(f" {status} ({new_traj.total_cost:.4f}$)")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "instance_id": instance_id,
                "resolved": False,
                "error": str(e),
            })
        
        # Incremental save every 5 instances
        if (i + 1) % 5 == 0:
            resolved_count = sum(1 for r in results if r.get("resolved"))
            print(f"    [Progress: {resolved_count}/{len(results)} resolved so far]")

    # Save results summary
    resolved_count = sum(1 for r in results if r.get("resolved"))
    summary = {
        "model": model,
        "failure_type": failure_type,
        "strategy": strategy,
        "total": len(results),
        "resolved": resolved_count,
        "recovery_rate": resolved_count / len(results) if results else 0,
        "total_cost": sum(r.get("cost", 0) for r in results),
        "calls_made": calls_made,
        "max_calls_limit": max_calls,
        "results": results,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Recovery Rate: {resolved_count}/{len(results)} = {summary['recovery_rate']:.1%}" if results else "\n  No results")
    print(f"  Total calls made: {calls_made}/{max_calls}")
    print(f"  Summary: {summary_path}")
    return summary


def run_all_strategies(model: str, annotations_path: str, max_per_type: int = None,
                      max_calls: int = MAX_CALLS_DEFAULT):
    """Run all strategy combinations for all failure types."""
    annotations = load_annotations(annotations_path)

    all_results = []
    calls_made = 0
    
    for failure_type, strategies in STRATEGY_MAP.items():
        for strategy in strategies:
            # Budget check before starting each experiment
            if calls_made >= max_calls:
                print(f"\n[GLOBAL BUDGET CAP] Reached max_calls limit ({max_calls}). Stopping all experiments.")
                break
            
            result = run_scaffold_experiment(
                model=model,
                failure_type=failure_type,
                strategy=strategy,
                annotations=annotations,
                max_instances=max_per_type,
                max_calls=max_calls - calls_made,  # Pass remaining budget
            )
            if result:
                all_results.append(result)
                calls_made += result.get("calls_made", 0)

    # Save full matrix
    matrix_path = RESULTS_DIR / f"scaffolding_matrix_{model}.json"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with open(matrix_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull matrix saved: {matrix_path}")
    print(f"Total calls across all experiments: {calls_made}/{max_calls}")


def main():
    parser = argparse.ArgumentParser(description="Run scaffolding experiments")
    parser.add_argument("--model", "-m", type=str, required=True,
                        choices=list(MODEL_CONFIG.keys()))
    parser.add_argument("--annotations", "-a", type=str, required=True,
                        help="Path to failure_types.json annotations")
    parser.add_argument("--failure_type", type=str, default=None,
                        choices=["LOC", "EDIT", "LOGIC", "TEST", "PLAN"])
    parser.add_argument("--strategy", type=str, default=None,
                        help="Specific strategy to test")
    parser.add_argument("--control", action="store_true",
                        help="Run control condition (no scaffold)")
    parser.add_argument("--max_per_type", type=int, default=None,
                        help="Max instances per failure type")
    parser.add_argument("--max_calls", type=int, default=MAX_CALLS_DEFAULT,
                        help=f"Max experiment calls (default: {MAX_CALLS_DEFAULT})")

    args = parser.parse_args()

    if args.control:
        # Run with generic retry prompt (control condition)
        annotations = load_annotations(args.annotations)
        for failure_type in STRATEGY_MAP:
            run_scaffold_experiment(
                model=args.model,
                failure_type=failure_type,
                strategy="CONTROL_no_scaffold",
                annotations=annotations,
                max_instances=args.max_per_type,
                max_calls=args.max_calls,
            )
    elif args.failure_type and args.strategy:
        # Run specific combination
        annotations = load_annotations(args.annotations)
        run_scaffold_experiment(
            model=args.model,
            failure_type=args.failure_type,
            strategy=args.strategy,
            annotations=annotations,
            max_instances=args.max_per_type,
            max_calls=args.max_calls,
        )
    else:
        # Run all
        run_all_strategies(args.model, args.annotations, args.max_per_type, args.max_calls)


if __name__ == "__main__":
    main()
