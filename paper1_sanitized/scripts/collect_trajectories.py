"""
Collect baseline trajectories using mini-swe-agent's Docker backend.

This script uses mini-swe-agent's SWE-bench evaluation infrastructure
for actual code execution in Docker containers.

Usage:
    # Collect baseline trajectories for all 200 instances
    python collect_trajectories.py --model gpt-4o-mini --subset ../data/swebench_subset.json

    # Collect with specific model and output dir
    python collect_trajectories.py --model gpt-4.1 --output_dir ../data/trajectories/gpt-4.1

    # Dry run (no API calls, just setup check)
    python collect_trajectories.py --model gpt-4o-mini --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# mini-swe-agent model name mapping
MINI_SWE_MODEL_MAP = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4.1": "openai/gpt-4.1",
    "deepseek-v4": "deepseek/deepseek-chat",
    "claude-sonnet-4": "anthropic/claude-sonnet-4-20250514",
}


def check_docker():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_api_key(model: str) -> bool:
    """Check if the required API key is set."""
    key_map = {
        "gpt-4o-mini": "OPENAI_API_KEY",
        "gpt-4.1": "OPENAI_API_KEY",
        "gpt-5.5": "OPENAI_API_KEY",
        "deepseek-v4": "DEEPSEEK_API_KEY",
        "claude-sonnet-4": "ANTHROPIC_API_KEY",
    }
    key_name = key_map.get(model)
    if key_name:
        return bool(os.environ.get(key_name))
    return False


def run_mini_swe_agent(instance_id: str, model: str, output_path: Path,
                       cost_limit: float = 1.0, step_limit: int = 30) -> dict:
    """
    Run mini-swe-agent on a single instance.

    Uses the mini-swe-agent CLI with SWE-bench config.
    """
    litellm_model = MINI_SWE_MODEL_MAP.get(model, model)

    cmd = [
        "mini-swe-agent",
        "-m", litellm_model,
        "-c", "swebench.yaml",  # Use swebench config
        "-c", f"model.model_kwargs.temperature=0",
        "-c", f"agent.step_limit={step_limit}",
        "-c", f"agent.cost_limit={cost_limit}",
        "-y",  # Non-interactive (yolo mode)
        "-t", instance_id,  # Task = instance_id for SWE-bench
        "-o", str(output_path),
    ]

    print(f"  Running: {' '.join(cmd[:6])}...")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes per instance
            env={**os.environ, "LITELLM_MODEL": litellm_model},
        )

        if result.returncode == 0:
            # Parse output trajectory
            if output_path.exists():
                with open(output_path) as f:
                    return json.load(f)
            return {"status": "success", "output": result.stdout[-500:]}
        else:
            return {"status": "failed", "error": result.stderr[-500:]}

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": "Exceeded 10 minute timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def collect_batch(subset_path: str, model: str, output_dir: Path,
                  cost_limit: float = 1.0, step_limit: int = 30,
                  start_from: int = 0, dry_run: bool = False):
    """Collect trajectories for all instances in subset."""

    # Load subset
    with open(subset_path) as f:
        subset_data = json.load(f)
    instance_ids = [inst["instance_id"] for inst in subset_data["instances"]]

    print(f"=" * 60)
    print(f"Trajectory Collection")
    print(f"  Model: {model}")
    print(f"  Instances: {len(instance_ids)} (starting from #{start_from})")
    print(f"  Output: {output_dir}")
    print(f"  Cost limit per instance: ${cost_limit}")
    print(f"  Step limit: {step_limit}")
    print(f"  Docker available: {check_docker()}")
    print(f"  API key set: {check_api_key(model)}")
    print(f"=" * 60)

    if dry_run:
        print("\n[DRY RUN] Would process these instances:")
        for i, iid in enumerate(instance_ids[:10]):
            print(f"  {i+1}. {iid}")
        if len(instance_ids) > 10:
            print(f"  ... and {len(instance_ids)-10} more")
        return

    if not check_api_key(model):
        print(f"\nERROR: API key not set for {model}")
        print(f"Please set the appropriate environment variable.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "collection_log.jsonl"

    success_count = 0
    fail_count = 0
    total_cost = 0.0

    for i, instance_id in enumerate(instance_ids[start_from:], start=start_from):
        print(f"\n[{i+1}/{len(instance_ids)}] {instance_id}")

        # Skip if already collected
        traj_path = output_dir / f"{instance_id}.json"
        if traj_path.exists():
            print(f"  -> Already exists, skipping")
            success_count += 1
            continue

        # Run
        start_time = time.time()
        result = run_mini_swe_agent(
            instance_id=instance_id,
            model=model,
            output_path=traj_path,
            cost_limit=cost_limit,
            step_limit=step_limit,
        )
        elapsed = time.time() - start_time

        # Log
        log_entry = {
            "instance_id": instance_id,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": elapsed,
            "status": result.get("status", "unknown"),
        }

        if result.get("status") == "success":
            success_count += 1
            print(f"  -> Success ({elapsed:.1f}s)")
        else:
            fail_count += 1
            print(f"  -> {result.get('status', 'failed')}: {result.get('error', '')[:100]}")
            log_entry["error"] = result.get("error", "")

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Rate limiting
        time.sleep(1)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Collection Complete")
    print(f"  Success: {success_count}/{len(instance_ids)}")
    print(f"  Failed: {fail_count}/{len(instance_ids)}")
    print(f"  Log: {log_path}")
    print(f"{'=' * 60}")

    # Save summary
    summary = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "total": len(instance_ids),
        "success": success_count,
        "failed": fail_count,
    }
    summary_path = output_dir / "collection_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Collect SWE-bench trajectories")
    parser.add_argument("--model", "-m", type=str, required=True,
                        choices=list(MINI_SWE_MODEL_MAP.keys()))
    parser.add_argument("--subset", "-s", type=str,
                        default=str(DATA_DIR / "swebench_subset.json"))
    parser.add_argument("--output_dir", "-o", type=str, default=None)
    parser.add_argument("--cost_limit", type=float, default=1.0,
                        help="Max cost per instance ($)")
    parser.add_argument("--step_limit", type=int, default=30)
    parser.add_argument("--start_from", type=int, default=0,
                        help="Start from this index (for resumption)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Check setup without running")

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = DATA_DIR / "trajectories" / args.model

    collect_batch(
        subset_path=args.subset,
        model=args.model,
        output_dir=output_dir,
        cost_limit=args.cost_limit,
        step_limit=args.step_limit,
        start_from=args.start_from,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
