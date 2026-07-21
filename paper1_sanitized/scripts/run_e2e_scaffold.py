#!/usr/bin/env python3
"""
End-to-end SWE-bench evaluation with scaffold injection.

Uses swebench Docker env images for containers, runs a minimal bash agent via
Venus API (gpt-4o-mini), optionally injects scaffold after first edit error,
then evaluates patches via swebench harness (binary pass/fail).

Usage:
    # Run full pilot (5 instances × 2 modes)
    python scripts/run_e2e_scaffold.py --pilot

    # Run single instance
    python scripts/run_e2e_scaffold.py --instance_id django__django-11206 --mode control

    # Only evaluate existing predictions
    python scripts/run_e2e_scaffold.py --evaluate --run_id pilot_20260517_203500
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI

# === Configuration ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "e2e_scaffold"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

VENUS_API_KEY = "<REDACTED_SECRET>"
VENUS_BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4.1"
MAX_STEPS = 15
RATE_LIMIT_DELAY = 3.5  # seconds between API calls
MAX_API_CALLS = 1000  # total budget for gpt-4.1 pilot

# Our 5 pilot EDIT instances (all django)
PILOT_INSTANCES = [
    "django__django-11206",
    "django__django-13406",
    "django__django-14017",
    "django__django-11885",
    "django__django-16502",
]

# Global call counter
_api_calls_made = 0


# ============================================================
# Docker utilities
# ============================================================

def docker_run(cmd: str, timeout: int = 120) -> Tuple[int, str]:
    """Run a sudo docker command."""
    try:
        result = subprocess.run(
            ["sudo"] + cmd.split() if isinstance(cmd, str) else ["sudo"] + cmd,
            capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 1, "[TIMEOUT]"


def docker_exec(container: str, command: str, timeout: int = 60) -> Tuple[int, str]:
    """Execute a command in a running Docker container."""
    try:
        result = subprocess.run(
            ["sudo", "docker", "exec", container, "bash", "-c", command],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + result.stderr
        if len(output) > 8000:
            output = output[:4000] + "\n...[TRUNCATED]...\n" + output[-4000:]
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "[TIMEOUT: exceeded {timeout}s]"
    except Exception as e:
        return 1, f"[ERROR: {e}]"


def setup_container(instance: dict, mode: str) -> str:
    """
    Create a Docker container with the repo at the correct commit.
    Uses swebench env images (pre-built) for proper environment.
    """
    instance_id = instance['instance_id']
    container = f"swe_{instance_id.replace('__', '_').replace('-', '_')}_{mode}"

    # Remove existing
    subprocess.run(["sudo", "docker", "rm", "-f", container],
                   capture_output=True, timeout=30)

    # Determine which env image to use
    from swebench.harness.test_spec.test_spec import make_test_spec
    spec = make_test_spec(instance, namespace='swebench',
                          instance_image_tag='latest', env_image_tag='latest')
    env_image = spec.env_image_key

    # Check image exists
    check = subprocess.run(["sudo", "docker", "image", "inspect", env_image],
                           capture_output=True, timeout=30)
    if check.returncode != 0:
        raise RuntimeError(f"Env image not found: {env_image}. Run image build first.")

    print(f"    Container: {container}")
    print(f"    Image: {env_image}")

    # Start container
    subprocess.run(
        ["sudo", "docker", "run", "-d", "--name", container,
         "-w", "/testbed", env_image, "sleep", "7200"],
        capture_output=True, check=True, timeout=60
    )

    # Setup repo at correct commit
    base_commit = instance['base_commit']
    repo = instance['repo']

    # Clone if not present, checkout correct commit
    rc, out = docker_exec(container, "ls /testbed/.git", timeout=10)
    if rc != 0:
        # Clone and setup
        docker_exec(container, f"git clone https://github.com/{repo}.git /testbed_new && "
                    f"mv /testbed_new/* /testbed/ 2>/dev/null; "
                    f"mv /testbed_new/.* /testbed/ 2>/dev/null; "
                    f"rm -rf /testbed_new", timeout=180)

    # Ensure we're at the right commit
    docker_exec(container, f"cd /testbed && git fetch --all 2>/dev/null; "
                f"git checkout -f {base_commit} 2>/dev/null || "
                f"git reset --hard {base_commit}", timeout=120)
    docker_exec(container, "cd /testbed && git checkout -b agent_work 2>/dev/null || true", timeout=10)

    # Install the package
    docker_exec(container, "cd /testbed && pip install -e . -q 2>/dev/null || true", timeout=180)

    return container


def cleanup_container(container: str):
    """Remove Docker container."""
    subprocess.run(["sudo", "docker", "rm", "-f", container],
                   capture_output=True, timeout=30)


def extract_patch(container: str) -> str:
    """Extract git diff from the container."""
    _, patch = docker_exec(container, "cd /testbed && git diff")
    return patch.strip()


# ============================================================
# Agent
# ============================================================

SYSTEM_PROMPT = """You are an expert software engineer fixing bugs in a Python repository.
The repository is at /testbed. You have a bash shell.

## How to work:
1. Understand the bug from the problem statement
2. Find relevant source code (use grep, find, cat)
3. Read the exact lines you want to change
4. Make a minimal fix using a Python edit script
5. Verify by running the specific failing test

## For editing files, use a Python script (most reliable):
```bash
python3 << 'EDITEOF'
with open('/testbed/path/to/file.py', 'r') as f:
    content = f.read()

# Read the EXACT text you want to replace by printing it first
old = '''exact old text'''
new = '''exact new text'''
assert old in content, f"old text not found!"
content = content.replace(old, new, 1)

with open('/testbed/path/to/file.py', 'w') as f:
    f.write(content)
print("Edit applied successfully")
EDITEOF
```

## Running Django tests (this is a Django repository):
Django has its own test runner at /testbed/tests/runtests.py.
```bash
cd /testbed/tests && python3 runtests.py <app_label>.<TestClass>.<test_method> --parallel 1
```
Examples:
- `cd /testbed/tests && python3 runtests.py utils_tests.test_numberformat`
- `cd /testbed/tests && python3 runtests.py delete.tests.FastDeleteTests.test_fast_delete_combined_relationships`
- `cd /testbed/tests && python3 runtests.py expressions.tests.BasicExpressionsTests`
Do NOT use `manage.py test` or `python -m unittest` — they will fail.

## Rules:
- Make MINIMAL changes to fix the bug
- Do NOT modify test files
- After editing, ALWAYS run the failing test to verify your fix
- When the test passes, say: DONE
- You have maximum 15 steps — be efficient
"""


def build_task_prompt(instance: dict) -> str:
    """Build the initial task prompt."""
    fail_to_pass = instance.get('FAIL_TO_PASS', '')
    if isinstance(fail_to_pass, str):
        try:
            fail_to_pass = json.loads(fail_to_pass)
        except:
            pass

    tests_str = ""
    test_commands = []
    if isinstance(fail_to_pass, list):
        tests_str = "\n".join(f"  - {t}" for t in fail_to_pass[:5])

        # Generate test commands for Django
        for test_id in fail_to_pass[:3]:
            match = re.match(r'(\w+)\s+\(([^)]+)\)', test_id)
            if match:
                method, path = match.groups()
                parts = path.split('.')
                if len(parts) >= 2:
                    test_label = f"{parts[0]}.{parts[1]}"
                    test_commands.append(f"cd /testbed/tests && python3 runtests.py {test_label} --parallel 1")

    test_cmd_str = ""
    if test_commands:
        test_cmd_str = f"\n\n## How to run the test:\n```bash\n{test_commands[0]}\n```"

    return f"""Fix the following bug:

{instance['problem_statement']}

## Failing test(s):
{tests_str}
{test_cmd_str}

Start by finding the relevant source code. Use grep and cat to understand the codebase, then make the fix."""


def load_scaffold_prompt() -> str:
    """Load EDIT_A scaffold prompt."""
    path = PROMPTS_DIR / "scaffolding" / "EDIT_A_reread_file.txt"
    return path.read_text()


class BashAgent:
    """Minimal bash-based agent for SWE-bench."""

    def __init__(self, mode: str = "control"):
        self.mode = mode
        self.client = OpenAI(api_key=VENUS_API_KEY, base_url=VENUS_BASE_URL)
        self.scaffold_prompt = load_scaffold_prompt() if mode == "scaffold" else None

    def call_llm(self, messages: list) -> str:
        """Call LLM with rate limiting and budget tracking."""
        global _api_calls_made
        if _api_calls_made >= MAX_API_CALLS:
            raise RuntimeError(f"BUDGET EXHAUSTED: {_api_calls_made}/{MAX_API_CALLS}")
        time.sleep(RATE_LIMIT_DELAY)
        response = self.client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096,
            temperature=0.0,
        )
        _api_calls_made += 1
        return response.choices[0].message.content

    def parse_command(self, response: str) -> Optional[str]:
        """Extract bash command from LLM response."""
        pattern = r'```(?:bash|sh)?\s*\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            cmd = matches[-1].strip()
            # Safety: limit command length
            if len(cmd) > 5000:
                cmd = cmd[:5000]
            return cmd
        return None

    def is_done(self, response: str, steps: list) -> bool:
        """Check if agent declares completion. Must have actually made edits."""
        # Must have "DONE" explicitly
        if "DONE" not in response:
            return False
        # Must have made at least one edit (git diff should be non-empty)
        has_edit = any(
            any(x in (s.get('command', '') or '') for x in
                ['replace(', 'f.write', 'sed ', 'python3 <<', 'content ='])
            for s in steps
        )
        # Only accept DONE if some edit was actually attempted
        return has_edit

    def detect_edit_error(self, steps: list) -> bool:
        """
        Detect if an edit attempt has been made and then a subsequent command
        failed - the classic EDIT cascade pattern. This is broader than just
        detecting a failed str_replace; it catches:
        - Edit succeeded but test fails afterward
        - Edit command itself failed (assertion error, file not found)
        - Agent is stuck in a loop after editing
        """
        # Find the last edit step
        edit_step = None
        for s in steps:
            cmd = s.get('command', '') or ''
            if any(x in cmd for x in ['replace(', 'content =', 'f.write',
                                        'sed ', 'python3 <<', 'EDITEOF',
                                        'open(', 'str_replace']):
                edit_step = s

        if edit_step is None:
            return False

        # Check if any step AFTER the edit had an error
        found_edit = False
        for s in steps:
            if s is edit_step:
                found_edit = True
                # Check if the edit itself failed
                if s.get('returncode', 0) != 0:
                    return True
                output = (s.get('output', '') or '').lower()
                if 'assert' in output or 'not found' in output or 'error' in output:
                    return True
                continue
            if found_edit and s.get('returncode', 0) != 0:
                return True

        return False

    def run(self, instance: dict) -> dict:
        """Run agent on one instance. Returns trajectory dict."""
        global _api_calls_made
        instance_id = instance['instance_id']
        start_calls = _api_calls_made

        trajectory = {
            "instance_id": instance_id,
            "mode": self.mode,
            "model": MODEL,
            "steps": [],
            "patch": "",
            "scaffold_injected": False,
            "scaffold_step": None,
            "done": False,
            "error": None,
            "api_calls_used": 0,
            "timestamp": datetime.now().isoformat(),
        }

        container = None
        try:
            # Setup container
            container = setup_container(instance, self.mode)

            # Conversation
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_task_prompt(instance)},
            ]

            scaffold_injected = False

            for step_num in range(1, MAX_STEPS + 1):
                # Call LLM
                response = self.call_llm(messages)

                # Check if done (must have made edits first)
                if self.is_done(response, trajectory["steps"]):
                    trajectory["steps"].append({
                        "step": step_num,
                        "response_preview": response[:500],
                        "command": None,
                        "output": "DONE declared",
                        "returncode": 0,
                    })
                    trajectory["done"] = True
                    break

                # Parse command
                command = self.parse_command(response)
                if not command:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content":
                        "You MUST provide exactly one bash command in a ```bash\\n...\\n``` code block. "
                        "Do not explain — just give the command. For example:\n"
                        "```bash\ngrep -rn 'pattern' /testbed/\n```"})
                    trajectory["steps"].append({
                        "step": step_num,
                        "response_preview": response[:500],
                        "command": None,
                        "output": "[no command parsed - reprompted]",
                        "returncode": -1,
                    })
                    continue

                # Execute
                returncode, output = docker_exec(container, command)

                trajectory["steps"].append({
                    "step": step_num,
                    "response_preview": response[:500],
                    "command": command[:500],
                    "output": output[:2000],
                    "returncode": returncode,
                })

                # Check for edit error → inject scaffold (check history of steps)
                if (self.mode == "scaffold" and not scaffold_injected
                    and self.detect_edit_error(trajectory["steps"])):
                    scaffold_injected = True
                    trajectory["scaffold_injected"] = True
                    trajectory["scaffold_step"] = step_num

                    messages.append({"role": "assistant", "content": response})
                    feedback = f"Command output (exit code {returncode}):\n{output}\n\n---\n\n{self.scaffold_prompt}"
                    messages.append({"role": "user", "content": feedback})
                    print(f"      ★ SCAFFOLD INJECTED at step {step_num}")
                else:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Command output (exit code {returncode}):\n{output}"})

            # Extract patch
            if container:
                trajectory["patch"] = extract_patch(container)

        except Exception as e:
            trajectory["error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            if container:
                cleanup_container(container)

        trajectory["api_calls_used"] = _api_calls_made - start_calls
        return trajectory


# ============================================================
# Evaluation
# ============================================================

def run_swebench_evaluation(predictions_path: str, run_id: str, instance_ids: list,
                            report_dir: str) -> dict:
    """Run swebench evaluation harness on predictions."""
    print(f"\n  Running swebench harness for {run_id}...")
    os.makedirs(report_dir, exist_ok=True)

    cmd = [
        "/data/home/xiankunlin/miniconda3/envs/emnlp/bin/python",
        "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "princeton-nlp/SWE-bench_Verified",
        "--split", "test",
        "--predictions_path", predictions_path,
        "--run_id", run_id,
        "--instance_ids"] + instance_ids + [
        "--max_workers", "1",
        "--timeout", "300",
        "--namespace", "swebench",
        "--report_dir", report_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    output = result.stdout + result.stderr
    print(f"  Harness exit code: {result.returncode}")

    # Parse results from output
    resolved = 0
    total = len(instance_ids)
    if "Instances resolved:" in output:
        match = re.search(r"Instances resolved:\s*(\d+)", output)
        if match:
            resolved = int(match.group(1))

    # Also try reading the report file
    report_files = list(Path(report_dir).glob(f"*.{run_id}.json"))
    report_data = {}
    if report_files:
        with open(report_files[0]) as f:
            report_data = json.load(f)

    return {
        "run_id": run_id,
        "resolved": resolved,
        "total": total,
        "resolve_rate": resolved / total if total > 0 else 0,
        "output": output[-3000:],
        "report_file": str(report_files[0]) if report_files else None,
        "report_data": report_data,
    }


# ============================================================
# Main workflows
# ============================================================

def save_predictions(trajectories: list, output_dir: Path, label: str) -> str:
    """Save trajectories as swebench predictions file."""
    predictions = []
    for traj in trajectories:
        predictions.append({
            "instance_id": traj["instance_id"],
            "model_name_or_path": f"gpt41_{label}",
            "model_patch": traj.get("patch", ""),
        })

    os.makedirs(output_dir, exist_ok=True)
    pred_path = output_dir / f"predictions_{label}.json"
    with open(pred_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    return str(pred_path)


def load_instances(instance_ids: list) -> dict:
    """Load instances from HuggingFace dataset."""
    os.environ.setdefault('HF_HOME', '/data/home/xiankunlin/.cache/huggingface')
    from datasets import load_dataset
    ds = load_dataset('princeton-nlp/SWE-bench_Verified', split='test',
                      cache_dir='/data/home/xiankunlin/.cache/huggingface/datasets')
    instances = {}
    for item in ds:
        if item['instance_id'] in instance_ids:
            instances[item['instance_id']] = dict(item)
    return instances


def run_pilot():
    """Run full pilot: 5 instances × {control, scaffold}, then evaluate."""
    global _api_calls_made

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"pilot_{timestamp}"
    output_dir = RESULTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"E2E SCAFFOLD PILOT EXPERIMENT")
    print(f"Run ID: {run_id}")
    print(f"Instances: {len(PILOT_INSTANCES)} (EDIT-type, django)")
    print(f"Model: {MODEL} via Venus API")
    print(f"Max steps: {MAX_STEPS}, Rate limit: {RATE_LIMIT_DELAY}s")
    print(f"Budget: {MAX_API_CALLS} API calls")
    print("=" * 70)

    # Load instances
    print("\nLoading instances...")
    instances = load_instances(PILOT_INSTANCES)
    print(f"  Loaded {len(instances)} instances")
    for iid in PILOT_INSTANCES:
        if iid not in instances:
            print(f"  WARNING: {iid} not found in dataset!")

    # Run both modes
    all_trajectories = {}

    for mode in ["control", "scaffold"]:
        print(f"\n{'='*60}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'='*60}")

        agent = BashAgent(mode=mode)
        mode_trajectories = []

        for i, iid in enumerate(PILOT_INSTANCES):
            if iid not in instances:
                continue
            if _api_calls_made >= MAX_API_CALLS:
                print(f"\n  !! BUDGET EXHAUSTED ({_api_calls_made}/{MAX_API_CALLS}) !!")
                break

            print(f"\n  [{mode} {i+1}/{len(PILOT_INSTANCES)}] {iid}")
            print(f"    Budget: {_api_calls_made}/{MAX_API_CALLS} calls used")

            trajectory = agent.run(instances[iid])
            mode_trajectories.append(trajectory)

            # Summary
            patch_lines = len(trajectory["patch"].splitlines()) if trajectory["patch"] else 0
            print(f"    → done={trajectory['done']}, patch={patch_lines} lines, "
                  f"calls={trajectory['api_calls_used']}, "
                  f"scaffold={'step '+str(trajectory['scaffold_step']) if trajectory['scaffold_injected'] else 'N/A'}")

        all_trajectories[mode] = mode_trajectories

    # Save trajectories
    with open(output_dir / "all_trajectories.json", 'w') as f:
        json.dump(all_trajectories, f, indent=2, default=str)

    # Save predictions in swebench format
    pred_paths = {}
    for mode, trajs in all_trajectories.items():
        if trajs:
            pred_paths[mode] = save_predictions(trajs, output_dir, mode)
            print(f"\n  Predictions ({mode}): {pred_paths[mode]}")

    # Evaluate with swebench harness
    print(f"\n{'='*70}")
    print("SWEBENCH EVALUATION")
    print("=" * 70)

    eval_results = {}
    for mode, pred_path in pred_paths.items():
        # Only evaluate instances that produced non-empty patches
        mode_trajs = all_trajectories[mode]
        eval_ids = [t["instance_id"] for t in mode_trajs if t.get("patch")]
        if not eval_ids:
            print(f"\n  {mode}: No patches to evaluate")
            eval_results[mode] = {"resolved": 0, "total": len(PILOT_INSTANCES)}
            continue

        eval_result = run_swebench_evaluation(
            pred_path, f"{mode}_{run_id}", eval_ids,
            str(output_dir / "reports")
        )
        eval_results[mode] = eval_result
        print(f"  {mode}: {eval_result['resolved']}/{eval_result['total']} resolved")

    # Final report
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print("=" * 70)

    control_resolved = eval_results.get("control", {}).get("resolved", 0)
    scaffold_resolved = eval_results.get("scaffold", {}).get("resolved", 0)
    total = len(PILOT_INSTANCES)

    print(f"\n  Control:  {control_resolved}/{total} resolved ({100*control_resolved/total:.0f}%)")
    print(f"  Scaffold: {scaffold_resolved}/{total} resolved ({100*scaffold_resolved/total:.0f}%)")
    print(f"  Delta:    +{scaffold_resolved - control_resolved}")
    print(f"\n  API calls used: {_api_calls_made}/{MAX_API_CALLS}")
    print(f"  Output: {output_dir}")

    # Per-instance breakdown
    print(f"\n  Per-instance:")
    for mode in ["control", "scaffold"]:
        print(f"\n    {mode.upper()}:")
        for traj in all_trajectories.get(mode, []):
            patch_lines = len(traj["patch"].splitlines()) if traj["patch"] else 0
            print(f"      {traj['instance_id']:40s} patch={patch_lines:3d}L "
                  f"done={traj['done']!s:5s} err={traj.get('error', '-')!s:.30s}")

    # Save final report
    report = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": {
            "model": MODEL,
            "max_steps": MAX_STEPS,
            "rate_limit": RATE_LIMIT_DELAY,
            "instances": PILOT_INSTANCES,
        },
        "results": {
            "control_resolved": control_resolved,
            "scaffold_resolved": scaffold_resolved,
            "total": total,
            "delta": scaffold_resolved - control_resolved,
        },
        "api_calls_total": _api_calls_made,
        "eval_results": {k: {kk: vv for kk, vv in v.items() if kk != 'output'}
                         for k, v in eval_results.items()},
    }
    with open(output_dir / "final_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Report: {output_dir / 'final_report.json'}")
    return output_dir


def run_single(instance_id: str, mode: str):
    """Run a single instance."""
    print(f"Running {instance_id} in {mode} mode...")
    instances = load_instances([instance_id])
    if instance_id not in instances:
        print(f"ERROR: Instance {instance_id} not found")
        return

    agent = BashAgent(mode=mode)
    trajectory = agent.run(instances[instance_id])

    # Save
    output_dir = RESULTS_DIR / "single"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{instance_id}_{mode}.json"
    with open(output_file, 'w') as f:
        json.dump(trajectory, f, indent=2, default=str)

    patch_lines = len(trajectory["patch"].splitlines()) if trajectory["patch"] else 0
    print(f"\nResult: done={trajectory['done']}, patch={patch_lines} lines")
    print(f"Saved: {output_file}")

    # Evaluate if we have a patch
    if trajectory.get("patch"):
        pred_path = save_predictions([trajectory], output_dir, f"{instance_id}_{mode}")
        eval_result = run_swebench_evaluation(
            pred_path, f"single_{instance_id}_{mode}",
            [instance_id], str(output_dir / "reports")
        )
        print(f"\nEvaluation: {'RESOLVED' if eval_result['resolved'] > 0 else 'FAILED'}")


def evaluate_existing(run_id: str):
    """Evaluate existing predictions from a previous run."""
    run_dir = RESULTS_DIR / run_id
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    for mode in ["control", "scaffold"]:
        pred_file = run_dir / f"predictions_{mode}.json"
        if pred_file.exists():
            with open(pred_file) as f:
                preds = json.load(f)
            instance_ids = [p["instance_id"] for p in preds if p.get("model_patch")]
            if instance_ids:
                print(f"\nEvaluating {mode} ({len(instance_ids)} instances)...")
                eval_result = run_swebench_evaluation(
                    str(pred_file), f"{mode}_{run_id}", instance_ids,
                    str(run_dir / "reports")
                )
                print(f"  {mode}: {eval_result['resolved']}/{eval_result['total']} resolved")


def main():
    parser = argparse.ArgumentParser(description="E2E SWE-bench scaffold evaluation")
    parser.add_argument("--instance_id", "-i", type=str)
    parser.add_argument("--mode", choices=["control", "scaffold"], default="control")
    parser.add_argument("--pilot", action="store_true", help="Run full pilot")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing run")
    parser.add_argument("--run_id", type=str)
    args = parser.parse_args()

    if args.pilot:
        run_pilot()
    elif args.evaluate and args.run_id:
        evaluate_existing(args.run_id)
    elif args.instance_id:
        run_single(args.instance_id, args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
