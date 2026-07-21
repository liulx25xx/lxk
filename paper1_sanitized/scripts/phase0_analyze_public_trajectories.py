#!/usr/bin/env python3
"""
Phase 0: Download public trajectories and perform rule-based failure annotation.

This script:
1. Downloads 143 failed O1-agent trajectories that overlap with our 200-instance subset
2. Parses each trajectory into structured action-observation steps
3. Applies rule-based heuristics to classify failure type (LOC/EDIT/LOGIC/TEST/PLAN)
4. Performs cascade analysis (identifies first error step, measures waste)
5. Saves everything for downstream use (scaffolding experiments, paper writing)

Cost: ¥0 (no API calls)
Output: results/phase0_annotations/

Go/No-Go gate: ≥100 annotated failures with reasonable type distribution
"""

import json, os, re, sys, time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase0_annotations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =================================================================
# Step 1: Load trajectories from HuggingFace
# =================================================================
def load_trajectories():
    """Load O1 trajectories and filter to our subset failures."""
    from datasets import load_dataset

    print("[1/5] Loading O1 trajectories from HuggingFace...")
    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')

    # Load our 200-instance subset
    with open(DATA_DIR / "swebench_subset.json") as f:
        subset = json.load(f)
    our_ids = set(inst['instance_id'] for inst in subset['instances'])

    # Filter: overlapping AND failed
    failed_trajs = []
    for row in ds:
        iid = row['issue_name']
        if iid in our_ids and not row.get('resolved', row.get('success', False)):
            failed_trajs.append({
                'instance_id': iid,
                'project': row['project'],
                'num_turns': row['num_turns'],
                'full_conversation': row['full_conversation_jsonl'],
                'patch': row.get('patch', ''),
                'patch_applied': row.get('patch_successfully_applied', False),
            })

    print(f"   Found {len(failed_trajs)} failed trajectories in our subset")
    return failed_trajs


# =================================================================
# Step 2: Parse trajectory into structured steps
# =================================================================
def parse_trajectory(traj_jsonl):
    """Parse conversation into action-observation steps.

    Format: JSON array of turns, each with 'messages' (context) and 'response' (agent action).
    """
    steps = []
    try:
        data = json.loads(traj_jsonl)
        if not isinstance(data, list):
            return steps

        for turn in data:
            msgs = turn.get('messages', [])
            resp = turn.get('response', {})

            # Extract observation from the last user message
            user_msgs = [m for m in msgs if m.get('role') == 'user']
            if user_msgs:
                last_obs = user_msgs[-1]
                content = last_obs.get('content', '')
                if isinstance(content, list):
                    content = ' '.join(c.get('text', '') for c in content if isinstance(c, dict))
                if content and ('EXECUTION RESULT' in content or len(content) > 50):
                    steps.append({
                        'role': 'observation',
                        'content': content[:2000],
                        'has_error': bool(re.search(
                            r'Error|FAILED|No replacement|did not appear verbatim|SyntaxError|'
                            r'ImportError|ModuleNotFoundError|FileNotFoundError|Traceback',
                            content
                        )),
                        'error_type': classify_observation_error(content),
                    })

            # Extract agent action from response
            resp_content = ''
            if isinstance(resp, dict):
                choices = resp.get('choices', [])
                if choices and isinstance(choices, list):
                    resp_content = choices[0].get('message', {}).get('content', '') if isinstance(choices[0], dict) else ''
            if resp_content:
                steps.append({
                    'role': 'assistant',
                    'content': resp_content[:2000],
                    'has_edit': bool(re.search(r'str_replace|create_file|insert', resp_content)),
                    'has_search': bool(re.search(r'find_file|search_dir|grep|view|cat\s', resp_content)),
                    'has_bash': bool(re.search(r'execute_bash', resp_content)),
                    'has_test': bool(re.search(r'pytest|python.*test|run_tests|unittest', resp_content, re.I)),
                })
    except (json.JSONDecodeError, Exception) as e:
        pass

    return steps


def classify_observation_error(content):
    """Classify the type of error in an observation."""
    if re.search(r'No replacement was performed|did not appear verbatim', content):
        return 'str_replace_fail'
    elif re.search(r'SyntaxError|IndentationError', content):
        return 'syntax_error'
    elif re.search(r'ImportError|ModuleNotFoundError', content):
        return 'import_error'
    elif re.search(r'FAILED.*test|AssertionError|assert.*==', content):
        return 'test_failure'
    elif re.search(r'FileNotFoundError|No such file', content):
        return 'file_not_found'
    elif re.search(r'Error|error|ERROR', content):
        return 'generic_error'
    return None


# =================================================================
# Step 3: Rule-based failure type classification
# =================================================================
def classify_failure_type(steps, patch_applied, patch_content):
    """
    Classify failure into one of 5 types based on trajectory patterns.

    Types:
    - LOC: Agent never found/edited the correct file/function
    - EDIT: Agent found right place but edit command failed (str_replace mismatch)
    - LOGIC: Edit applied but logic is wrong (tests fail)
    - TEST: Agent misunderstood the requirements
    - PLAN: Agent took a fundamentally wrong approach

    Returns: (type, confidence, evidence)
    """
    # Count patterns
    edit_errors = sum(1 for s in steps if s.get('error_type') == 'str_replace_fail')
    syntax_errors = sum(1 for s in steps if s.get('error_type') == 'syntax_error')
    test_failures = sum(1 for s in steps if s.get('error_type') == 'test_failure')
    search_steps = sum(1 for s in steps if s.get('has_search', False))
    edit_steps = sum(1 for s in steps if s.get('has_edit', False))
    total_steps = len([s for s in steps if s.get('role') == 'assistant'])

    # Decision logic (rule-based, ordered by specificity)

    # EDIT: Multiple str_replace failures dominate the trajectory
    if edit_errors >= 2 and edit_errors / max(total_steps, 1) > 0.15:
        return ('EDIT', 0.8, f'{edit_errors} str_replace failures in {total_steps} steps')

    # LOGIC: Patch applied but tests fail
    if patch_applied and test_failures >= 1:
        return ('LOGIC', 0.7, f'Patch applied but {test_failures} test failures')

    # LOC: Very few edits, lots of searching, or never found target
    if edit_steps <= 1 and search_steps >= 5:
        return ('LOC', 0.6, f'Only {edit_steps} edits after {search_steps} searches')

    # LOC: Agent searched many files without settling
    if search_steps > total_steps * 0.6 and edit_steps <= 2:
        return ('LOC', 0.6, f'Spent {search_steps}/{total_steps} steps searching')

    # TEST: Agent ran tests and then changed approach multiple times
    if test_failures >= 2 and edit_steps >= 3:
        return ('TEST', 0.5, f'{test_failures} test fails with {edit_steps} edit attempts')

    # PLAN: Many steps without convergence, no clear pattern
    if total_steps >= 15 and edit_errors == 0 and test_failures == 0:
        return ('PLAN', 0.4, f'{total_steps} steps without clear failure signal')

    # EDIT: Even 1 str_replace failure if it's the final state
    if edit_errors >= 1 and not patch_applied:
        return ('EDIT', 0.6, f'{edit_errors} str_replace failure(s), patch not applied')

    # Default: PLAN (fundamentally wrong approach)
    return ('PLAN', 0.3, f'No clear pattern: {total_steps} steps, {edit_steps} edits, {search_steps} searches')


# =================================================================
# Step 4: Cascade analysis
# =================================================================
def analyze_cascade(steps):
    """
    Identify the first error and measure cascade (wasted steps after first error).

    Returns: {first_error_step, cascade_length, waste_ratio, recovery_attempts}
    """
    first_error_idx = None
    total_assistant_steps = 0
    errors_seen = 0
    recovery_attempts = 0  # Steps where agent tries something new after error

    for i, step in enumerate(steps):
        if step.get('role') == 'assistant':
            total_assistant_steps += 1

        if step.get('has_error') and first_error_idx is None:
            first_error_idx = i

        if first_error_idx is not None and step.get('role') == 'assistant':
            # Count recovery attempts (edits or searches after first error)
            if step.get('has_edit') or step.get('has_search'):
                recovery_attempts += 1

    if first_error_idx is None:
        return {
            'first_error_step': None,
            'cascade_length': 0,
            'waste_ratio': 0.0,
            'total_steps': total_assistant_steps,
            'recovery_attempts': 0,
        }

    # Steps after first error
    steps_after_error = sum(1 for s in steps[first_error_idx:] if s.get('role') == 'assistant')
    waste_ratio = steps_after_error / max(total_assistant_steps, 1)

    return {
        'first_error_step': first_error_idx,
        'cascade_length': steps_after_error,
        'waste_ratio': round(waste_ratio, 3),
        'total_steps': total_assistant_steps,
        'recovery_attempts': recovery_attempts,
    }


# =================================================================
# Main
# =================================================================
def main():
    print("=" * 60)
    print("Phase 0: Public Trajectory Analysis (Zero Cost)")
    print("=" * 60)
    print()

    # Step 1: Load
    trajs = load_trajectories()

    # Step 2-4: Parse, classify, analyze each trajectory
    print(f"\n[2/5] Parsing and annotating {len(trajs)} trajectories...")
    results = []
    type_counts = Counter()
    cascade_stats = []

    for i, traj in enumerate(trajs):
        steps = parse_trajectory(traj['full_conversation'])

        failure_type, confidence, evidence = classify_failure_type(
            steps, traj['patch_applied'], traj['patch']
        )

        cascade = analyze_cascade(steps)

        result = {
            'instance_id': traj['instance_id'],
            'project': traj['project'],
            'num_turns': traj['num_turns'],
            'num_parsed_steps': len(steps),
            'failure_type': failure_type,
            'type_confidence': confidence,
            'type_evidence': evidence,
            'cascade': cascade,
            'patch_applied': traj['patch_applied'],
        }
        results.append(result)
        type_counts[failure_type] += 1
        cascade_stats.append(cascade)

        if (i + 1) % 20 == 0:
            print(f"   Processed {i+1}/{len(trajs)}")

    # Step 5: Save results
    print(f"\n[3/5] Saving results...")

    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'source': 'AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results',
            'total_trajectories': len(results),
            'api_cost': 0,
        },
        'type_distribution': dict(type_counts),
        'cascade_summary': {
            'mean_waste_ratio': round(sum(c['waste_ratio'] for c in cascade_stats) / len(cascade_stats), 3),
            'mean_cascade_length': round(sum(c['cascade_length'] for c in cascade_stats) / len(cascade_stats), 1),
            'trajectories_with_cascade': sum(1 for c in cascade_stats if c['first_error_step'] is not None),
        },
        'annotations': results,
    }

    output_path = RESULTS_DIR / "phase0_full_annotations.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"   Saved to: {output_path}")

    # Print summary
    print(f"\n[4/5] Summary:")
    print(f"{'=' * 60}")
    print(f"Total annotated: {len(results)}")
    print(f"\nFailure Type Distribution:")
    for ftype, count in type_counts.most_common():
        print(f"  {ftype:8s}: {count:3d} ({100*count/len(results):.0f}%)")

    print(f"\nCascade Statistics:")
    print(f"  Mean waste ratio: {output['cascade_summary']['mean_waste_ratio']:.1%}")
    print(f"  Mean cascade length: {output['cascade_summary']['mean_cascade_length']:.1f} steps")
    print(f"  Trajectories with errors: {output['cascade_summary']['trajectories_with_cascade']}/{len(results)}")

    # Go/No-Go check
    print(f"\n[5/5] Go/No-Go Gate Check:")
    gate_pass = True
    if len(results) < 100:
        print(f"  ❌ Need ≥100 annotated failures, got {len(results)}")
        gate_pass = False
    else:
        print(f"  ✅ {len(results)} annotated failures (≥100)")

    if len(type_counts) < 3:
        print(f"  ❌ Need ≥3 failure types, got {len(type_counts)}")
        gate_pass = False
    else:
        print(f"  ✅ {len(type_counts)} failure types detected")

    min_per_type = min(type_counts.values()) if type_counts else 0
    if min_per_type < 5:
        print(f"  ⚠️  Smallest type has only {min_per_type} instances (want ≥10)")
    else:
        print(f"  ✅ All types have ≥{min_per_type} instances")

    if gate_pass:
        print(f"\n  🟢 GATE PASSED — proceed to Phase 1")
    else:
        print(f"\n  🔴 GATE FAILED — review annotations before proceeding")

    return output


if __name__ == "__main__":
    main()
