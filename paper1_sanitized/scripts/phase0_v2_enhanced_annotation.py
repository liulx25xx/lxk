#!/usr/bin/env python3
"""
Phase 0 v2: Enhanced rule-based failure annotation using gold patch comparison.

Key improvement over v1: Uses gold patch files to distinguish LOC (wrong file)
from PLAN/TEST/LOGIC (right file, wrong approach/logic).

Cost: ¥0
Output: results/phase0_annotations/phase0_v2_annotations.json
"""

import json, os, re, sys
from pathlib import Path
from collections import Counter
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase0_annotations"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_gold_patches():
    """Load gold patch file paths for all SWE-bench Verified instances."""
    from datasets import load_dataset
    swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold = {}
    for inst in swe:
        files = re.findall(r'--- a/(.*?)\n', inst['patch'])
        gold[inst['instance_id']] = {
            'files': set(files),
            'patch': inst['patch'],
            'problem_statement': inst['problem_statement'][:500],
        }
    return gold


def load_failed_trajectories():
    """Load 143 failed O1-agent trajectories overlapping with our subset."""
    from datasets import load_dataset
    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')

    with open(PROJECT_ROOT / "data" / "swebench_subset.json") as f:
        subset = json.load(f)
    our_ids = set(inst['instance_id'] for inst in subset['instances'])

    failed = []
    for row in ds:
        iid = row['issue_name']
        if iid in our_ids and not row.get('resolved', row.get('success', False)):
            failed.append({
                'instance_id': iid,
                'project': row['project'],
                'num_turns': row['num_turns'],
                'conversation': row['full_conversation_jsonl'],
                'patch': row.get('patch', ''),
                'patch_applied': row.get('patch_successfully_applied', False),
            })
    return failed


def files_overlap(gold_files, agent_files):
    """Check if agent edited any of the gold patch files (suffix matching)."""
    for gf in gold_files:
        for af in agent_files:
            if af.endswith('/' + gf) or af == gf:
                return True
    return False


def extract_agent_actions(conversation_json):
    """Parse trajectory to extract agent's edited files and action patterns."""
    try:
        data = json.loads(conversation_json)
    except json.JSONDecodeError:
        return {'edited_files': set(), 'searched_files': set(), 'steps': [],
                'edit_count': 0, 'search_count': 0, 'test_count': 0, 'error_count': 0}

    edited_files = set()
    searched_files = set()
    steps = []
    edit_count = 0
    search_count = 0
    test_count = 0
    error_count = 0
    str_replace_errors = 0

    for turn in data:
        # Agent action (response)
        resp = turn.get('response', {})
        choices = resp.get('choices', [])
        content = ''
        if choices and isinstance(choices[0], dict):
            content = choices[0].get('message', {}).get('content', '')

        if content:
            # Extract edited files
            file_paths = re.findall(r'<parameter=path>(.*?)</parameter>', content)
            has_edit = bool(re.search(r'str_replace|create_file|insert', content))
            has_search = bool(re.search(r'find_file|search_dir|grep|view', content))
            has_test = bool(re.search(r'pytest|python.*test|run_tests', content, re.I))

            if has_edit:
                edit_count += 1
                edited_files.update(file_paths)
            if has_search:
                search_count += 1
                searched_files.update(file_paths)
            if has_test:
                test_count += 1

            steps.append({
                'role': 'assistant',
                'has_edit': has_edit,
                'has_search': has_search,
                'has_test': has_test,
                'files': file_paths,
            })

        # Observation (last user message in context)
        msgs = turn.get('messages', [])
        user_msgs = [m for m in msgs if m.get('role') == 'user']
        if user_msgs:
            obs = user_msgs[-1].get('content', '')
            if isinstance(obs, list):
                obs = ' '.join(c.get('text', '') for c in obs if isinstance(c, dict))

            has_error = bool(re.search(
                r'Error|FAILED|No replacement|did not appear verbatim|SyntaxError|Traceback',
                obs
            ))
            is_str_replace_error = bool(re.search(r'No replacement|did not appear verbatim', obs))
            is_test_error = bool(re.search(r'FAILED|AssertionError', obs))

            if has_error:
                error_count += 1
            if is_str_replace_error:
                str_replace_errors += 1

            steps.append({
                'role': 'observation',
                'has_error': has_error,
                'is_str_replace_error': is_str_replace_error,
                'is_test_error': is_test_error,
            })

    return {
        'edited_files': edited_files,
        'searched_files': searched_files,
        'steps': steps,
        'edit_count': edit_count,
        'search_count': search_count,
        'test_count': test_count,
        'error_count': error_count,
        'str_replace_errors': str_replace_errors,
        'total_turns': len(data),
    }


def classify_failure_v2(actions, gold_info, patch_applied):
    """
    Enhanced classification using gold patch comparison.

    Decision tree:
    1. Did agent edit the CORRECT file? (gold patch file overlap)
       NO → LOC (localization failure)
       YES → proceed
    2. Did str_replace commands fail repeatedly?
       YES → EDIT (edit-application failure)
    3. Did patch apply but tests fail?
       YES → LOGIC (logic error) or TEST (misinterpretation)
    4. Everything else → PLAN (wrong strategy)
    """
    edited_correct_file = files_overlap(gold_info['files'], actions['edited_files'])
    total_steps = len([s for s in actions['steps'] if s.get('role') == 'assistant'])

    # LOC: Agent never edited the correct file
    if not edited_correct_file and actions['edit_count'] > 0:
        return ('LOC', 0.85, f"Edited {actions['edit_count']} files but none overlap with gold: {gold_info['files']}")

    # LOC: Agent only searched, never committed to editing
    if actions['edit_count'] == 0 and actions['search_count'] >= 3:
        return ('LOC', 0.7, f"Never edited anything after {actions['search_count']} searches")

    # EDIT: Multiple str_replace failures
    if actions['str_replace_errors'] >= 2:
        return ('EDIT', 0.85, f"{actions['str_replace_errors']} str_replace failures")

    # EDIT: Even 1 str_replace failure if it's dominant pattern
    if actions['str_replace_errors'] >= 1 and not patch_applied:
        ratio = actions['str_replace_errors'] / max(actions['error_count'], 1)
        if ratio >= 0.5:
            return ('EDIT', 0.7, f"str_replace errors are {ratio:.0%} of all errors")

    # LOGIC: Patch applied (edit worked) but still failed overall
    if patch_applied:
        # Check if there were test failures in observations
        test_errors = sum(1 for s in actions['steps'] if s.get('is_test_error'))
        if test_errors > 0:
            return ('LOGIC', 0.75, f"Patch applied but {test_errors} test failures")
        else:
            return ('LOGIC', 0.6, "Patch applied but issue not resolved (likely wrong logic)")

    # TEST: Agent ran tests multiple times and kept failing differently
    test_errors = sum(1 for s in actions['steps'] if s.get('is_test_error'))
    if test_errors >= 2 and edited_correct_file:
        return ('TEST', 0.6, f"{test_errors} test failures after finding correct file")

    # PLAN: Agent found correct file but took too many unproductive steps
    if edited_correct_file and total_steps >= 15:
        return ('PLAN', 0.5, f"Found correct file but {total_steps} steps without resolution")

    # LOC: Default for agents that searched a lot but couldn't converge
    if actions['search_count'] > actions['edit_count'] * 2:
        return ('LOC', 0.5, f"Search-heavy ({actions['search_count']} searches, {actions['edit_count']} edits)")

    # PLAN: Default fallback
    return ('PLAN', 0.4, f"No clear pattern: {total_steps} steps, {actions['edit_count']} edits")


def analyze_cascade_v2(steps):
    """Identify first error and measure cascade waste."""
    first_error_idx = None
    total_assistant_steps = 0

    for i, step in enumerate(steps):
        if step.get('role') == 'assistant':
            total_assistant_steps += 1
        if step.get('has_error') and first_error_idx is None:
            first_error_idx = i

    if first_error_idx is None or total_assistant_steps == 0:
        return {'first_error_step': None, 'cascade_length': 0,
                'waste_ratio': 0.0, 'total_steps': total_assistant_steps}

    steps_after_error = sum(1 for s in steps[first_error_idx:] if s.get('role') == 'assistant')
    waste_ratio = steps_after_error / total_assistant_steps

    return {
        'first_error_step': first_error_idx,
        'cascade_length': steps_after_error,
        'waste_ratio': round(waste_ratio, 3),
        'total_steps': total_assistant_steps,
    }


def main():
    print("=" * 60)
    print("Phase 0 v2: Enhanced Failure Annotation (Gold Patch)")
    print("=" * 60)

    print("\n[1/4] Loading data...")
    gold = load_gold_patches()
    trajs = load_failed_trajectories()
    print(f"   Gold patches: {len(gold)} instances")
    print(f"   Failed trajectories: {len(trajs)}")

    print(f"\n[2/4] Annotating {len(trajs)} trajectories...")
    results = []
    type_counts = Counter()
    cascade_stats = []
    skipped = 0

    for i, traj in enumerate(trajs):
        iid = traj['instance_id']
        if iid not in gold:
            skipped += 1
            continue

        actions = extract_agent_actions(traj['conversation'])
        failure_type, confidence, evidence = classify_failure_v2(
            actions, gold[iid], traj['patch_applied']
        )
        cascade = analyze_cascade_v2(actions['steps'])

        result = {
            'instance_id': iid,
            'project': traj['project'],
            'failure_type': failure_type,
            'confidence': confidence,
            'evidence': evidence,
            'cascade': cascade,
            'stats': {
                'edit_count': actions['edit_count'],
                'search_count': actions['search_count'],
                'test_count': actions['test_count'],
                'error_count': actions['error_count'],
                'str_replace_errors': actions['str_replace_errors'],
                'total_turns': actions['total_turns'],
                'edited_correct_file': files_overlap(gold[iid]['files'], actions['edited_files']),
            },
            'gold_files': list(gold[iid]['files']),
            'agent_edited_files': list(actions['edited_files'])[:10],
            'patch_applied': traj['patch_applied'],
        }
        results.append(result)
        type_counts[failure_type] += 1
        cascade_stats.append(cascade)

        if (i + 1) % 30 == 0:
            print(f"   Processed {i+1}/{len(trajs)}")

    # Save
    print(f"\n[3/4] Saving ({skipped} skipped, {len(results)} annotated)...")

    # Cascade summary
    valid_cascades = [c for c in cascade_stats if c['first_error_step'] is not None]
    cascade_summary = {
        'mean_waste_ratio': round(sum(c['waste_ratio'] for c in valid_cascades) / max(len(valid_cascades), 1), 3),
        'median_waste_ratio': round(sorted(c['waste_ratio'] for c in valid_cascades)[len(valid_cascades)//2], 3) if valid_cascades else 0,
        'mean_cascade_length': round(sum(c['cascade_length'] for c in valid_cascades) / max(len(valid_cascades), 1), 1),
        'trajectories_with_errors': len(valid_cascades),
        'total_trajectories': len(results),
    }

    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'version': 'v2',
            'method': 'rule-based + gold patch comparison',
            'total_annotated': len(results),
            'api_cost': 0,
        },
        'type_distribution': dict(type_counts.most_common()),
        'cascade_summary': cascade_summary,
        'annotations': results,
    }

    output_path = RESULTS_DIR / "phase0_v2_annotations.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"   Saved: {output_path}")

    # Summary
    print(f"\n[4/4] Results:")
    print(f"{'=' * 60}")
    print(f"Total annotated: {len(results)}")
    print(f"\nFailure Type Distribution:")
    for ftype, count in type_counts.most_common():
        pct = 100 * count / len(results)
        bar = '█' * int(pct / 2)
        print(f"  {ftype:8s}: {count:3d} ({pct:4.1f}%) {bar}")

    print(f"\nCascade Statistics:")
    print(f"  Mean waste ratio:   {cascade_summary['mean_waste_ratio']:.1%}")
    print(f"  Median waste ratio: {cascade_summary['median_waste_ratio']:.1%}")
    print(f"  Mean cascade:       {cascade_summary['mean_cascade_length']:.1f} steps")
    print(f"  With errors:        {cascade_summary['trajectories_with_errors']}/{len(results)}")

    # Per-type cascade
    print(f"\nWaste Ratio by Failure Type:")
    for ftype in type_counts:
        type_cascades = [r['cascade'] for r in results if r['failure_type'] == ftype and r['cascade']['first_error_step'] is not None]
        if type_cascades:
            avg = sum(c['waste_ratio'] for c in type_cascades) / len(type_cascades)
            print(f"  {ftype:8s}: {avg:.1%} waste ({len(type_cascades)} with cascade)")

    # Gate check
    print(f"\n{'=' * 60}")
    print("Go/No-Go Gate:")
    n_types = len(type_counts)
    min_count = min(type_counts.values()) if type_counts else 0
    print(f"  Types detected: {n_types} (need ≥4)")
    print(f"  Smallest type:  {min_count} instances (want ≥10)")
    print(f"  Total:          {len(results)} (need ≥100)")

    if n_types >= 4 and len(results) >= 100:
        print(f"\n  🟢 GATE PASSED")
    elif n_types >= 3 and len(results) >= 100:
        print(f"\n  🟡 PARTIAL PASS — {n_types} types, may need LLM refinement for remaining")
    else:
        print(f"\n  🔴 GATE FAILED")


if __name__ == "__main__":
    main()
