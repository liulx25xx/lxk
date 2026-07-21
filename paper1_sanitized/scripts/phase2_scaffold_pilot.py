#!/usr/bin/env python3
"""
Phase 2: Scaffolding Pilot — Test if behavioral scaffolding generalizes across failure types.

This is the CRITICAL experiment. It tests the paper's central claim:
"Behavioral scaffolding works across failure types, but optimal strategy is type-dependent."

Design:
- 4 failure types (LOC, EDIT, LOGIC, PLAN) from Phase 0 annotations
- Per type: best-strategy scaffold + control (no scaffold)
- Model: GPT-4o-mini (cheapest, in Paper 6's "scaffold-sensitive" sweet spot)
- 10 instances per type × 2 conditions = 80 calls
- Evaluation: single-turn repair (like Paper 6), not full SWE-bench

Why single-turn: Paper 6 proved single-turn scaffold works for EDIT. We extend
to other types using the SAME protocol. Full SWE-bench eval comes later (Phase 3).

Task format: Given the agent's trajectory up to the first error + scaffold prompt,
can the model produce a better next action?

Cost: ~¥8 (80 calls × gpt-4o-mini)
Go/No-Go: scaffold > control by ≥15% on ≥2 types

Output: results/phase2_scaffold_pilot/
"""

import json, os, re, sys, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase2_scaffold_pilot"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Venus API
API_KEY = os.environ.get("OPENAI_API_KEY", "<REDACTED_SECRET>")
BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4o-mini"
INTER_CALL_DELAY = 6.5
MAX_CALLS = 100  # budget cap

# Scaffolding strategies per type (from prompts/scaffolding/)
STRATEGIES = {
    "LOC": "LOC_C_test_guided",      # "Run the failing test first to locate the bug"
    "EDIT": "EDIT_A_reread_file",     # "Re-read the file exactly" (proven in Paper 6)
    "LOGIC": "LOGIC_A_test_analysis", # "Read the failing test, compare with your implementation"
    "PLAN": "PLAN_A_step_back",       # "Stop, list 3 strategies, evaluate which works"
}

CONTROL = "CONTROL_no_scaffold"

# Load scaffold prompts
def load_scaffold(name):
    path = PROJECT_ROOT / "prompts" / "scaffolding" / f"{name}.txt"
    if path.exists():
        return path.read_text().strip()
    raise FileNotFoundError(f"Scaffold not found: {path}")


def call_llm(prompt):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    time.sleep(INTER_CALL_DELAY)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  API Error: {str(e)[:100]}")
        time.sleep(10)
        return ""


def build_scaffold_prompt(instance_id, trajectory_context, failure_type, scaffold_text, gold_info):
    """
    Build a single-turn scaffold prompt.

    Format: "Here's what the agent did so far [trajectory]. It failed because [type].
    [scaffold instruction]. What should the agent do next?"
    """
    prompt = f"""You are an expert software engineer helping debug a failed code agent.

The agent was working on issue: {instance_id}
It attempted to fix a bug but failed. Below is a summary of what happened.

## Agent's Trajectory (summarized)
{trajectory_context[:3000]}

## The Problem
The agent's approach failed. Based on analysis, this is a **{failure_type}** failure.

## Your Task
{scaffold_text}

## Instructions
Based on the scaffold guidance above, provide the SINGLE BEST next action the agent should take.
Output your response as a concrete bash command or file edit that would help resolve the issue.
Be specific — give exact file paths, function names, or commands.

## Gold Patch Files (for reference — the fix should modify these)
Files that need changes: {', '.join(gold_info['files'])}

## Response Format
Provide ONLY the next action (one bash command or one file edit). No explanation needed.
"""
    return prompt


def extract_trajectory_context(conversation_json, max_turns=5):
    """Extract first few turns of trajectory as context."""
    try:
        data = json.loads(conversation_json)
    except json.JSONDecodeError:
        return "Unable to parse trajectory"

    context_parts = []
    turns_added = 0

    for turn in data[:max_turns]:
        resp = turn.get('response', {})
        choices = resp.get('choices', [])
        if choices and isinstance(choices[0], dict):
            content = choices[0].get('message', {}).get('content', '')
            if content:
                context_parts.append(f"[Agent action]: {content[:500]}")
                turns_added += 1

        msgs = turn.get('messages', [])
        user_msgs = [m for m in msgs if m.get('role') == 'user']
        if user_msgs:
            obs = user_msgs[-1].get('content', '')
            if isinstance(obs, list):
                obs = ' '.join(c.get('text', '') for c in obs if isinstance(c, dict))
            if obs and 'EXECUTION RESULT' in obs:
                context_parts.append(f"[Observation]: {obs[:300]}")

    return '\n\n'.join(context_parts) if context_parts else "No trajectory available"


def evaluate_response(response, gold_info, failure_type):
    """
    Evaluate if the scaffold-guided response is better than control.

    Metrics:
    - mentions_correct_file: Does it reference the gold patch file?
    - actionable: Does it contain a concrete command/edit?
    - relevant_to_type: Is it addressing the right failure mode?
    """
    if not response:
        return {"score": 0, "mentions_correct_file": False, "actionable": False, "relevant": False}

    # Check if response mentions the correct file
    mentions_file = any(gf.split('/')[-1] in response for gf in gold_info['files'])
    mentions_path = any(gf in response for gf in gold_info['files'])

    # Check if actionable (contains command-like content)
    actionable = bool(re.search(
        r'(cat |grep |find |cd |python |pytest |str_replace|def |class |import )',
        response
    ))

    # Check relevance to failure type
    relevant = False
    if failure_type == "LOC":
        relevant = bool(re.search(r'search|find|grep|locate|look.*for', response, re.I))
    elif failure_type == "EDIT":
        relevant = bool(re.search(r'read|cat|view|exact|character|whitespace', response, re.I))
    elif failure_type == "LOGIC":
        relevant = bool(re.search(r'test|assert|expect|return|logic|fix', response, re.I))
    elif failure_type == "PLAN":
        relevant = bool(re.search(r'approach|strategy|instead|alternative|different', response, re.I))

    score = sum([mentions_file or mentions_path, actionable, relevant])

    return {
        "score": score,  # 0-3
        "mentions_correct_file": mentions_file or mentions_path,
        "actionable": actionable,
        "relevant": relevant,
    }


def main():
    print("=" * 60)
    print("Phase 2: Scaffolding Pilot (Critical Experiment)")
    print(f"Model: {MODEL}, Budget: {MAX_CALLS} calls")
    print("=" * 60)

    # Load Phase 0 annotations
    annot_path = PROJECT_ROOT / "results" / "phase0_annotations" / "phase0_v2_annotations.json"
    with open(annot_path) as f:
        phase0 = json.load(f)
    annotations = phase0['annotations']

    # Load gold patches
    from datasets import load_dataset
    swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold_map = {}
    for inst in swe:
        files = re.findall(r'--- a/(.*?)\n', inst['patch'])
        gold_map[inst['instance_id']] = {'files': set(files), 'patch': inst['patch'][:500]}

    # Load trajectories
    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')
    traj_map = {r['issue_name']: r['full_conversation_jsonl'] for r in ds}

    # Select 10 instances per type
    SAMPLES_PER_TYPE = 10
    selected = {}
    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        typed = [a for a in annotations if a['failure_type'] == ftype]
        selected[ftype] = typed[:SAMPLES_PER_TYPE]
        print(f"  {ftype}: {len(selected[ftype])} instances selected")

    # Load scaffold prompts
    scaffolds = {}
    for ftype, sname in STRATEGIES.items():
        scaffolds[ftype] = load_scaffold(sname)
    control_text = load_scaffold(CONTROL)

    print(f"\nRunning experiment...")
    results = []
    call_count = 0

    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        print(f"\n--- {ftype} ({len(selected[ftype])} instances) ---")

        for inst in selected[ftype]:
            iid = inst['instance_id']
            if iid not in gold_map or iid not in traj_map:
                continue

            traj_context = extract_trajectory_context(traj_map[iid])
            gold_info = gold_map[iid]

            # Condition 1: With scaffold
            if call_count >= MAX_CALLS:
                print("  BUDGET CAP REACHED")
                break
            scaffold_prompt = build_scaffold_prompt(iid, traj_context, ftype, scaffolds[ftype], gold_info)
            scaffold_response = call_llm(scaffold_prompt)
            scaffold_eval = evaluate_response(scaffold_response, gold_info, ftype)
            call_count += 1

            # Condition 2: Control (no scaffold)
            if call_count >= MAX_CALLS:
                print("  BUDGET CAP REACHED")
                break
            control_prompt = build_scaffold_prompt(iid, traj_context, ftype, control_text, gold_info)
            control_response = call_llm(control_prompt)
            control_eval = evaluate_response(control_response, gold_info, ftype)
            call_count += 1

            results.append({
                "instance_id": iid,
                "failure_type": ftype,
                "scaffold_score": scaffold_eval["score"],
                "control_score": control_eval["score"],
                "scaffold_eval": scaffold_eval,
                "control_eval": control_eval,
                "scaffold_response": scaffold_response[:300],
                "control_response": control_response[:300],
            })

            print(f"  {iid}: scaffold={scaffold_eval['score']}, control={control_eval['score']}")

        if call_count >= MAX_CALLS:
            break

    # Save results
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "total_calls": call_count,
            "samples_per_type": SAMPLES_PER_TYPE,
        },
        "results": results,
    }

    # Compute summary
    summary = {}
    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        typed = [r for r in results if r["failure_type"] == ftype]
        if not typed:
            continue
        scaffold_avg = sum(r["scaffold_score"] for r in typed) / len(typed)
        control_avg = sum(r["control_score"] for r in typed) / len(typed)
        scaffold_file = sum(1 for r in typed if r["scaffold_eval"]["mentions_correct_file"]) / len(typed)
        control_file = sum(1 for r in typed if r["control_eval"]["mentions_correct_file"]) / len(typed)

        summary[ftype] = {
            "n": len(typed),
            "scaffold_avg_score": round(scaffold_avg, 2),
            "control_avg_score": round(control_avg, 2),
            "delta": round(scaffold_avg - control_avg, 2),
            "scaffold_file_rate": round(scaffold_file * 100, 1),
            "control_file_rate": round(control_file * 100, 1),
        }

    output["summary"] = summary

    out_path = RESULTS_DIR / "pilot_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Type':<8} {'N':<4} {'Scaffold':<10} {'Control':<10} {'Delta':<8} {'File%(S)':<10} {'File%(C)'}")
    print("-" * 60)
    types_improved = 0
    for ftype, s in summary.items():
        improved = "✓" if s["delta"] > 0 else "✗"
        print(f"{ftype:<8} {s['n']:<4} {s['scaffold_avg_score']:<10} {s['control_avg_score']:<10} {s['delta']:+.2f}    {s['scaffold_file_rate']:<10} {s['control_file_rate']}")
        if s["delta"] > 0.3:  # meaningful improvement (>10% on 0-3 scale)
            types_improved += 1

    print(f"\n{'=' * 60}")
    print(f"Go/No-Go: scaffold > control on {types_improved}/4 types (need ≥2)")
    if types_improved >= 2:
        print("🟢 GATE PASSED — scaffolding generalizes! Proceed to Phase 3.")
    else:
        print("🔴 GATE FAILED — scaffolding does not generalize. Reconsider approach.")

    print(f"\nTotal API calls: {call_count} (~¥{call_count * 0.1:.0f})")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
