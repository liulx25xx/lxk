#!/usr/bin/env python3
"""
Phase 3: Full Scaffolding Matrix — Multi-strategy comparison across all types.

Key improvements over Phase 2 pilot:
1. All 3 strategies per type (not just 1 best guess)
2. Full sample size (all annotated instances per type)
3. LOC now tests all 3 strategies to find which works

Design:
- 4 types × 3 strategies + 1 control = 13 conditions
- But we only run all 3 strategies for types where it matters
- LOC: 37 × 4 conditions (3 strategies + control) = 148 calls
- EDIT: 28 × 2 conditions (best from pilot + control) = 56 calls
- LOGIC: 70 × 3 conditions (2 strategies + control) = 210 calls (sample 30)
- Total: ~300 calls, ~¥30

Cost: ~¥30
Output: results/phase3_full_scaffold/
"""

import json, os, re, sys, time
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_full_scaffold"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENAI_API_KEY", "<REDACTED_SECRET>")
BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4o-mini"
INTER_CALL_DELAY = 6.5
MAX_CALLS = 400

# Per-type strategy matrix
TYPE_STRATEGIES = {
    "LOC": ["LOC_A_broaden_search", "LOC_B_reread_issue", "LOC_C_test_guided"],
    "EDIT": ["EDIT_A_reread_file", "EDIT_B_smaller_edit"],  # A is proven best
    "LOGIC": ["LOGIC_A_test_analysis", "LOGIC_B_minimal_fix", "LOGIC_C_edge_cases"],
    "PLAN": ["PLAN_A_step_back", "PLAN_B_scope_check"],
}
CONTROL = "CONTROL_no_scaffold"

# Sample limits per type (balance coverage vs cost)
MAX_SAMPLES = {"LOC": 30, "EDIT": 28, "LOGIC": 30, "PLAN": 8}


def load_scaffold(name):
    path = PROJECT_ROOT / "prompts" / "scaffolding" / f"{name}.txt"
    return path.read_text().strip() if path.exists() else f"[MISSING: {name}]"


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


def extract_trajectory_context(conversation_json, max_turns=5):
    try:
        data = json.loads(conversation_json)
    except json.JSONDecodeError:
        return "Unable to parse trajectory"

    parts = []
    for turn in data[:max_turns]:
        resp = turn.get('response', {})
        choices = resp.get('choices', [])
        if choices and isinstance(choices[0], dict):
            content = choices[0].get('message', {}).get('content', '')
            if content:
                parts.append(f"[Agent]: {content[:400]}")
        msgs = turn.get('messages', [])
        user_msgs = [m for m in msgs if m.get('role') == 'user']
        if user_msgs:
            obs = user_msgs[-1].get('content', '')
            if isinstance(obs, list):
                obs = ' '.join(c.get('text', '') for c in obs if isinstance(c, dict))
            if obs and len(obs) > 50:
                parts.append(f"[Obs]: {obs[:300]}")
    return '\n'.join(parts)


def build_prompt(instance_id, traj_context, failure_type, scaffold_text, gold_files):
    return f"""You are an expert software engineer helping a failed code agent recover.

Issue: {instance_id}
Failure type: {failure_type}

## Agent's trajectory (first few steps):
{traj_context[:2500]}

## Recovery guidance:
{scaffold_text}

## Task:
Based on the guidance, provide the SINGLE BEST next action. Be specific with file paths and commands.
The fix likely involves: {', '.join(gold_files)}

Output ONLY the next action (bash command or file edit). No explanation."""


def evaluate_response(response, gold_files, failure_type):
    if not response:
        return {"score": 0, "file_hit": False, "actionable": False, "relevant": False}

    file_hit = any(gf.split('/')[-1] in response for gf in gold_files)
    actionable = bool(re.search(r'(cat |grep |find |cd |python |pytest |str_replace|def |class |import |sed )', response))

    relevant = False
    if failure_type == "LOC":
        relevant = bool(re.search(r'search|find|grep|locate|look|different.*file|other.*module', response, re.I))
    elif failure_type == "EDIT":
        relevant = bool(re.search(r'read|cat|view|exact|character|whitespace|indent', response, re.I))
    elif failure_type == "LOGIC":
        relevant = bool(re.search(r'test|assert|expect|return|logic|fix|check|edge', response, re.I))
    elif failure_type == "PLAN":
        relevant = bool(re.search(r'approach|strategy|instead|alternative|step back|reconsider', response, re.I))

    return {"score": sum([file_hit, actionable, relevant]), "file_hit": file_hit,
            "actionable": actionable, "relevant": relevant}


def main():
    print("=" * 60)
    print("Phase 3: Full Scaffolding Matrix")
    print(f"Model: {MODEL}, Budget: {MAX_CALLS} calls")
    print("=" * 60)

    # Load data
    with open(PROJECT_ROOT / "results" / "phase0_annotations" / "phase0_v2_annotations.json") as f:
        phase0 = json.load(f)
    annotations = phase0['annotations']

    from datasets import load_dataset
    swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold_map = {inst['instance_id']: set(re.findall(r'--- a/(.*?)\n', inst['patch'])) for inst in swe}

    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')
    traj_map = {r['issue_name']: r['full_conversation_jsonl'] for r in ds}

    # Run per type
    all_results = []
    call_count = 0

    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        typed = [a for a in annotations if a['failure_type'] == ftype][:MAX_SAMPLES[ftype]]
        strategies = TYPE_STRATEGIES[ftype] + [CONTROL]

        print(f"\n{'='*40}")
        print(f"{ftype}: {len(typed)} instances × {len(strategies)} conditions = {len(typed)*len(strategies)} calls")
        print(f"{'='*40}")

        for strat_name in strategies:
            scaffold_text = load_scaffold(strat_name)
            strat_label = strat_name.replace(f"{ftype}_", "").replace("CONTROL_", "")

            scores = []
            for inst in typed:
                if call_count >= MAX_CALLS:
                    print("  ⚠️ BUDGET CAP")
                    break

                iid = inst['instance_id']
                if iid not in gold_map or iid not in traj_map:
                    continue

                traj_ctx = extract_trajectory_context(traj_map[iid])
                prompt = build_prompt(iid, traj_ctx, ftype, scaffold_text, gold_map[iid])
                response = call_llm(prompt)
                ev = evaluate_response(response, gold_map[iid], ftype)
                call_count += 1

                scores.append(ev["score"])
                all_results.append({
                    "instance_id": iid, "failure_type": ftype,
                    "strategy": strat_name, "eval": ev,
                    "response_preview": response[:200],
                })

            if scores:
                avg = sum(scores) / len(scores)
                print(f"  {strat_label:<20s}: avg={avg:.2f} (n={len(scores)})")

            # Incremental save
            if call_count % 50 == 0:
                _save(all_results, call_count)

        if call_count >= MAX_CALLS:
            break

    _save(all_results, call_count)
    _print_summary(all_results, call_count)


def _save(results, call_count):
    output = {
        "metadata": {"timestamp": datetime.now().isoformat(), "model": MODEL, "calls": call_count},
        "results": results,
    }
    with open(RESULTS_DIR / "full_results.json", 'w') as f:
        json.dump(output, f, indent=2)


def _print_summary(results, call_count):
    print(f"\n{'=' * 60}")
    print("FULL RESULTS SUMMARY")
    print(f"{'=' * 60}")

    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        typed = [r for r in results if r["failure_type"] == ftype]
        if not typed:
            continue

        print(f"\n--- {ftype} ---")
        strategies = set(r["strategy"] for r in typed)
        strat_scores = {}
        for s in sorted(strategies):
            s_results = [r for r in typed if r["strategy"] == s]
            avg = sum(r["eval"]["score"] for r in s_results) / len(s_results)
            file_rate = sum(1 for r in s_results if r["eval"]["file_hit"]) / len(s_results)
            strat_scores[s] = avg
            label = s.split("_", 2)[-1] if "_" in s else s
            print(f"  {label:<25s}: score={avg:.2f}, file_hit={file_rate:.0%} (n={len(s_results)})")

        # Best vs control
        control_score = strat_scores.get(CONTROL, 0)
        best_strat = max((s for s in strat_scores if s != CONTROL), key=lambda s: strat_scores[s], default=None)
        if best_strat:
            delta = strat_scores[best_strat] - control_score
            print(f"  → Best: {best_strat.split('_',2)[-1]} (delta={delta:+.2f} vs control)")

    print(f"\nTotal calls: {call_count} (~¥{call_count * 0.1:.0f})")


if __name__ == "__main__":
    main()
