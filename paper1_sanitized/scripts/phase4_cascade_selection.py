#!/usr/bin/env python3
"""
Phase 4 Experiment 2: Cascade-Aware Scaffold Selection
-------------------------------------------------------
Test whether failure-type-specific scaffolding (automatically selected
based on Phase 0 annotations) outperforms a one-size-fits-all approach.

Conditions (× 96 instances = 288 calls on gpt-4o-mini):
  1. "Oracle selector": Phase 0 annotation → best strategy per type
     LOC→reread_issue, EDIT→reread_file, LOGIC→minimal_fix, PLAN→step_back
  2. "Fixed scaffold": EDIT_A (reread_file) for ALL instances regardless of type
  3. "Control": No scaffold (CONTROL_no_scaffold)
"""

import json, os, re, sys, time, threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase4_cascade_selection"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENAI_API_KEY", "<REDACTED_SECRET>")
BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4o-mini"
DELAY = 3.5
MAX_WORKERS = 2

# Oracle selector: best scaffold per failure type
ORACLE_MAP = {
    "LOC":   "LOC_B_reread_issue",
    "EDIT":  "EDIT_A_reread_file",
    "LOGIC": "LOGIC_B_minimal_fix",
    "PLAN":  "PLAN_A_step_back",
}
FIXED_SCAFFOLD = "EDIT_A_reread_file"   # one-size-fits-all
CONTROL = "CONTROL_no_scaffold"

# All 96 instances
MAX_SAMPLES = {"LOC": 30, "EDIT": 28, "LOGIC": 30, "PLAN": 8}

# Three conditions
CONDITIONS = ["oracle", "fixed", "control"]

lock = threading.Lock()
all_results = []
call_count = [0]


def load_scaffold(name):
    path = PROJECT_ROOT / "prompts" / "scaffolding" / f"{name}.txt"
    return path.read_text().strip() if path.exists() else f"[MISSING: {name}]"


def call_llm(prompt, retry=3):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    time.sleep(DELAY)
    for attempt in range(retry):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024, temperature=0,
            )
            with lock:
                call_count[0] += 1
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e):
                time.sleep(8 * (attempt + 1))
            else:
                if attempt >= 1:
                    return ""
                time.sleep(3)
    return ""


def extract_context(conv_json, max_turns=4):
    try:
        data = json.loads(conv_json)
    except:
        return ""
    parts = []
    for turn in data[:max_turns]:
        resp = turn.get('response', {})
        choices = resp.get('choices', [])
        if choices and isinstance(choices[0], dict):
            c = choices[0].get('message', {}).get('content', '')
            if c: parts.append(f"[Agent]: {c[:350]}")
        msgs = turn.get('messages', [])
        um = [m for m in msgs if m.get('role') == 'user']
        if um:
            obs = um[-1].get('content', '')
            if isinstance(obs, list):
                obs = ' '.join(x.get('text','') for x in obs if isinstance(x,dict))
            if obs and len(obs) > 30:
                parts.append(f"[Obs]: {obs[:250]}")
    return '\n'.join(parts)


def evaluate(response, gold_files, ftype):
    if not response:
        return {"score": 0, "file_hit": False, "actionable": False, "relevant": False}
    file_hit = any(gf.split('/')[-1] in response for gf in gold_files)
    actionable = bool(re.search(r'(cat |grep |find |python |pytest |str_replace|def |import |sed )', response))
    relevant = False
    if ftype == "LOC":
        relevant = bool(re.search(r'search|find|grep|locate|other.*file|different', response, re.I))
    elif ftype == "EDIT":
        relevant = bool(re.search(r'read|cat|view|exact|character|whitespace', response, re.I))
    elif ftype == "LOGIC":
        relevant = bool(re.search(r'test|assert|expect|return|logic|fix|edge', response, re.I))
    elif ftype == "PLAN":
        relevant = bool(re.search(r'approach|strategy|instead|alternative|reconsider', response, re.I))
    return {"score": sum([file_hit, actionable, relevant]), "file_hit": file_hit,
            "actionable": actionable, "relevant": relevant}


def _save():
    out = {
        "metadata": {
            "experiment": "phase4_cascade_selection",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "calls": call_count[0],
            "conditions": CONDITIONS,
            "description": "Oracle (type-specific) vs Fixed (one-size-fits-all) vs Control",
        },
        "results": list(all_results),
    }
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(out, f, indent=2)


def get_scaffold_for_condition(condition, ftype):
    """Return (strategy_name, scaffold_text) for the given condition and failure type."""
    if condition == "oracle":
        strat = ORACLE_MAP[ftype]
        return strat, load_scaffold(strat)
    elif condition == "fixed":
        return FIXED_SCAFFOLD, load_scaffold(FIXED_SCAFFOLD)
    else:  # control
        return CONTROL, load_scaffold(CONTROL)


def run_condition(condition, all_annots, traj_map, gold_map):
    """Run one condition across all instances."""
    scores_by_type = defaultdict(list)

    for inst in all_annots:
        iid = inst['instance_id']
        ftype = inst['failure_type']
        if iid not in gold_map or iid not in traj_map:
            continue

        strat_name, scaffold_text = get_scaffold_for_condition(condition, ftype)
        ctx = extract_context(traj_map[iid])
        gold_files = gold_map[iid]

        prompt = f"""Expert engineer helping a failed code agent. Issue: {iid}, Failure: {ftype}

Agent trajectory:
{ctx[:2000]}

Recovery guidance:
{scaffold_text}

Files to fix: {', '.join(gold_files)}

Provide the SINGLE BEST next action (bash command or file edit). Be specific. No explanation."""

        response = call_llm(prompt)
        ev = evaluate(response, gold_files, ftype)
        scores_by_type[ftype].append(ev["score"])

        entry = {
            "instance_id": iid,
            "failure_type": ftype,
            "condition": condition,
            "strategy": strat_name,
            "eval": ev,
            "response": response[:150],
        }
        with lock:
            all_results.append(entry)
            if len(all_results) % 10 == 0:
                _save()
                print(f"  [checkpoint] {len(all_results)} results saved, {call_count[0]} calls")

    # Print per-type summary for this condition
    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        sc = scores_by_type.get(ftype, [])
        if sc:
            avg = sum(sc) / len(sc)
            print(f"  [{condition:>7s}] {ftype}: avg={avg:.2f} (n={len(sc)})")


def main():
    print("=" * 60)
    print("Phase 4 Exp2: Cascade-Aware Scaffold Selection")
    print(f"Model: {MODEL}, Delay: {DELAY}s, Workers: {MAX_WORKERS}")
    print(f"Conditions: oracle / fixed / control")
    print("=" * 60)

    # Load annotations
    with open(PROJECT_ROOT / "results" / "phase0_annotations" / "phase0_v2_annotations.json") as f:
        annotations = json.load(f)['annotations']

    # Load gold patches
    from datasets import load_dataset
    print("Loading SWE-bench gold patches...")
    swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold_map = {i['instance_id']: set(re.findall(r'--- a/(.*?)\n', i['patch'])) for i in swe}

    # Load trajectories
    print("Loading trajectory data...")
    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')
    traj_map = {r['issue_name']: r['full_conversation_jsonl'] for r in ds}

    # Group by type and cap, then flatten
    by_type = defaultdict(list)
    for a in annotations:
        by_type[a['failure_type']].append(a)

    all_annots = []
    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        capped = by_type[ftype][:MAX_SAMPLES.get(ftype, 30)]
        all_annots.extend(capped)
        print(f"  {ftype}: {len(capped)} instances")

    total_expected = len(all_annots) * len(CONDITIONS)
    print(f"\nTotal instances: {len(all_annots)}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Expected calls: ~{total_expected}")
    print(f"Estimated time: ~{total_expected * DELAY / 60 / MAX_WORKERS:.1f} min")
    print()

    start = time.time()

    # Run conditions with 2 threads — "oracle" and "fixed" in parallel first,
    # then "control" alongside any remaining work
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {}
        for cond in CONDITIONS:
            fut = pool.submit(run_condition, cond, all_annots, traj_map, gold_map)
            futures[fut] = cond
        for f in as_completed(futures):
            cond = futures[f]
            try:
                f.result()
                print(f"\n  ✓ Condition '{cond}' done.")
            except Exception as e:
                print(f"\n  ✗ Condition '{cond}' failed: {e}")

    _save()
    elapsed = time.time() - start

    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print(f"PHASE 4 EXP2 COMPLETE — {elapsed/60:.1f} min, {call_count[0]} API calls")
    print(f"Model: {MODEL}")
    print(f"{'='*60}")

    # Per-type breakdown
    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        typed = [r for r in all_results if r["failure_type"] == ftype]
        if not typed:
            continue
        print(f"\n--- {ftype} ---")
        for cond in CONDITIONS:
            cr = [r for r in typed if r["condition"] == cond]
            if not cr:
                continue
            avg = sum(r["eval"]["score"] for r in cr) / len(cr)
            fhit = sum(1 for r in cr if r["eval"]["file_hit"]) / len(cr)
            print(f"  {cond:<10s}: {avg:.2f}  file_hit={fhit:.0%}  (n={len(cr)})")

    # Overall condition comparison
    print(f"\n{'='*60}")
    print(f"OVERALL CONDITION COMPARISON:")
    for cond in CONDITIONS:
        cr = [r for r in all_results if r["condition"] == cond]
        if not cr:
            continue
        avg = sum(r["eval"]["score"] for r in cr) / len(cr)
        fhit = sum(1 for r in cr if r["eval"]["file_hit"]) / len(cr)
        act = sum(1 for r in cr if r["eval"]["actionable"]) / len(cr)
        rel = sum(1 for r in cr if r["eval"]["relevant"]) / len(cr)
        print(f"  {cond:<10s}: avg={avg:.2f}  file_hit={fhit:.0%}  actionable={act:.0%}  relevant={rel:.0%}  (n={len(cr)})")

    # Delta: oracle vs fixed, oracle vs control
    oracle_r = [r for r in all_results if r["condition"] == "oracle"]
    fixed_r = [r for r in all_results if r["condition"] == "fixed"]
    control_r = [r for r in all_results if r["condition"] == "control"]
    if oracle_r and fixed_r and control_r:
        avg_o = sum(r["eval"]["score"] for r in oracle_r) / len(oracle_r)
        avg_f = sum(r["eval"]["score"] for r in fixed_r) / len(fixed_r)
        avg_c = sum(r["eval"]["score"] for r in control_r) / len(control_r)
        print(f"\n  Oracle vs Fixed:   {avg_o - avg_f:+.2f}")
        print(f"  Oracle vs Control: {avg_o - avg_c:+.2f}")
        print(f"  Fixed  vs Control: {avg_f - avg_c:+.2f}")

    print(f"{'='*60}")
    print(f"\nResults saved to: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
