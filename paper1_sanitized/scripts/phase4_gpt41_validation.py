#!/usr/bin/env python3
"""
Phase 4 Experiment 1: GPT-4.1 Cross-Model Validation
-----------------------------------------------------
Verify that Phase 3 best scaffolding results hold on a stronger model (gpt-4.1).

Design:
  - LOC:   reread_issue  + control  (15 instances)
  - EDIT:  reread_file   + control  (15 instances)
  - LOGIC: minimal_fix   + control  (15 instances)
  - PLAN:  step_back     + control  (8 instances)
  Total: ~106 calls on gpt-4.1
"""

import json, os, re, sys, time, threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase4_gpt41_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENAI_API_KEY", "<REDACTED_SECRET>")
BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4.1"
DELAY = 3.5
MAX_WORKERS = 2

# Best strategy per type (from Phase 3) + control
BEST_STRATEGIES = {
    "LOC":   "LOC_B_reread_issue",
    "EDIT":  "EDIT_A_reread_file",
    "LOGIC": "LOGIC_B_minimal_fix",
    "PLAN":  "PLAN_A_step_back",
}
CONTROL = "CONTROL_no_scaffold"

# Sample sizes for gpt-4.1 (expensive model)
MAX_SAMPLES = {"LOC": 15, "EDIT": 15, "LOGIC": 15, "PLAN": 8}

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
            "experiment": "phase4_gpt41_validation",
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "calls": call_count[0],
            "description": "Cross-model validation of best Phase 3 strategies on gpt-4.1",
        },
        "results": list(all_results),
    }
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump(out, f, indent=2)


def run_type(ftype, typed_annots, traj_map, gold_map):
    """Run best strategy + control for one failure type on gpt-4.1."""
    best_strat = BEST_STRATEGIES[ftype]
    strategies = [best_strat, CONTROL]

    for strat_name in strategies:
        scaffold_text = load_scaffold(strat_name)
        scores = []

        for inst in typed_annots:
            iid = inst['instance_id']
            if iid not in gold_map or iid not in traj_map:
                continue

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
            scores.append(ev["score"])

            entry = {
                "instance_id": iid,
                "failure_type": ftype,
                "strategy": strat_name,
                "eval": ev,
                "response": response[:200],
            }
            with lock:
                all_results.append(entry)
                if len(all_results) % 10 == 0:
                    _save()
                    print(f"  [checkpoint] {len(all_results)} results saved, {call_count[0]} calls")

        label = strat_name.split("_", 2)[-1] if "_" in strat_name else strat_name
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  [{ftype}] {label:<22s}: {avg:.2f} (n={len(scores)})")


def main():
    print("=" * 60)
    print("Phase 4 Exp1: GPT-4.1 Cross-Model Validation")
    print(f"Model: {MODEL}, Delay: {DELAY}s, Workers: {MAX_WORKERS}")
    print(f"Strategies: best-per-type + control")
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

    # Group by type and cap samples
    by_type = defaultdict(list)
    for a in annotations:
        by_type[a['failure_type']].append(a)
    for k in by_type:
        by_type[k] = by_type[k][:MAX_SAMPLES.get(k, 15)]

    total_expected = sum(2 * len(by_type[t]) for t in by_type)  # 2 conditions per instance
    print(f"\nExpected calls: ~{total_expected}")
    print(f"Estimated time: ~{total_expected * DELAY / 60 / MAX_WORKERS:.1f} min")
    print()

    start = time.time()

    # Run types in parallel (2 at a time)
    type_order = ["LOC", "EDIT", "LOGIC", "PLAN"]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = []
        for ftype in type_order:
            if ftype in by_type and by_type[ftype]:
                futures.append(pool.submit(run_type, ftype, by_type[ftype], traj_map, gold_map))
        for f in as_completed(futures):
            f.result()

    _save()
    elapsed = time.time() - start

    # ===== SUMMARY =====
    print(f"\n{'='*60}")
    print(f"PHASE 4 EXP1 COMPLETE — {elapsed/60:.1f} min, {call_count[0]} API calls")
    print(f"Model: {MODEL}")
    print(f"{'='*60}")

    for ftype in type_order:
        typed = [r for r in all_results if r["failure_type"] == ftype]
        if not typed:
            continue
        print(f"\n--- {ftype} ---")
        strats = sorted(set(r["strategy"] for r in typed))
        ctrl_score = 0
        best_name, best_score = "", 0
        for s in strats:
            sr = [r for r in typed if r["strategy"] == s]
            avg = sum(r["eval"]["score"] for r in sr) / len(sr)
            fhit = sum(1 for r in sr if r["eval"]["file_hit"]) / len(sr)
            label = s.split("_", 2)[-1] if "_" in s else s
            if s == CONTROL:
                ctrl_score = avg
            else:
                best_score, best_name = avg, label
            print(f"  {label:<22s}: {avg:.2f}  file_hit={fhit:.0%}  (n={len(sr)})")
        if best_name:
            delta = best_score - ctrl_score
            sign = "+" if delta > 0 else ""
            print(f"  → Scaffold vs control: Δ={sign}{delta:.2f}")

    # Overall summary
    scaffold_results = [r for r in all_results if r["strategy"] != CONTROL]
    control_results = [r for r in all_results if r["strategy"] == CONTROL]
    if scaffold_results and control_results:
        avg_s = sum(r["eval"]["score"] for r in scaffold_results) / len(scaffold_results)
        avg_c = sum(r["eval"]["score"] for r in control_results) / len(control_results)
        fhit_s = sum(1 for r in scaffold_results if r["eval"]["file_hit"]) / len(scaffold_results)
        fhit_c = sum(1 for r in control_results if r["eval"]["file_hit"]) / len(control_results)
        print(f"\n{'='*60}")
        print(f"OVERALL (gpt-4.1):")
        print(f"  Scaffold avg: {avg_s:.2f}  file_hit: {fhit_s:.0%}  (n={len(scaffold_results)})")
        print(f"  Control  avg: {avg_c:.2f}  file_hit: {fhit_c:.0%}  (n={len(control_results)})")
        print(f"  Delta:        {avg_s - avg_c:+.2f}")
        print(f"{'='*60}")

    print(f"\nResults saved to: {RESULTS_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
