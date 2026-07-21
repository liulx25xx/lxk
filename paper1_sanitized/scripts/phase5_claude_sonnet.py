#!/usr/bin/env python3
"""
Phase 5: Claude Sonnet 4.6 Validation
Addresses "outdated models" criticism by running best scaffold per type + control
on a May 2026 frontier model (claude-sonnet-4-6).

Design (same as Phase 3 best-strategy subset):
  LOC:   LOC_B_reread_issue   + CONTROL  (30 instances)
  EDIT:  EDIT_A_reread_file   + CONTROL  (28 instances)
  LOGIC: LOGIC_B_minimal_fix  + CONTROL  (30 instances)
  PLAN:  PLAN_A_step_back     + CONTROL  (8 instances)
  Total: ~192 calls

Model: claude-sonnet-4-6 (NOT a reasoning model)
       max_tokens=1024, temperature=0
Delay: 3.5s per call
"""

import json, os, re, sys, time, threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5_claude_sonnet"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = "<REDACTED_SECRET>"
BASE_URL = "<REDACTED_URL>"
MODEL = "claude-sonnet-4-6"
DELAY = 3.5
MAX_CALLS = 300

# Best strategy per type (from Phase 3 results) + control
TYPE_BEST = {
    "LOC":   "LOC_B_reread_issue",
    "EDIT":  "EDIT_A_reread_file",
    "LOGIC": "LOGIC_B_minimal_fix",
    "PLAN":  "PLAN_A_step_back",
}
CONTROL = "CONTROL_no_scaffold"
MAX_SAMPLES = {"LOC": 30, "EDIT": 28, "LOGIC": 30, "PLAN": 8}

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
                max_tokens=1024,
                temperature=0,
            )
            with lock:
                call_count[0] += 1
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [WARN] attempt {attempt+1} failed: {str(e)[:120]}")
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(10 * (attempt + 1))
            else:
                if attempt >= 1:
                    return ""
                time.sleep(5)
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
            if c:
                parts.append(f"[Agent]: {c[:350]}")
        msgs = turn.get('messages', [])
        um = [m for m in msgs if m.get('role') == 'user']
        if um:
            obs = um[-1].get('content', '')
            if isinstance(obs, list):
                obs = ' '.join(x.get('text', '') for x in obs if isinstance(x, dict))
            if obs and len(obs) > 30:
                parts.append(f"[Obs]: {obs[:250]}")
    return '\n'.join(parts)


def evaluate(response, gold_files, ftype):
    if not response:
        return {"score": 0, "file_hit": False, "actionable": False, "relevant": False}
    file_hit = any(gf.split('/')[-1] in response for gf in gold_files)
    actionable = bool(re.search(
        r'(cat |grep |find |python |pytest |str_replace|def |import |sed )', response))
    relevant = False
    if ftype == "LOC":
        relevant = bool(re.search(r'search|find|grep|locate|other.*file|different', response, re.I))
    elif ftype == "EDIT":
        relevant = bool(re.search(r'read|cat|view|exact|character|whitespace', response, re.I))
    elif ftype == "LOGIC":
        relevant = bool(re.search(r'test|assert|expect|return|logic|fix|edge', response, re.I))
    elif ftype == "PLAN":
        relevant = bool(re.search(r'approach|strategy|instead|alternative|reconsider', response, re.I))
    return {
        "score": sum([file_hit, actionable, relevant]),
        "file_hit": file_hit,
        "actionable": actionable,
        "relevant": relevant,
    }


def run_type(ftype, typed_annots, traj_map, gold_map):
    """Run best strategy + control for one failure type."""
    strategies = [TYPE_BEST[ftype], CONTROL]
    results = []

    for strat_name in strategies:
        scaffold_text = load_scaffold(strat_name)
        scores = []

        for inst in typed_annots:
            if call_count[0] >= MAX_CALLS:
                break
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

        label = strat_name.split("_", 2)[-1] if "_" in strat_name else strat_name
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  [{ftype}] {label:<22s}: {avg:.2f} (n={len(scores)})")

    return results


def _save():
    with open(RESULTS_DIR / "results.json", 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": MODEL,
                "calls": call_count[0],
                "description": "Phase 5: Claude Sonnet 4.6 validation (best scaffold + control)",
            },
            "results": list(all_results),
        }, f, indent=2)


def main():
    print("=" * 60)
    print("Phase 5: Claude Sonnet 4.6 Validation")
    print(f"Model: {MODEL}, Delay: {DELAY}s, Threads: 2")
    print(f"Best strategies: {list(TYPE_BEST.values())}")
    print("=" * 60)

    # Load data
    with open(PROJECT_ROOT / "results" / "phase0_annotations" / "phase0_v2_annotations.json") as f:
        annotations = json.load(f)['annotations']

    from datasets import load_dataset
    swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold_map = {i['instance_id']: set(re.findall(r'--- a/(.*?)\n', i['patch'])) for i in swe}

    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')
    traj_map = {r['issue_name']: r['full_conversation_jsonl'] for r in ds}

    # Group by type
    by_type = defaultdict(list)
    for a in annotations:
        by_type[a['failure_type']].append(a)
    for k in by_type:
        by_type[k] = by_type[k][:MAX_SAMPLES.get(k, 30)]

    start = time.time()

    # Run 2 types in parallel
    type_order = ["LOC", "EDIT", "LOGIC", "PLAN"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for ftype in type_order:
            futures.append(pool.submit(run_type, ftype, by_type[ftype], traj_map, gold_map))
        for f in as_completed(futures):
            f.result()

    _save()
    elapsed = time.time() - start

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} min, {call_count[0]} calls")
    print(f"Model: {MODEL}")
    print(f"{'='*60}")

    for ftype in type_order:
        typed = [r for r in all_results if r["failure_type"] == ftype]
        if not typed:
            continue
        print(f"\n--- {ftype} ---")
        strats = sorted(set(r["strategy"] for r in typed))
        ctrl_score = 0
        scaffold_name, scaffold_score = "", 0
        for s in strats:
            sr = [r for r in typed if r["strategy"] == s]
            avg = sum(r["eval"]["score"] for r in sr) / len(sr)
            fhit = sum(1 for r in sr if r["eval"]["file_hit"]) / len(sr)
            act  = sum(1 for r in sr if r["eval"]["actionable"]) / len(sr)
            rel  = sum(1 for r in sr if r["eval"]["relevant"]) / len(sr)
            label = s.split("_", 2)[-1] if "_" in s else s
            if s == CONTROL:
                ctrl_score = avg
            else:
                scaffold_score = avg
                scaffold_name = label
            print(f"  {label:<22s}: {avg:.2f}  file={fhit:.0%}  act={act:.0%}  rel={rel:.0%}  (n={len(sr)})")
        if scaffold_name:
            delta = scaffold_score - ctrl_score
            print(f"  → Δ = {delta:+.2f} (scaffold vs control)")

    # Cross-model comparison hint
    print(f"\n{'='*60}")
    print("Compare with Phase 3 (gpt-4o-mini) to show model-agnostic scaffold benefit.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
