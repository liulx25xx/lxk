#!/usr/bin/env python3
"""
Phase 5 Supplementary Experiments for EMNLP paper.
Three experiments run sequentially:
  1. DeepSeek V4 Cross-Family Validation (best strategy + control, 15 per type)
  2. Full-Sample Scaffold (all 143 instances, gpt-4o-mini)
  3. GPT-5.5 Ceiling Analysis (5 per type, reasoning model)

Uses 2 threads, 3.5s delay, incremental saves.
"""

import json, os, re, sys, time, threading
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent

API_KEY = "<REDACTED_SECRET>"
BASE_URL = "<REDACTED_URL>"
DELAY = 3.5
MAX_CALLS = 500

# Best strategy per failure type (from Phase 4 results)
BEST_STRATEGY = {
    "LOC": "LOC_B_reread_issue",
    "EDIT": "EDIT_A_reread_file",
    "LOGIC": "LOGIC_B_minimal_fix",
    "PLAN": "PLAN_A_step_back",
}
CONTROL = "CONTROL_no_scaffold"

lock = threading.Lock()
total_calls = [0]


def load_scaffold(name):
    path = PROJECT_ROOT / "prompts" / "scaffolding" / f"{name}.txt"
    return path.read_text().strip() if path.exists() else f"[MISSING: {name}]"


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


def call_llm(prompt, model="gpt-4o-mini", is_reasoning=False, retry=3):
    """Call LLM with appropriate parameters based on model type."""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    time.sleep(DELAY)

    for attempt in range(retry):
        try:
            if is_reasoning:
                # Reasoning model: no temperature, use max_completion_tokens
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=4096,
                )
            else:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0,
                )
            with lock:
                total_calls[0] += 1
            return resp.choices[0].message.content or ""
        except Exception as e:
            if "429" in str(e):
                time.sleep(8 * (attempt + 1))
            else:
                if attempt >= 1:
                    return ""
                time.sleep(3)
    return ""


def build_prompt(instance_id, ftype, ctx, gold_files, scaffold_text):
    return f"""Expert engineer helping a failed code agent. Issue: {instance_id}, Failure: {ftype}

Agent trajectory:
{ctx[:2000]}

Recovery guidance:
{scaffold_text}

Files to fix: {', '.join(gold_files)}

Provide the SINGLE BEST next action (bash command or file edit). Be specific. No explanation."""


def run_experiment_batch(instances, strategies, model, is_reasoning, traj_map, gold_map, result_list, exp_label):
    """Run a batch of (instance, strategy) pairs with 2 threads."""
    tasks = []
    for inst in instances:
        iid = inst['instance_id']
        ftype = inst['failure_type']
        if iid not in gold_map or iid not in traj_map:
            continue
        for strat in strategies:
            tasks.append((inst, strat))

    def process_task(task):
        inst, strat_name = task
        if total_calls[0] >= MAX_CALLS:
            return None
        iid = inst['instance_id']
        ftype = inst['failure_type']
        ctx = extract_context(traj_map[iid])
        gold_files = gold_map[iid]
        scaffold_text = load_scaffold(strat_name)
        prompt = build_prompt(iid, ftype, ctx, gold_files, scaffold_text)
        response = call_llm(prompt, model=model, is_reasoning=is_reasoning)
        ev = evaluate(response, gold_files, ftype)
        entry = {
            "instance_id": iid,
            "failure_type": ftype,
            "strategy": strat_name,
            "model": model,
            "experiment": exp_label,
            "eval": ev,
            "response": response[:200],
        }
        with lock:
            result_list.append(entry)
        return entry

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(process_task, t) for t in tasks]
        for f in as_completed(futures):
            f.result()


def save_results(result_list, output_path, model, exp_label):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "experiment": exp_label,
                "total_calls": len(result_list),
            },
            "results": result_list,
        }, f, indent=2)


def print_summary(result_list, exp_label):
    print(f"\n{'='*60}")
    print(f"Summary: {exp_label}")
    print(f"{'='*60}")
    by_type = defaultdict(list)
    for r in result_list:
        by_type[r['failure_type']].append(r)

    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        typed = by_type.get(ftype, [])
        if not typed:
            continue
        print(f"\n--- {ftype} (n={len(typed)}) ---")
        strats = sorted(set(r['strategy'] for r in typed))
        ctrl_score = 0
        best_name, best_score = "", 0
        for s in strats:
            sr = [r for r in typed if r['strategy'] == s]
            avg = sum(r['eval']['score'] for r in sr) / len(sr) if sr else 0
            fhit = sum(1 for r in sr if r['eval']['file_hit']) / len(sr) if sr else 0
            label = s.split("_", 2)[-1] if "_" in s else s
            print(f"  {label:<22s}: {avg:.2f}  file_hit={fhit:.0%}  (n={len(sr)})")
            if s == CONTROL:
                ctrl_score = avg
            elif avg > best_score:
                best_score, best_name = avg, label
        if best_name:
            delta = best_score - ctrl_score
            print(f"  → Best: {best_name} (Δ={delta:+.2f} vs control)")

    overall_scores = [r['eval']['score'] for r in result_list]
    if overall_scores:
        print(f"\nOverall avg score: {sum(overall_scores)/len(overall_scores):.2f} (n={len(overall_scores)})")


def main():
    print("=" * 70)
    print("Phase 5: Supplementary Experiments")
    print(f"API delay: {DELAY}s, Max calls: {MAX_CALLS}, Threads: 2")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # ─── Load shared data ───────────────────────────────────────────────
    print("\n[Loading data...]")
    with open(PROJECT_ROOT / "results" / "phase0_annotations" / "phase0_v2_annotations.json") as f:
        annotations = json.load(f)['annotations']

    from datasets import load_dataset
    swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold_map = {i['instance_id']: set(re.findall(r'--- a/(.*?)\n', i['patch'])) for i in swe}

    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')
    traj_map = {r['issue_name']: r['full_conversation_jsonl'] for r in ds}

    # Group annotations by type
    by_type = defaultdict(list)
    for a in annotations:
        by_type[a['failure_type']].append(a)

    print(f"  Annotations: {len(annotations)} total")
    for k, v in sorted(by_type.items()):
        print(f"    {k}: {len(v)}")
    print(f"  Gold map: {len(gold_map)} entries")
    print(f"  Traj map: {len(traj_map)} entries")

    start_time = time.time()

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: DeepSeek V4 Cross-Family Validation
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("EXPERIMENT 1: DeepSeek V4 Cross-Family Validation")
    print(f"Model: deepseek-v4-pro, 15 instances/type (8 for PLAN)")
    print(f"{'═'*70}")

    exp1_results = []
    exp1_samples = {
        "LOC": by_type["LOC"][:15],
        "EDIT": by_type["EDIT"][:15],
        "LOGIC": by_type["LOGIC"][:15],
        "PLAN": by_type["PLAN"][:8],
    }

    exp1_instances = []
    for ftype, insts in exp1_samples.items():
        for inst in insts:
            inst_copy = dict(inst)
            exp1_instances.append(inst_copy)

    # Each instance gets best strategy + control
    tasks_by_type = defaultdict(list)
    for inst in exp1_instances:
        tasks_by_type[inst['failure_type']].append(inst)

    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        if total_calls[0] >= MAX_CALLS:
            break
        strategies = [BEST_STRATEGY[ftype], CONTROL]
        typed_insts = tasks_by_type[ftype]
        run_experiment_batch(typed_insts, strategies, "deepseek-v4-pro", False,
                            traj_map, gold_map, exp1_results, "deepseek_validation")
        # Incremental save
        save_results(exp1_results,
                     PROJECT_ROOT / "results" / "phase5_deepseek_validation" / "results.json",
                     "deepseek-v4-pro", "DeepSeek V4 Cross-Family Validation")

    save_results(exp1_results,
                 PROJECT_ROOT / "results" / "phase5_deepseek_validation" / "results.json",
                 "deepseek-v4-pro", "DeepSeek V4 Cross-Family Validation")
    print_summary(exp1_results, "Experiment 1: DeepSeek V4 Cross-Family Validation")
    print(f"\n  Total API calls so far: {total_calls[0]}")

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Full-Sample Scaffold (all 143 instances)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("EXPERIMENT 2: Full-Sample Scaffold (all 143 instances)")
    print(f"Model: gpt-4o-mini, best strategy + control per type")
    print(f"{'═'*70}")

    exp2_results = []

    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        if total_calls[0] >= MAX_CALLS:
            print(f"  [BUDGET LIMIT] Stopping at {total_calls[0]} calls")
            break
        strategies = [BEST_STRATEGY[ftype], CONTROL]
        typed_insts = by_type[ftype]  # ALL instances of this type
        print(f"  Running {ftype}: {len(typed_insts)} instances × 2 strategies = {len(typed_insts)*2} calls")
        run_experiment_batch(typed_insts, strategies, "gpt-4o-mini", False,
                            traj_map, gold_map, exp2_results, "full_sample")
        # Incremental save
        save_results(exp2_results,
                     PROJECT_ROOT / "results" / "phase5_full_sample" / "results.json",
                     "gpt-4o-mini", "Full-Sample Scaffold (143 instances)")

    save_results(exp2_results,
                 PROJECT_ROOT / "results" / "phase5_full_sample" / "results.json",
                 "gpt-4o-mini", "Full-Sample Scaffold (143 instances)")
    print_summary(exp2_results, "Experiment 2: Full-Sample Scaffold")
    print(f"\n  Total API calls so far: {total_calls[0]}")

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: GPT-5.5 Ceiling Analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("EXPERIMENT 3: GPT-5.5 Ceiling Analysis")
    print(f"Model: gpt-5.5 (reasoning), 5 instances/type, best strategy + control")
    print(f"{'═'*70}")

    exp3_results = []
    exp3_samples = {
        "LOC": by_type["LOC"][:5],
        "EDIT": by_type["EDIT"][:5],
        "LOGIC": by_type["LOGIC"][:5],
        "PLAN": by_type["PLAN"][:5],
    }

    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        if total_calls[0] >= MAX_CALLS:
            print(f"  [BUDGET LIMIT] Stopping at {total_calls[0]} calls")
            break
        strategies = [BEST_STRATEGY[ftype], CONTROL]
        typed_insts = exp3_samples[ftype]
        print(f"  Running {ftype}: {len(typed_insts)} instances × 2 strategies")
        run_experiment_batch(typed_insts, strategies, "gpt-5.5", True,
                            traj_map, gold_map, exp3_results, "gpt55_ceiling")
        # Incremental save
        save_results(exp3_results,
                     PROJECT_ROOT / "results" / "phase5_gpt55_ceiling" / "results.json",
                     "gpt-5.5", "GPT-5.5 Ceiling Analysis")

    save_results(exp3_results,
                 PROJECT_ROOT / "results" / "phase5_gpt55_ceiling" / "results.json",
                 "gpt-5.5", "GPT-5.5 Ceiling Analysis")
    print_summary(exp3_results, "Experiment 3: GPT-5.5 Ceiling Analysis")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    print(f"\n{'═'*70}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"{'═'*70}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Total API calls: {total_calls[0]}")
    print(f"  Exp 1 (DeepSeek V4): {len(exp1_results)} results")
    print(f"  Exp 2 (Full Sample): {len(exp2_results)} results")
    print(f"  Exp 3 (GPT-5.5):     {len(exp3_results)} results")
    print(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
