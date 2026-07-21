#!/usr/bin/env python3
"""
Phase 6: Multi-model expansion. Run remaining 4 models to complete the 9-model matrix.
Models: deepseek-v4-flash, qwen3.5-35b-a3b, claude-opus-4-7, o4-mini

Design: Best strategy + control per type, 15 instances per type (8 for PLAN)
Total: 4 models × ~96 calls = ~384 calls
"""

import json, os, re, sys, time, threading
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase6_multimodel"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("OPENAI_API_KEY", "<REDACTED_SECRET>")
BASE_URL = "<REDACTED_URL>"
DELAY = 3.5
MAX_CALLS = 500

MODELS = [
    {"name": "deepseek-v4-flash", "tier": "mid", "reasoning": False},
    {"name": "qwen3.5-35b-a3b", "tier": "mid", "reasoning": False},
    {"name": "claude-opus-4-7", "tier": "frontier", "reasoning": False},
    {"name": "o4-mini", "tier": "frontier", "reasoning": True},
]

BEST_STRATEGIES = {
    "LOC": "LOC_B_reread_issue",
    "EDIT": "EDIT_A_reread_file",
    "LOGIC": "LOGIC_B_minimal_fix",
    "PLAN": "PLAN_A_step_back",
}
CONTROL = "CONTROL_no_scaffold"
SAMPLES_PER_TYPE = {"LOC": 15, "EDIT": 15, "LOGIC": 15, "PLAN": 8}

lock = threading.Lock()
all_results = []
call_count = [0]


def load_scaffold(name):
    path = PROJECT_ROOT / "prompts" / "scaffolding" / f"{name}.txt"
    return path.read_text().strip() if path.exists() else f"[MISSING: {name}]"


def call_llm(model_info, prompt):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    time.sleep(DELAY)
    kwargs = {"model": model_info["name"], "messages": [{"role": "user", "content": prompt}]}
    if model_info["reasoning"]:
        kwargs["max_completion_tokens"] = 2048
    else:
        kwargs["max_tokens"] = 1024
        kwargs["temperature"] = 0
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(**kwargs)
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


def run_model(model_info, annotations, traj_map, gold_map):
    """Run best+control for one model."""
    model_results = []
    mname = model_info["name"]
    print(f"\n{'='*50}")
    print(f"Model: {mname} (tier={model_info['tier']}, reasoning={model_info['reasoning']})")
    print(f"{'='*50}")

    for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
        typed = [a for a in annotations if a['failure_type'] == ftype][:SAMPLES_PER_TYPE[ftype]]
        for strat_name in [BEST_STRATEGIES[ftype], CONTROL]:
            scaffold_text = load_scaffold(strat_name)
            scores = []
            for inst in typed:
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

                response = call_llm(model_info, prompt)
                ev = evaluate(response, gold_files, ftype)
                scores.append(ev["score"])
                model_results.append({
                    "instance_id": iid, "model": mname, "failure_type": ftype,
                    "strategy": strat_name, "eval": ev
                })

            if scores:
                label = "scaffold" if strat_name != CONTROL else "control"
                avg = sum(scores) / len(scores)
                print(f"  [{ftype}] {label}: {avg:.2f} (n={len(scores)})")

    return model_results


def main():
    print("=" * 60)
    print("Phase 6: Multi-Model Expansion (4 new models)")
    print(f"Budget: {MAX_CALLS} calls, Delay: {DELAY}s")
    print("=" * 60)

    # Load data
    with open(PROJECT_ROOT / "results" / "phase0_annotations" / "phase0_v2_annotations.json") as f:
        annotations = json.load(f)['annotations']

    from datasets import load_dataset
    swe = load_dataset('princeton-nlp/SWE-bench_Verified', split='test')
    gold_map = {i['instance_id']: set(re.findall(r'--- a/(.*?)\n', i['patch'])) for i in swe}

    ds = load_dataset('AlexCuadron/SWE-Bench-Verified-O1-reasoning-high-results', split='test')
    traj_map = {r['issue_name']: r['full_conversation_jsonl'] for r in ds}

    start = time.time()

    # Run each model sequentially (different rate limits per model)
    for model_info in MODELS:
        if call_count[0] >= MAX_CALLS:
            print(f"\n⚠️ Budget cap reached at {call_count[0]} calls")
            break
        results = run_model(model_info, annotations, traj_map, gold_map)
        all_results.extend(results)

        # Save incrementally after each model
        output = {
            "metadata": {"timestamp": datetime.now().isoformat(), "calls": call_count[0]},
            "results": all_results
        }
        with open(RESULTS_DIR / "results.json", 'w') as f:
            json.dump(output, f, indent=2)

    elapsed = time.time() - start

    # Final summary
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} min, {call_count[0]} calls")
    print(f"{'='*60}")

    for model_info in MODELS:
        mname = model_info["name"]
        model_data = [r for r in all_results if r["model"] == mname]
        if not model_data:
            continue
        scaff = [r for r in model_data if r["strategy"] != CONTROL]
        ctrl = [r for r in model_data if r["strategy"] == CONTROL]
        s_avg = sum(r["eval"]["score"] for r in scaff) / len(scaff) if scaff else 0
        c_avg = sum(r["eval"]["score"] for r in ctrl) / len(ctrl) if ctrl else 0
        print(f"\n{mname} ({model_info['tier']}):")
        print(f"  Scaffold: {s_avg:.2f}, Control: {c_avg:.2f}, Δ={s_avg-c_avg:+.2f}")

        for ftype in ["LOC", "EDIT", "LOGIC", "PLAN"]:
            fs = [r for r in scaff if r["failure_type"] == ftype]
            fc = [r for r in ctrl if r["failure_type"] == ftype]
            if fs and fc:
                sa = sum(r["eval"]["score"] for r in fs) / len(fs)
                ca = sum(r["eval"]["score"] for r in fc) / len(fc)
                print(f"    {ftype}: scaffold={sa:.2f}, ctrl={ca:.2f}, Δ={sa-ca:+.2f}")


if __name__ == "__main__":
    main()
