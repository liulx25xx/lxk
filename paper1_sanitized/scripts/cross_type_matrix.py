#!/usr/bin/env python3
"""
Cross-type mismatch matrix: apply each type's best strategy to all OTHER types.
Produces a 4x4 score matrix showing match vs mismatch effects.
"""
import json, os, re, time, sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
API_KEY = "<REDACTED_SECRET>"
BASE_URL = "<REDACTED_URL>"
MODEL = "gpt-4o-mini"
DELAY = 4.0

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

best_strategies = {
    'EDIT': 'EDIT_A_reread_file',
    'LOC': 'LOC_A_broaden_search',
    'LOGIC': 'LOGIC_B_minimal_fix',
    'PLAN': 'PLAN_A_step_back',
}

def call_llm(prompt):
    time.sleep(DELAY)
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role":"user","content":prompt}],
            max_tokens=1024, temperature=0)
        return resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  ERR: {e}", file=sys.stderr)
        time.sleep(8)
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=[{"role":"user","content":prompt}],
                max_tokens=1024, temperature=0)
            return resp.choices[0].message.content or ""
        except:
            return ""

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

def main():
    # Load data
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    gold_map = {a['instance_id']: a['gold_files'] for a in annots['annotations']}
    
    # Load existing results to avoid re-running
    out_path = PROJECT_ROOT / "results/cross_type_matrix.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        done_keys = set(f"{r['instance_id']}_{r['applied_strategy']}" for r in existing['results'])
        results = existing['results']
    else:
        done_keys = set()
        results = []
    
    # Load scaffold prompts
    scaffolds = {}
    for ft, strat in best_strategies.items():
        path = PROJECT_ROOT / "prompts/scaffolding" / f"{strat}.txt"
        scaffolds[strat] = path.read_text().strip()
    
    # Group instances by type
    type_instances = defaultdict(list)
    for a in annots['annotations']:
        type_instances[a['failure_type']].append(a)
    
    # For each (target_type, source_strategy) pair where source != target
    batch_done = 0
    MAX_BATCH = 12  # per run to fit timeout
    
    for target_ft in ['EDIT', 'LOC', 'LOGIC', 'PLAN']:
        for source_ft, strat in best_strategies.items():
            if source_ft == target_ft:
                continue
            
            instances = type_instances[target_ft][:30]
            for inst in instances:
                iid = inst['instance_id']
                key = f"{iid}_{strat}"
                if key in done_keys:
                    continue
                if batch_done >= MAX_BATCH:
                    break
                
                gold_files = gold_map[iid]
                prompt = f"""Expert engineer helping a failed code agent. Issue: {iid}, Failure: {target_ft}

Recovery guidance:
{scaffolds[strat]}

Files to fix: {', '.join(gold_files)}

Provide the SINGLE BEST next action (bash command or file edit). Be specific. No explanation."""
                
                response = call_llm(prompt)
                ev = evaluate(response, gold_files, target_ft)
                
                results.append({
                    "instance_id": iid,
                    "true_type": target_ft,
                    "applied_strategy": strat,
                    "source_type": source_ft,
                    "eval": ev,
                    "response": response[:150]
                })
                done_keys.add(key)
                batch_done += 1
            
            if batch_done >= MAX_BATCH:
                break
        if batch_done >= MAX_BATCH:
            break
    
    # Save
    with open(out_path, 'w') as f:
        json.dump({"n": len(results), "results": results}, f, indent=2)
    
    print(f"Saved: {len(results)} total ({batch_done} new this batch)", flush=True)

if __name__ == "__main__":
    main()
