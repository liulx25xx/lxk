#!/usr/bin/env python3
"""
Random scaffold baseline: for each instance, randomly assign a scaffold from ANY type.
If random < control, proves diagnosis is REQUIRED (blind intervention harms).
"""
import json, re, time, random, sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
client = OpenAI(api_key="<REDACTED_SECRET>", base_url="<REDACTED_URL>")
DELAY = 4.0

ALL_SCAFFOLDS = {
    'EDIT_A_reread_file': None,
    'LOC_A_broaden_search': None,
    'LOGIC_B_minimal_fix': None,
    'PLAN_A_step_back': None,
}

def load_scaffolds():
    for name in ALL_SCAFFOLDS:
        path = PROJECT_ROOT / "prompts/scaffolding" / f"{name}.txt"
        ALL_SCAFFOLDS[name] = path.read_text().strip()

def call_llm(prompt):
    time.sleep(DELAY)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role":"user","content":prompt}],
            max_tokens=1024, temperature=0)
        return resp.choices[0].message.content or ""
    except:
        time.sleep(6)
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role":"user","content":prompt}],
                max_tokens=1024, temperature=0)
            return resp.choices[0].message.content or ""
        except:
            return ""

def evaluate(response, gold_files, ftype):
    if not response:
        return {"score": 0, "file_hit": False, "actionable": False, "relevant": False}
    file_hit = any(gf.split('/')[-1] in response for gf in gold_files)
    actionable = bool(re.search(r'(cat |grep |find |python |pytest |str_replace|def |import |sed )', response))
    relevant_patterns = {
        "EDIT": r'read|cat|view|exact|character|whitespace',
        "LOC": r'search|find|grep|locate|other.*file|different',
        "LOGIC": r'test|assert|expect|return|logic|fix|edge',
        "PLAN": r'approach|strategy|instead|alternative|reconsider',
    }
    relevant = bool(re.search(relevant_patterns[ftype], response, re.I))
    return {"score": sum([file_hit, actionable, relevant]), "file_hit": file_hit,
            "actionable": actionable, "relevant": relevant}

def main():
    load_scaffolds()
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    gold_map = {a['instance_id']: a['gold_files'] for a in annots['annotations']}
    
    out_path = PROJECT_ROOT / "results/random_scaffold_baseline.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        done_ids = set(r['instance_id'] for r in existing['results'])
        results = existing['results']
    else:
        done_ids = set()
        results = []
    
    random.seed(42)
    scaffold_names = list(ALL_SCAFFOLDS.keys())
    
    batch_done = 0
    MAX_BATCH = 12
    
    for a in annots['annotations']:
        iid = a['instance_id']
        if iid in done_ids:
            continue
        if batch_done >= MAX_BATCH:
            break
        
        ft = a['failure_type']
        gold_files = gold_map[iid]
        
        # Randomly select a scaffold (may or may not match)
        chosen = random.choice(scaffold_names)
        scaffold_text = ALL_SCAFFOLDS[chosen]
        
        prompt = f"""Expert engineer helping a failed code agent. Issue: {iid}, Failure type: {ft}

Recovery guidance:
{scaffold_text}

Files to fix: {', '.join(gold_files)}

Provide the SINGLE BEST next action (bash command or file edit). Be specific."""
        
        response = call_llm(prompt)
        ev = evaluate(response, gold_files, ft)
        
        is_matched = (chosen.startswith(ft))
        
        results.append({
            "instance_id": iid,
            "failure_type": ft,
            "chosen_scaffold": chosen,
            "is_matched": is_matched,
            "eval": ev,
            "response": response[:100]
        })
        done_ids.add(iid)
        batch_done += 1
    
    with open(out_path, 'w') as f:
        json.dump({"n": len(results), "results": results}, f, indent=2)
    print(f"Saved: {len(results)} total ({batch_done} new)", flush=True)

if __name__ == "__main__":
    main()
