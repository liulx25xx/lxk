#!/usr/bin/env python3
"""
Scaffold dose-response experiment: test weak/medium/strong versions of each scaffold.
This proves the effect is not binary (scaffold vs no-scaffold) but has a gradient.
"""
import json, re, time, sys
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
client = OpenAI(api_key="<REDACTED_SECRET>", base_url="<REDACTED_URL>")
DELAY = 4.0

# Three dose levels per type
DOSES = {
    'EDIT': {
        'weak': "You might want to check the file contents.",
        'medium': "Re-read the target file before attempting the edit again.",
        'strong': "STOP. You must re-read the exact current contents of the target file using cat or a read command. Your previous edit failed because your text did not match the file exactly. Do not attempt any edit until you have refreshed your view of the file.",
    },
    'LOC': {
        'weak': "Consider whether you're looking in the right place.",
        'medium': "Re-read the issue description and identify clues about which file is responsible.",
        'strong': "STOP. You are editing the WRONG FILE. The bug is NOT in the file you are currently modifying. Go back to the issue description, trace the reported behavior to its source, and search in a completely different module.",
    },
    'LOGIC': {
        'weak': "Think carefully about your fix.",
        'medium': "Focus on making the smallest possible change that addresses the core issue.",
        'strong': "STOP. Your fix is logically incorrect. Analyze the failing test assertion carefully—what exact value does it expect? Your current approach produces the wrong result. Identify the specific computation error and fix only that.",
    },
    'PLAN': {
        'weak': "You might want to reconsider.",
        'medium': "Step back and re-read the issue. Consider whether your current approach is right.",
        'strong': "STOP. Your entire approach is wrong. The issue asks for something fundamentally different from what you are implementing. Re-read the issue from scratch, identify what is ACTUALLY being requested, and start over with a completely different strategy.",
    },
}

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
    annots = json.load(open(PROJECT_ROOT / "results/phase0_annotations/phase0_v2_annotations.json"))
    gold_map = {a['instance_id']: a['gold_files'] for a in annots['annotations']}
    
    out_path = PROJECT_ROOT / "results/dose_response.json"
    if out_path.exists():
        existing = json.load(open(out_path))
        done_keys = set(f"{r['instance_id']}_{r['dose']}" for r in existing['results'])
        results = existing['results']
    else:
        done_keys = set()
        results = []
    
    # Select 10 instances per type for dose-response
    type_instances = defaultdict(list)
    for a in annots['annotations']:
        type_instances[a['failure_type']].append(a)
    
    batch_done = 0
    MAX_BATCH = 12
    
    for ft in ['EDIT', 'LOC', 'LOGIC', 'PLAN']:
        instances = type_instances[ft][:25]
        for inst in instances:
            for dose in ['weak', 'medium', 'strong']:
                key = f"{inst['instance_id']}_{dose}"
                if key in done_keys:
                    continue
                if batch_done >= MAX_BATCH:
                    break
                
                gold_files = gold_map[inst['instance_id']]
                scaffold = DOSES[ft][dose]
                
                prompt = f"""Expert engineer helping a failed code agent. Issue: {inst['instance_id']}, Failure type: {ft}

Recovery guidance:
{scaffold}

Files to fix: {', '.join(gold_files)}

Provide the SINGLE BEST next action (bash command or file edit). Be specific."""
                
                response = call_llm(prompt)
                ev = evaluate(response, gold_files, ft)
                
                results.append({
                    "instance_id": inst['instance_id'],
                    "failure_type": ft,
                    "dose": dose,
                    "eval": ev,
                    "response": response[:100]
                })
                done_keys.add(key)
                batch_done += 1
            if batch_done >= MAX_BATCH:
                break
        if batch_done >= MAX_BATCH:
            break
    
    with open(out_path, 'w') as f:
        json.dump({"n": len(results), "results": results}, f, indent=2)
    print(f"Saved: {len(results)} total ({batch_done} new)", flush=True)

if __name__ == "__main__":
    main()
